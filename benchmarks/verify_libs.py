import numpy as np
from ase.build import bulk
from ase.neighborlist import neighbor_list
import matscipy.neighbours
import freud
import vesin
import neighborlist_rs

# torch_nl might need specific import
try:
    from torch_nl import compute_neighborlist
except ImportError:
    compute_neighborlist = None


def get_ase_neighbors(atoms, cutoff):
    i, j, S = neighbor_list("ijS", atoms, cutoff)
    return i, j, S


def get_matscipy_neighbors(atoms, cutoff):
    i, j, S = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)
    return i, j, S


def get_freud_neighbors(atoms, cutoff):
    box = freud.box.Box.from_matrix(atoms.get_cell().array)
    points = atoms.get_positions()
    aq = freud.locality.AABBQuery(box, points)
    # exclude_ii=True to match ASE's default of no self-interaction typically unless specified
    # ASE 'ij' usually excludes self unless specified? No, 'ij' includes self if i==j is within cutoff?
    # Actually ASE neighbor_list does NOT include self-edges by default for r>0.

    nlist = aq.query(points, dict(r_max=cutoff, exclude_ii=True)).toNeighborList()

    # Freud returns (query_point_index, point_index) -> (i, j)
    # And images.
    return nlist.query_point_indices, nlist.point_indices, nlist.neighbor_counts


def get_vesin_neighbors(atoms, cutoff):
    # vesin 0.4.2
    # Init: cutoff, full_list
    nl = vesin.NeighborList(cutoff=cutoff, full_list=True)

    box = atoms.get_cell().array
    points = atoms.get_positions()

    # Compute: points, box, periodic
    # Returns quantities specified (ijS default? No, strictly what is asked)
    # help says: quantities can contain "i", "j", "S"
    # return is List[numpy.ndarray]
    i, j, S = nl.compute(points, box=box, periodic=True, quantities="ijS")
    return i, j, S


def get_neighborlist_rs_neighbors(atoms, cutoff):
    cell = neighborlist_rs.PyCell(atoms.get_cell().array)
    pos = atoms.get_positions()
    res = neighborlist_rs.build_neighborlists(cell, pos, cutoff)
    edge_index = res["edge_index"]
    # edge_index is (2, E)
    return edge_index[0], edge_index[1], res["shift"]


def to_canonical_half_list(i, j):
    """
    Convert neighbor list to a canonical half-list (undirected edges).
    1. For every pair (u, v), ensure u < v.
    2. Remove duplicates.
    3. Sort.
    """
    if len(i) == 0:
        return np.array([]), np.array([])

    # Stack to (N, 2)
    pairs = np.vstack((i, j)).T

    # Sort each row so that col 0 is min, col 1 is max
    pairs = np.sort(pairs, axis=1)

    # Unique rows
    pairs = np.unique(pairs, axis=0)

    # Return split
    return pairs[:, 0], pairs[:, 1]


def verify():
    print("Setting up system...")
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms = atoms.repeat((3, 3, 3))
    atoms.rattle(stdev=0.01)
    cutoff = 3.0

    print(f"System: {len(atoms)} atoms, Cutoff: {cutoff}")

    results = {}

    # ASE
    print("Running ASE...")
    i, j, S = get_ase_neighbors(atoms, cutoff)
    i_ase, j_ase = to_canonical_half_list(i, j)
    results["ase"] = (i_ase, j_ase)
    print(f"ASE (Half-list): {len(i_ase)} edges")

    # Neighborlist-rs
    print("Running Neighborlist-rs...")
    i_rs, j_rs, S_rs = get_neighborlist_rs_neighbors(atoms, cutoff)
    i_rs_half, j_rs_half = to_canonical_half_list(i_rs, j_rs)
    results["rs"] = (i_rs_half, j_rs_half)
    print(f"RS (Half-list): {len(i_rs_half)} edges")

    # Compare RS vs ASE
    assert len(results["ase"][0]) == len(results["rs"][0]), (
        f"Count mismatch: ASE={len(results['ase'][0])}, RS={len(results['rs'][0])}"
    )

    np.testing.assert_array_equal(
        results["ase"][0], results["rs"][0], err_msg="Source indices mismatch"
    )
    np.testing.assert_array_equal(
        results["ase"][1], results["rs"][1], err_msg="Target indices mismatch"
    )
    print("ASE vs Neighborlist-rs: MATCH")

    # Matscipy
    try:
        print("Running Matscipy...")
        i_ms, j_ms, S_ms = get_matscipy_neighbors(atoms, cutoff)
        i_ms_half, j_ms_half = to_canonical_half_list(i_ms, j_ms)
        assert len(i_ms_half) == len(i_ase)
        np.testing.assert_array_equal(i_ms_half, i_ase)
        np.testing.assert_array_equal(j_ms_half, j_ase)
        print("Matscipy: MATCH")
    except Exception as e:
        print(f"Matscipy failed: {e}")

    # Vesin
    try:
        print("Running Vesin...")
        i_vs, j_vs, S_vs = get_vesin_neighbors(atoms, cutoff)
        i_vs_half, j_vs_half = to_canonical_half_list(i_vs, j_vs)

        if len(i_vs_half) != len(i_ase):
            print(f"Vesin count mismatch: {len(i_vs_half)} vs {len(i_ase)}")
        else:
            np.testing.assert_array_equal(i_vs_half, i_ase)
            np.testing.assert_array_equal(j_vs_half, j_ase)
            print("Vesin: MATCH")
    except Exception as e:
        print(f"Vesin failed: {e}")

    # Freud
    try:
        print("Running Freud...")
        i_fr, j_fr, _ = get_freud_neighbors(atoms, cutoff)
        i_fr_half, j_fr_half = to_canonical_half_list(i_fr, j_fr)

        if len(i_fr_half) != len(i_ase):
            print(f"Freud count mismatch: {len(i_fr_half)} vs {len(i_ase)}")
        else:
            np.testing.assert_array_equal(i_fr_half, i_ase)
            np.testing.assert_array_equal(j_fr_half, j_ase)
            print("Freud: MATCH")
    except Exception as e:
        print(f"Freud failed: {e}")


if __name__ == "__main__":
    verify()
