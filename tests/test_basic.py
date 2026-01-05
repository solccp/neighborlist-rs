import neighborlist_rs
import pytest
import numpy as np
from ase.build import bulk


def test_pycell_creation():
    h = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    cell = neighborlist_rs.PyCell(h)
    assert cell is not None


def test_pycell_wrap():
    h = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    cell = neighborlist_rs.PyCell(h)

    pos = [15.0, -2.0, 8.0]
    wrapped = cell.wrap(pos)

    assert wrapped[0] == pytest.approx(5.0)
    assert wrapped[1] == pytest.approx(8.0)
    assert wrapped[2] == pytest.approx(8.0)


def test_pycell_invalid():
    with pytest.raises(ValueError):
        neighborlist_rs.PyCell([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_build_neighborlists():
    # Large box to ensure bin size > cutoff easily
    h = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]
    cell = neighborlist_rs.PyCell(h)

    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],  # Neighbor to 0 (dist 0.5)
            [19.5, 1.0, 1.0],  # Neighbor to 0 via PBC (dist 1.5)
        ],
        dtype=np.float64,
    )

    cutoff = 2.0
    result = neighborlist_rs.build_neighborlists(cell, positions, cutoff)

    assert "edge_index" in result
    edge_index = result["edge_index"]
    shifts = result["shift"]

    # Pairs: (0, 1) and (0, 2)
    assert edge_index.shape[1] == 2

    found_01 = False
    found_02 = False

    for k in range(edge_index.shape[1]):
        i, j = edge_index[:, k]
        s = shifts[k]
        if i == 0 and j == 1:
            assert np.all(s == [0, 0, 0])
            found_01 = True
        if i == 0 and j == 2:
            assert np.all(s == [-1, 0, 0])
            found_02 = True

    assert found_01
    assert found_02


def test_large_system():
    # 1000 atoms in a 20x20x20 box
    box_size = 20.0
    h = [[box_size, 0.0, 0.0], [0.0, box_size, 0.0], [0.0, 0.0, box_size]]
    cell = neighborlist_rs.PyCell(h)

    rng = np.random.default_rng(42)
    positions = rng.random((1000, 3)) * box_size

    cutoff = 3.5
    result = neighborlist_rs.build_neighborlists(cell, positions, cutoff)

    edge_index = result["edge_index"]
    edge_i = edge_index[0]
    edge_j = edge_index[1]

    assert len(edge_i) > 0
    # Basic check for i < j (since we enforce it in search and sort by i then j)
    assert np.all(edge_i < edge_j)
    # Check for no self-interactions
    assert not np.any(edge_i == edge_j)


def test_silicon_bulk():
    # Silicon diamond structure, cubic cell
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    pos = atoms.get_positions()
    h_T = atoms.get_cell()[:].T.tolist()

    cell = neighborlist_rs.PyCell(h_T)
    # 1st neighbor distance is a * sqrt(3) / 4 = 5.43 * 0.433 = 2.35
    cutoff = 2.5

    result = neighborlist_rs.build_neighborlists(cell, pos, cutoff)
    edge_i = result["edge_index"][0]

    # Each Si atom has 4 neighbors
    # Total edges (undirected) = 8 * 4 / 2 = 16
    assert len(edge_i) == 16


def test_build_neighborlists_no_cell():
    # Two atoms close to each other
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    cutoff = 2.0
    # cell=None should trigger auto-box
    result = neighborlist_rs.build_neighborlists(None, positions, cutoff)

    edge_index = result["edge_index"]
    edge_i = edge_index[0]
    edge_j = edge_index[1]
    shifts = result["shift"]

    assert len(edge_i) == 1
    assert edge_i[0] == 0
    assert edge_j[0] == 1
    # No PBC shifts expected for auto-boxed isolated system
    assert np.all(shifts[0] == [0, 0, 0])


def test_build_neighborlists_multi():
    box_size = 10.0
    h = [[box_size, 0.0, 0.0], [0.0, box_size, 0.0], [0.0, 0.0, box_size]]
    cell = neighborlist_rs.PyCell(h)

    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],  # Dist 1.0
            [5.0, 1.0, 1.0],  # Dist 4.0
        ],
        dtype=np.float64,
    )

    cutoffs = [2.0, 5.0]
    result = neighborlist_rs.build_neighborlists_multi(cell, positions, cutoffs)

    assert 0 in result
    assert 1 in result

    # Cutoff 2.0 (index 0): only (0, 1)
    res2 = result[0]
    assert "edge_index" in res2
    ei2 = res2["edge_index"]
    assert ei2.shape == (2, 1)
    assert ei2[0, 0] == 0
    assert ei2[1, 0] == 1

    # Cutoff 5.0 (index 1): (0, 1) and (0, 2) and (1, 2)
    # (1, 2) distance is 3.0
    res5 = result[1]
    assert res5["edge_index"].shape == (2, 3)

    # Verify correctness against single calls
    for i, r in enumerate(cutoffs):
        single = neighborlist_rs.build_neighborlists(cell, positions, r)
        multi = result[i]

        # Sort for comparison
        si, sj = single["edge_index"][0], single["edge_index"][1]
        mi, mj = multi["edge_index"][0], multi["edge_index"][1]

        ks = np.lexsort((sj, si))
        km = np.lexsort((mj, mi))

        np.testing.assert_array_equal(si[ks], mi[km])
        np.testing.assert_array_equal(sj[ks], mj[km])
