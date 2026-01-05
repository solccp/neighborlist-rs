import pytest
import numpy as np
import neighborlist_rs

try:
    from ase import Atoms
    from ase.build import bulk

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not installed")
def test_ase_pbc_true():
    # Simple cubic cell
    a = 3.0
    atoms = Atoms("H", positions=[[0.1, 0.1, 0.1]], cell=[a, a, a], pbc=True)

    # Mirror image at [3.1, 0.1, 0.1]
    # Distance should be 3.0.
    # Let's add another atom close to the boundary
    atoms.extend(Atoms("H", positions=[[2.9, 0.1, 0.1]]))

    cutoff = 1.0  # Distance is 0.2 across boundary
    res = neighborlist_rs.build_from_ase(atoms, cutoff)

    edge_index = res["edge_index"]
    shifts = res["shift"]

    # 0 and 1 are 2.8 apart in cell, but 0.2 apart via PBC
    # (0.1, 0.1, 0.1) and (2.9, 0.1, 0.1)
    # Displacement: (2.8, 0, 0)
    # Minimum image: (2.8 - 3.0, 0, 0) = (-0.2, 0, 0)
    # Shift: [-1, 0, 0] for atom 1 relative to 0?
    # Or [1, 0, 0] for atom 0 relative to 1?
    # Our convention: r_j_image = r_j + shift * cell
    # r_0_image = r_0 + [1, 0, 0] * 3.0 = [3.1, 0.1, 0.1]
    # Dist(r_0_image, r_1) = dist([3.1, 0.1, 0.1], [2.9, 0.1, 0.1]) = 0.2 < 1.0

    assert edge_index.shape[1] == 1
    # i < j requirement means edge is (0, 1) or (1, 0)?
    # Rust core usually returns i < j for single system.
    # If i=0, j=1:
    # r_1_image = r_1 + shift * cell.
    # shift = [-1, 0, 0] -> r_1_img = [2.9-3, 0.1, 0.1] = [-0.1, 0.1, 0.1]
    # dist(r_0, r_1_img) = dist([0.1...], [-0.1...]) = 0.2. Correct.

    assert edge_index[0, 0] == 0
    assert edge_index[1, 0] == 1
    np.testing.assert_array_equal(shifts[0], [-1, 0, 0])

    # Check against explicit PyCell call
    # PyCell expects columns as vectors. ASE cell is [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]
    # So we MUST transpose ASE cell for PyCell.
    h_T = atoms.get_cell().T.tolist()
    cell = neighborlist_rs.PyCell(h_T)
    res_manual = neighborlist_rs.build_neighborlists(
        cell, atoms.get_positions(), cutoff
    )

    np.testing.assert_array_equal(res["edge_index"], res_manual["edge_index"])
    np.testing.assert_array_equal(res["shift"], res_manual["shift"])


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not installed")
def test_ase_mixed_pbc_success():
    # 2D periodicity: periodic in X, Y; non-periodic in Z
    atoms = Atoms("H2", positions=[[0.1, 0.1, 0.1], [0.1, 0.1, 9.9]])
    atoms.set_cell([10, 10, 10])
    atoms.pbc = [True, True, False]

    # Across Z boundary (9.9 - 0.1 = 9.8). If periodic in Z, it would be 0.2.
    # Since NOT periodic in Z, distance is 9.8 > 1.5.
    res = neighborlist_rs.build_from_ase(atoms, 1.5)
    assert res["edge_index"].shape[1] == 0

    # Across X boundary (0.1 and 9.9). Periodic in X.
    atoms.positions[1] = [9.9, 0.1, 0.1]
    res = neighborlist_rs.build_from_ase(atoms, 1.5)
    assert res["edge_index"].shape[1] == 1


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not installed")
def test_ase_no_pbc():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]])
    atoms.pbc = False

    res = neighborlist_rs.build_from_ase(atoms, 1.5)
    # Should work (isolated)
    # The library returns half-neighbor lists (i < j), so we expect only 1 edge (0->1)
    edge_index = res["edge_index"]
    assert edge_index.shape[1] == 1
    assert edge_index[0, 0] == 0
    assert edge_index[1, 0] == 1


@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not installed")
def test_ase_multi_cutoff():
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    cutoffs = [2.0, 3.0]

    res = neighborlist_rs.build_multi_from_ase(atoms, cutoffs)

    assert 0 in res
    assert 1 in res

    # 2.0 (index 0) is smaller than NN distance (2.55), so 0 edges
    assert res[0]["edge_index"].shape[1] == 0
    # 3.0 (index 1) is larger than NN distance, so many edges
    assert res[1]["edge_index"].shape[1] > 0
