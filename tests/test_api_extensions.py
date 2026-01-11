import neighborlist_rs
import pytest
import numpy as np
from ase import Atoms


def test_global_config_threads():
    # Initial value
    initial_threads = neighborlist_rs.get_num_threads()
    assert initial_threads > 0

    # Try setting to a different value
    # Note: Rayon global pool can only be initialized once.
    neighborlist_rs.set_num_threads(initial_threads + 1)

    new_threads = neighborlist_rs.get_num_threads()
    assert new_threads > 0


def test_global_config_thresholds():
    # Brute force threshold
    initial_bf = neighborlist_rs.get_brute_force_threshold()
    neighborlist_rs.set_brute_force_threshold(1234)
    assert neighborlist_rs.get_brute_force_threshold() == 1234
    neighborlist_rs.set_brute_force_threshold(initial_bf)

    # Parallel threshold
    initial_parallel = neighborlist_rs.get_parallel_threshold()
    neighborlist_rs.set_parallel_threshold(567)
    assert neighborlist_rs.get_parallel_threshold() == 567
    neighborlist_rs.set_parallel_threshold(initial_parallel)

    # Stack threshold
    initial_stack = neighborlist_rs.get_stack_threshold()
    neighborlist_rs.set_stack_threshold(890)
    assert neighborlist_rs.get_stack_threshold() == 890
    neighborlist_rs.set_stack_threshold(initial_stack)


def test_init_logging():
    # This just checks it doesn't crash
    neighborlist_rs.init_logging("debug")
    neighborlist_rs.init_logging("info")
    neighborlist_rs.init_logging(None)


def test_multi_cutoff_with_labels():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=np.float64
    )
    cutoffs = [1.6, 3.0]
    labels = ["short", "long"]

    # Single system
    res = neighborlist_rs.build_neighborlists_multi(
        None, positions, cutoffs, labels=labels
    )
    assert "short" in res
    assert "long" in res
    assert res["short"]["edge_index"].shape[1] == 2  # (0,1) and (1,2)
    assert res["long"]["edge_index"].shape[1] == 3  # (0,1), (1,2), (0,2)

    # Batched system
    batch = np.array([0, 0, 0], dtype=np.int32)
    res_batch = neighborlist_rs.build_neighborlists_batch_multi(
        positions, batch, cutoffs=cutoffs, labels=labels
    )
    assert "short" in res_batch
    assert "long" in res_batch
    assert res_batch["short"]["edge_index"].shape[1] == 2
    assert res_batch["long"]["edge_index"].shape[1] == 3


def test_multi_cutoff_disjoint():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=np.float64
    )
    cutoffs = [1.6, 3.0]

    # Single system
    res = neighborlist_rs.build_neighborlists_multi(
        None, positions, cutoffs, disjoint=True
    )
    # 0: [0, 1.6] -> (0,1) and (1,2)
    assert res[0]["edge_index"].shape[1] == 2
    # 1: (1.6, 3.0] -> (0,2)
    assert res[1]["edge_index"].shape[1] == 1
    assert res[1]["edge_index"][0, 0] == 0
    assert res[1]["edge_index"][1, 0] == 2


def test_batch_labels_mismatch():
    positions = np.random.rand(10, 3)
    batch = np.zeros(10, dtype=np.int32)
    cutoffs = [1.0, 2.0]
    labels = ["one"]  # Mismatch

    with pytest.raises(
        ValueError, match="Length of labels must match length of cutoffs"
    ):
        neighborlist_rs.build_neighborlists_batch_multi(
            positions, batch, cutoffs=cutoffs, labels=labels
        )


def test_batch_all_isolated():
    # 10 atoms, each in its own batch
    positions = np.random.rand(10, 3) * 100.0  # Spread them out
    batch = np.arange(10, dtype=np.int32)
    cutoff = 1.0

    res = neighborlist_rs.build_neighborlists_batch(positions, batch, cutoff=cutoff)
    assert res["edge_index"].shape[1] == 0


def test_batch_single_large_batch():
    # 10 atoms, all in one batch
    positions = np.array([[i, 0, 0] for i in range(10)], dtype=np.float64)
    batch = np.zeros(10, dtype=np.int32)
    cutoff = 1.5

    res = neighborlist_rs.build_neighborlists_batch(positions, batch, cutoff=cutoff)
    # Pairs (0,1), (1,2), ..., (8,9) -> 9 edges
    assert res["edge_index"].shape[1] == 9


def test_ase_build_multi():
    atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [2.5, 0, 0]])
    cutoffs = [1.6, 3.0]
    labels = ["s", "l"]

    res = neighborlist_rs.build_multi_from_ase(atoms, cutoffs, labels=labels)
    assert "s" in res
    assert "l" in res
    assert res["s"]["edge_index"].shape[1] == 2
    assert res["l"]["edge_index"].shape[1] == 3


def test_empty_cutoffs():
    positions = np.random.rand(5, 3)
    res = neighborlist_rs.build_neighborlists_multi(None, positions, [])
    assert len(res) == 0

    batch = np.zeros(5, dtype=np.int32)
    res_batch = neighborlist_rs.build_neighborlists_batch_multi(
        positions, batch, cutoffs=[]
    )
    assert len(res_batch) == 0


def test_batch_mixed_pbc():
    # System 0: PBC
    pos0 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    cell0 = np.eye(3) * 10.0

    # System 1: Non-PBC (zero cell)
    pos1 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    cell1 = np.zeros((3, 3))

    positions = np.concatenate([pos0, pos1])
    batch = np.array([0, 0, 1, 1], dtype=np.int32)
    cells = np.stack([cell0, cell1], dtype=np.float64)

    res = neighborlist_rs.build_neighborlists_batch(
        positions, batch, cells=cells, cutoff=1.5
    )
    assert res["edge_index"].shape[1] == 2  # One pair in each system

    # Verify shifts
    shifts = res["shift"]
    assert np.all(shifts == 0)


def test_batch_too_few_cells():
    positions = np.random.rand(4, 3)
    batch = np.array([0, 0, 1, 1], dtype=np.int32)
    cells = np.zeros((1, 3, 3))  # Only 1 cell for 2 systems

    with pytest.raises(ValueError, match="Expected at least 2 cells, but got 1"):
        neighborlist_rs.build_neighborlists_batch(
            positions, batch, cells=cells, cutoff=1.0
        )


def test_batch_large_n_systems():
    n_systems = 100
    atoms_per_system = 2
    total_atoms = n_systems * atoms_per_system

    positions = np.random.rand(total_atoms, 3)
    # Ensure they are close enough to have neighbors
    for i in range(n_systems):
        positions[i * 2 + 1] = positions[i * 2] + np.array([0.5, 0, 0])

    batch = np.repeat(np.arange(n_systems, dtype=np.int32), atoms_per_system)

    res = neighborlist_rs.build_neighborlists_batch(positions, batch, cutoff=1.0)
    assert res["edge_index"].shape[1] == n_systems


def test_pycell_repr_pbc():
    h = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    cell = neighborlist_rs.PyCell(h, pbc=[True, False, True])
    r = repr(cell)
    assert "pbc=[true, false, true]" in r.lower()


def test_multi_cutoff_large_n():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    cutoffs = [0.1 * i for i in range(1, 21)]  # 20 cutoffs

    res = neighborlist_rs.build_neighborlists_multi(None, positions, cutoffs)
    assert len(res) == 20
    assert res[10]["edge_index"].shape[1] == 1


def test_batch_all_atoms_in_one_batch_with_cells():
    positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    batch = np.array([0, 0], dtype=np.int32)
    cells = np.array([[[10, 0, 0], [0, 10, 0], [0, 0, 10]]], dtype=np.float64)

    res = neighborlist_rs.build_neighborlists_batch(
        positions, batch, cells=cells, cutoff=1.5
    )
    assert res["edge_index"].shape[1] == 1


def test_build_neighborlists_empty_atoms():
    pos = np.zeros((0, 3), dtype=np.float64)
    res = neighborlist_rs.build_neighborlists(None, pos, 1.0)
    assert res["edge_index"].shape == (2, 0)
    assert res["shift"].shape == (0, 3)

    res_m = neighborlist_rs.build_neighborlists_multi(None, pos, [1.0, 2.0])
    assert len(res_m) == 2
    assert res_m[0]["edge_index"].shape == (2, 0)

    batch = np.zeros(0, dtype=np.int32)
    res_b = neighborlist_rs.build_neighborlists_batch(pos, batch, cutoff=1.0)
    assert res_b["edge_index"].shape == (2, 0)


def test_mixed_pbc_logic():
    h = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    positions = np.array([[5.0, 5.0, 0.5], [5.0, 5.0, 9.5]], dtype=np.float64)

    cell = neighborlist_rs.PyCell(h, pbc=[True, True, False])
    res = neighborlist_rs.build_neighborlists(cell, positions, 2.0)
    assert res["edge_index"].shape[1] == 0

    cell_pbc = neighborlist_rs.PyCell(h, pbc=[True, True, True])
    res_pbc = neighborlist_rs.build_neighborlists(cell_pbc, positions, 2.0)
    assert res_pbc["edge_index"].shape[1] == 1
    assert np.all(res_pbc["shift"][0] == [0, 0, -1])
