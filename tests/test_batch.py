import neighborlist_rs
import numpy as np


def test_build_neighborlists_batch_basic():
    # System 1: 2 atoms, distance 1.0 (cutoff 2.0 -> 1 neighbor pair)
    pos1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # System 2: 3 atoms, linear, distances 1.0, 1.0 (cutoff 2.0 -> 3 pairs: (0,1), (1,2), (0,2)? No (0,2) is dist 2.0)
    # Wait, (0,2) is dist 2.0. If cutoff=2.0, maybe not included if strict <.
    # Let's use cutoff 2.5.
    pos2 = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0]])

    positions = np.concatenate([pos1, pos2], axis=0)
    batch = np.array([0, 0, 1, 1, 1], dtype=np.int32)

    cutoff = 2.5

    # This should fail as build_neighborlists_batch is not yet implemented
    result = neighborlist_rs.build_neighborlists_batch(positions, batch, cutoff=cutoff)

    edge_index = result["edge_index"]
    edge_i = edge_index[0]
    edge_j = edge_index[1]

    # System 1 (indices 0, 1): pair (0, 1)
    # System 2 (indices 2, 3, 4): pairs (2, 3), (3, 4), (2, 4) [dist 2.0 < 2.5]
    # Total edges: 1 + 3 = 4
    assert len(edge_i) == 4

    # Sort for verification
    pairs = sorted(zip(edge_i, edge_j))
    expected = [(0, 1), (2, 3), (2, 4), (3, 4)]
    assert pairs == expected


def test_build_neighborlists_batch_multi():
    # System 1: 2 atoms, dist 1.0
    pos1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # System 2: 2 atoms, dist 3.0
    pos2 = np.array([[10.0, 0.0, 0.0], [13.0, 0.0, 0.0]])

    positions = np.concatenate([pos1, pos2], axis=0)
    batch = np.array([0, 0, 1, 1], dtype=np.int32)

    cutoffs = [2.0, 4.0]

    # This should fail initially
    result = neighborlist_rs.build_neighborlists_batch_multi(
        positions, batch, cutoffs=cutoffs
    )

    assert 0 in result
    assert 1 in result

    # 2.0A: Only system 1 pair (0, 1)
    res2 = result[0]
    assert res2["edge_index"].shape == (2, 1)
    assert res2["edge_index"][0, 0] == 0
    assert res2["edge_index"][1, 0] == 1

    # 4.0A: System 1 pair (0, 1) and System 2 pair (2, 3)
    res4 = result[1]
    assert res4["edge_index"].shape == (2, 2)
    pairs4 = sorted(zip(res4["edge_index"][0], res4["edge_index"][1]))
    assert pairs4 == [(0, 1), (2, 3)]


def test_build_neighborlists_batch_multi_pbc():
    from ase.build import bulk

    # System 1: Si Bulk (8 atoms)
    atoms1 = bulk("Si", "diamond", a=5.43, cubic=True)
    pos1 = atoms1.get_positions()

    # System 2: 2 atoms, dist 1.0 (non-PBC)
    pos2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    positions = np.concatenate([pos1, pos2], axis=0)
    batch = np.array([0] * len(pos1) + [1] * len(pos2), dtype=np.int32)

    # Cells: (2, 3, 3)
    # Note: neighborlist-rs PyCell takes transposed ase cell (column major)
    # But build_neighborlists_batch takes the 3x3 as-is from numpy if we matched single call.
    # Actually PyCell code: h[0][0], h[0][1]... so it expects row-major input which it converts to Matrix3.
    # Our batch extraction logic: m[[0,0]], m[[0,1]]... also row-major.
    cells = np.zeros((2, 3, 3))
    cells[0] = atoms1.get_cell()[:]  # ase cell is row-major [[ax, ay, az], [bx...]]
    cells[1] = np.zeros((3, 3))  # non-PBC

    cutoffs = [2.5, 6.0]

    result = neighborlist_rs.build_neighborlists_batch_multi(
        positions, batch, cells=cells, cutoffs=cutoffs
    )

    # Verify against single calls
    for i, r in enumerate(cutoffs):
        # System 1 single
        res1 = neighborlist_rs.build_neighborlists(
            neighborlist_rs.PyCell(cells[0].T.tolist()), pos1, r
        )
        # System 2 single
        res2 = neighborlist_rs.build_neighborlists(None, pos2, r)

        # Combine manually
        ei1 = res1["edge_index"]
        ei2 = res2["edge_index"]
        expected_i = np.concatenate([ei1[0], ei2[0] + len(pos1)])
        expected_j = np.concatenate([ei1[1], ei2[1] + len(pos1)])

        # RS result
        actual_i = result[i]["edge_index"][0]
        actual_j = result[i]["edge_index"][1]

        # Compare unique pairs
        p_exp = set((min(u, v), max(u, v)) for u, v in zip(expected_i, expected_j))
        p_act = set((min(u, v), max(u, v)) for u, v in zip(actual_i, actual_j))

        assert p_exp == p_act, f"Mismatch for cutoff {r}"
        assert len(expected_i) == len(actual_i)


if __name__ == "__main__":
    test_build_neighborlists_batch_basic()
    test_build_neighborlists_batch_multi()
    test_build_neighborlists_batch_multi_pbc()
    print("All batch tests passed!")
