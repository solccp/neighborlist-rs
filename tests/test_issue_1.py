import numpy as np
import neighborlist_rs


def test_pbc_self_interaction():
    # 1 atom at (0,0,0)
    positions = np.array([[0.0, 0.0, 0.0]])

    # Small box: 5.0
    # h matrix for PyCell: [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]
    h = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    cell = neighborlist_rs.PyCell(h, [True, True, True])

    # Large cutoff: 6.0 (captures the 6 nearest images at dist 5.0)
    cutoff = 6.0

    # The half list holds each self-image once: +S and -S are the same pair.
    # Shifts: (0, 0, 1), (0, 1, 0), (1, 0, 0)

    results = neighborlist_rs.build_neighborlists(cell, positions, cutoff)

    idx_i = results["edge_index"][0]
    shifts = results["shift"]

    print(f"Found {len(idx_i)} neighbors")
    print(f"Shifts:\n{shifts}")

    assert len(idx_i) == 3
    assert sorted(map(tuple, shifts)) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    # Verify we didn't get the self-loop (0,0,0)
    for s in shifts:
        assert not np.all(s == 0), "Found self-interaction with zero shift!"


if __name__ == "__main__":
    test_pbc_self_interaction()
