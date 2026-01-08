#!/usr/bin/env python3
"""Test case to verify the fix for bug_002: missing periodic images in batched mode."""

import numpy as np


def test_batched_pbc():
    """Test that build_neighborlists_batch includes periodic images correctly."""
    from neighborlist_rs import build_neighborlists_batch, build_neighborlists, PyCell

    # Setup: 2 atoms in a 5x5x5 Angstrom box (batch of 1 system)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    batch = np.array([0, 0], dtype=np.int32)  # Both atoms in system 0

    # PBC cell: 5x5x5 box
    cells = np.array(
        [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]], dtype=np.float64
    )  # Shape: (1, 3, 3)

    # Compute neighbor list with cutoff that should include periodic images
    cutoff = 6.0  # Should include images at ~4.0 Angstrom
    result_batch = build_neighborlists_batch(
        positions=positions, batch=batch, cells=cells, cutoff=cutoff, parallel=True
    )

    print("=" * 60)
    print("BATCHED API RESULTS")
    print("=" * 60)
    print(f"Edge index shape: {result_batch['edge_index'].shape}")
    print(f"Edge index:\n{result_batch['edge_index']}")
    print(f"\nEdge shift shape: {result_batch['shift'].shape}")
    print(f"Edge shift:\n{result_batch['shift']}")
    print(f"\nNumber of edges: {result_batch['edge_index'].shape[1]}")
    print(f"Non-zero shifts: {np.any(result_batch['shift'] != 0, axis=1).sum()}")

    # Compare with single-system API (should work correctly)
    # Note: PyCell expects column-major (transposed) cell matrix
    pycell = PyCell(cells[0].T.tolist())
    result_single = build_neighborlists(cell=pycell, positions=positions, cutoff=cutoff)

    print("\n" + "=" * 60)
    print("SINGLE-SYSTEM API RESULTS (for comparison)")
    print("=" * 60)
    print(f"Edge index shape: {result_single['edge_index'].shape}")
    print(f"Number of edges: {result_single['edge_index'].shape[1]}")
    print(f"Non-zero shifts: {np.any(result_single['shift'] != 0, axis=1).sum()}")

    # Validate the fix
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    n_edges_batch = result_batch["edge_index"].shape[1]
    n_nonzero_shifts_batch = np.any(result_batch["shift"] != 0, axis=1).sum()

    n_edges_single = result_single["edge_index"].shape[1]

    # Assertions
    success = True

    if n_edges_batch < 10:
        print(
            f"❌ FAIL: Expected at least 10 edges (with periodic images), got {n_edges_batch}"
        )
        success = False
    else:
        print(f"✓ PASS: Found {n_edges_batch} edges (includes periodic images)")

    if n_nonzero_shifts_batch == 0:
        print("❌ FAIL: Expected non-zero shifts, but all shifts are zero")
        success = False
    else:
        print(f"✓ PASS: Found {n_nonzero_shifts_batch} edges with non-zero shifts")

    # Compare batch vs single
    if n_edges_batch != n_edges_single:
        print(
            f"⚠ WARNING: Batch API ({n_edges_batch} edges) differs from single API ({n_edges_single} edges)"
        )
        print("  This may be acceptable if both include periodic images")
    else:
        print("✓ PASS: Batch and single API produce same number of edges")

    if success:
        print("\n" + "=" * 60)
        print("✓✓✓ BUG FIX VERIFIED - PERIODIC IMAGES ARE NOW INCLUDED ✓✓✓")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("❌❌❌ BUG STILL PRESENT - PERIODIC IMAGES MISSING ❌❌❌")
        print("=" * 60)
        return False


if __name__ == "__main__":
    try:
        success = test_batched_pbc()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
