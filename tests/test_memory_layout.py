import numpy as np
import neighborlist_rs
import pytest


def test_contiguous_vs_non_contiguous():
    # Create a system
    N = 100
    positions = np.random.rand(N, 3) * 10.0
    cutoff = 1.5

    # 1. Contiguous (Standard)
    positions_c = np.ascontiguousarray(positions)
    res_c = neighborlist_rs.build_neighborlists(None, positions_c, cutoff)

    # 2. Non-contiguous (Sliced from larger array)
    # Create (N, 4) and slice first 3 columns
    large = np.zeros((N, 4))
    large[:, :3] = positions
    positions_nc = large[:, :3]

    assert not positions_nc.flags["C_CONTIGUOUS"]

    res_nc = neighborlist_rs.build_neighborlists(None, positions_nc, cutoff)

    # Compare results
    np.testing.assert_array_equal(res_c["edge_index"], res_nc["edge_index"])
    np.testing.assert_array_equal(res_c["shift"], res_nc["shift"])


def test_batch_memory_layout():
    # Test for batch processing too
    N = 50
    positions = np.random.rand(N, 3) * 10.0
    batch = np.zeros(N, dtype=np.int32)  # Single batch
    cutoff = 1.5

    # Non-contiguous positions
    large = np.zeros((N, 4))
    large[:, :3] = positions
    positions_nc = large[:, :3]

    # Non-contiguous batch (strided)
    batch_large = np.zeros(N * 2, dtype=np.int32)
    batch_large[::2] = batch
    batch_nc = batch_large[::2]

    assert not positions_nc.flags["C_CONTIGUOUS"]
    assert not batch_nc.flags[
        "C_CONTIGUOUS"
    ]  # Might be depending on numpy version, but let's try

    # Note: The current Rust code forces batch to be contiguous via `.as_slice()`.
    # If it fails, it raises ValueError.
    # Let's check if numpy handles the conversion or if Rust needs to.
    # PyReadonlyArray1 can handle non-contiguous, but .as_slice() returns None if not contiguous.
    # We should ensure our new code handles this or explicitly requires contiguous.

    try:
        res = neighborlist_rs.build_neighborlists_batch(
            positions_nc, batch_nc, None, cutoff
        )
        assert len(res["edge_index"][0]) >= 0
    except ValueError as e:
        # If it fails currently, that's a baseline behavior to note.
        # But ideally we want it to work.
        pytest.fail(f"Failed on non-contiguous inputs: {e}")
