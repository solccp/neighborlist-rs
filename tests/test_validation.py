import numpy as np
import neighborlist_rs
import pytest


def test_batch_position_shape_validation():
    # N=10, D=2 (Invalid, should be 3)
    positions = np.random.rand(10, 2)
    batch = np.zeros(10, dtype=np.int32)
    cutoff = 5.0

    with pytest.raises(ValueError, match=r"Positions must be \(N, 3\)"):
        neighborlist_rs.build_neighborlists_batch(positions, batch, None, cutoff)

    with pytest.raises(ValueError, match=r"Positions must be \(N, 3\)"):
        neighborlist_rs.build_neighborlists_batch_multi(
            positions, batch, None, [cutoff]
        )


def test_single_position_shape_validation():
    # N=10, D=4 (Invalid)
    positions = np.random.rand(10, 4)
    cutoff = 5.0

    with pytest.raises(ValueError, match=r"Positions must be \(N, 3\)"):
        neighborlist_rs.build_neighborlists(None, positions, cutoff)


def test_cutoff_validation():
    positions = np.random.rand(10, 3)

    # Negative
    with pytest.raises(ValueError, match="Cutoff must be positive and finite"):
        neighborlist_rs.build_neighborlists(None, positions, -1.0)

    # Zero
    with pytest.raises(ValueError, match="Cutoff must be positive and finite"):
        neighborlist_rs.build_neighborlists(None, positions, 0.0)

    # NaN
    with pytest.raises(ValueError, match="Cutoff must be positive and finite"):
        neighborlist_rs.build_neighborlists(None, positions, float("nan"))

    # Inf
    with pytest.raises(ValueError, match="Cutoff must be positive and finite"):
        neighborlist_rs.build_neighborlists(None, positions, float("inf"))


def test_multi_cutoff_validation():
    positions = np.random.rand(10, 3)

    # One invalid
    with pytest.raises(ValueError, match="All cutoffs must be positive and finite"):
        neighborlist_rs.build_neighborlists_multi(None, positions, [1.0, -2.0, 3.0])


def test_unsorted_batch_validation():
    positions = np.random.rand(10, 3)
    # Unsorted batch ids: [0, 0, 1, 1, 0, 0]
    batch = np.array([0, 0, 1, 1, 0, 0, 1, 1, 2, 2], dtype=np.int32)
    cutoff = 5.0

    with pytest.raises(
        ValueError, match=r"Batch IDs must be monotonic \(non-decreasing\)"
    ):
        neighborlist_rs.build_neighborlists_batch(positions, batch, None, cutoff)


def test_edge_index_dtype():
    positions = np.random.rand(10, 3)
    res = neighborlist_rs.build_neighborlists(None, positions, 5.0)
    assert res["edge_index"].dtype == np.int64
