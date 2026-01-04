import neighborlist_rs
import numpy as np
import pytest

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
    
    local = result["local"]
    edge_i = local["edge_i"]
    edge_j = local["edge_j"]
    
    # System 1 (indices 0, 1): pair (0, 1)
    # System 2 (indices 2, 3, 4): pairs (2, 3), (3, 4), (2, 4) [dist 2.0 < 2.5]
    # Total edges: 1 + 3 = 4
    assert len(edge_i) == 4
    
    # Sort for verification
    pairs = sorted(zip(edge_i, edge_j))
    expected = [(0, 1), (2, 3), (2, 4), (3, 4)]
    assert pairs == expected

if __name__ == "__main__":
    test_build_neighborlists_batch_basic()
