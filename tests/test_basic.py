import neighborlist_rs
import pytest
import numpy as np

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
        neighborlist_rs.PyCell([[0,0,0],[0,0,0],[0,0,0]])

def test_build_neighborlists():
    # Large box to ensure bin size > cutoff easily
    h = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]
    cell = neighborlist_rs.PyCell(h)
    
    positions = np.array([
        [1.0, 1.0, 1.0],
        [1.5, 1.0, 1.0], # Neighbor to 0 (dist 0.5)
        [19.5, 1.0, 1.0], # Neighbor to 0 via PBC (dist 1.5)
    ], dtype=np.float64)
    
    cutoff = 2.0
    result = neighborlist_rs.build_neighborlists(cell, positions, cutoff)
    
    assert "local" in result
    local = result["local"]
    
    edge_i = local["edge_i"]
    edge_j = local["edge_j"]
    shifts = local["shift"]
    
    # Pairs: (0, 1) and (0, 2)
    assert len(edge_i) == 2
    
    found_01 = False
    found_02 = False
    
    for i, j, s in zip(edge_i, edge_j, shifts):
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            # We enforce i < j in search, so it should be (0, 1)
            assert i == 0 and j == 1
            assert np.all(s == [0, 0, 0])
            found_01 = True
        if (i == 0 and j == 2) or (i == 2 and j == 0):
            assert i == 0 and j == 2
            assert np.all(s == [-1, 0, 0])
            found_02 = True
            
    assert found_01
    assert found_02