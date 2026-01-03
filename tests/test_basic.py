import neighborlist_rs
import pytest

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
