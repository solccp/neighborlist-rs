import neighborlist_rs
import numpy as np

def test_pycell_repr():
    h = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
    cell = neighborlist_rs.PyCell(h)
    
    repr_str = repr(cell)
    print(f"Repr: {repr_str}")
    
    # Expected informative repr
    assert "PyCell" in repr_str
    assert "10" in repr_str
    assert "0" in repr_str
