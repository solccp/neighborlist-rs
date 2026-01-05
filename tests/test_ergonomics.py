import neighborlist_rs
import numpy as np

def test_pycell_repr():
    h = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
    cell = neighborlist_rs.PyCell(h)
    
    repr_str = repr(cell)
    print(f"Repr: {repr_str}")
    
def test_pycell_from_ase():
    try:
        from ase.build import bulk
    except ImportError:
        pytest.skip("ASE not installed")
        
    atoms = bulk("Cu", "fcc", a=3.6)
    cell = neighborlist_rs.PyCell.from_ase(atoms)
    
    # ASE cell: [[0, 1.8, 1.8], [1.8, 0, 1.8], [1.8, 1.8, 0]]
    # PyCell should have the transpose (though this one is symmetric)
    # Let's check the repr
    repr_str = repr(cell)
    assert "PyCell" in repr_str
    assert "1.8" in repr_str
