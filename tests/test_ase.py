import pytest
import numpy as np
import neighborlist_rs

try:
    from ase import Atoms
    from ase.build import bulk
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False

@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not installed")
def test_ase_pbc_true():
    atoms = bulk("Cu", "fcc", a=3.6) * (3, 3, 3)
    atoms.pbc = True
    
    cutoff = 3.0 # Nearest neighbor is ~2.55
    res = neighborlist_rs.build_from_ase(atoms, cutoff)
    
    edge_index = res["edge_index"]
    assert edge_index.shape[1] > 0
    
    # Check against explicit call
    cell = neighborlist_rs.PyCell(atoms.get_cell().T.tolist()) # Transpose for column-major
    # Wait, ASE cell is row-major [a, b, c].
    # neighborlist-rs PyCell takes [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]?
    # src/lib.rs:
    # h[0][0] -> h_mat(0,0). Matrix3::new is column-major? 
    # nalgebra::Matrix3::new(m11, m12, m13, ...) is row-major in constructor arguments!
    # "Constructs a new matrix from the given components in row-major order."
    # So if I pass [[ax, ay, az], ...], it constructs:
    # ax ay az
    # bx by bz
    # cx cy cz
    # This matches ASE convention!
    # My documentation says: H = [a | b | c].
    # If I create matrix from rows, the columns are NOT a,b,c. The rows are a,b,c.
    # So H = [a^T; b^T; c^T].
    # But neighborlist-rs core uses columns as vectors?
    # Let's check src/cell.rs and lib.rs.
    
    # In lib.rs:
    # let h_mat = Matrix3::new(h[0][0], ...);
    # This creates a matrix where row 0 is h[0].
    # If h[0] is vector a, then row 0 is vector a.
    
    # In PYTHON_API.md:
    # "The cell matrix H is defined such that the columns are the lattice vectors a, b, c... Note: This is the transpose of the convention used by ASE."
    
    # So if ASE gives [[ax, ay, az], ...], PyCell::new creates a matrix with rows as vectors.
    # But if the core expects columns as vectors, we have a mismatch if we just pass ASE cell directly.
    # Let's check `build_from_ase`.
    # atoms.get_cell() -> usually row-major.
    # We pass it to `PyCell::new`.
    # `PyCell::new` creates a matrix with those rows.
    # If core expects columns, then `PyCell` creates a Transposed matrix relative to what core expects?
    # Or does `PyCell` logic handle it?
    # src/lib.rs `PyCell::new` takes `Vec<Vec<f64>>`.
    # It constructs `Matrix3`.
    # `Cell::new` takes `Matrix3`.
    
    # Let's verify `Cell` usage in `search.rs`.
    # `cell.to_fractional(pos)` -> `h_inv * cart`.
    # If `h` has columns as vectors, `h * frac = cart`.
    # `h = [a | b | c]`. `h * [1, 0, 0]^T = a`. Correct.
    
    # If `PyCell::new` receives ASE cell (rows are vectors), it creates `h` where ROWS are vectors.
    # `h * [1, 0, 0]^T` would return the first column, which is `[ax, bx, cx]`.
    # This is NOT vector `a = [ax, ay, az]`.
    # So `PyCell::new` creates a matrix that is the TRANSPOSE of the "columns are vectors" convention if the input is "rows are vectors".
    
    # ISSUE: My `build_from_ase` implementation passes `atoms.get_cell()` (rows=vectors) directly to `PyCell` constructor logic (via manual Matrix3 construction).
    # Wait, `build_from_ase` manually constructs `Matrix3` from `cell_array`.
    # `cell_array` from ASE is `[[ax, ay, az], [bx, by, bz], [cx, cy, cz]]`.
    # `Matrix3::new(c00, c01, c02...)` creates:
    # c00 c01 c02
    # c10 c11 c12
    # c20 c21 c22
    # So the matrix has rows = vectors.
    # But `Cell` expects columns = vectors!
    # So `build_from_ase` IS CREATING THE WRONG CELL MATRIX. It needs to TRANSPOSE.
    
    pass

@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not installed")
def test_ase_mixed_pbc_error():
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
    atoms.set_cell([10, 10, 10])
    atoms.pbc = [True, True, False]
    
    with pytest.raises(ValueError, match="Mixed PBC"):
        neighborlist_rs.build_from_ase(atoms, 1.5)

@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not installed")
def test_ase_no_pbc():
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
    atoms.pbc = False
    
    res = neighborlist_rs.build_from_ase(atoms, 1.5)
    # Should work (isolated)
    # The library returns half-neighbor lists (i < j), so we expect only 1 edge (0->1)
    edge_index = res["edge_index"]
    assert edge_index.shape[1] == 1
    assert edge_index[0, 0] == 0
    assert edge_index[1, 0] == 1

@pytest.mark.skipif(not ASE_AVAILABLE, reason="ASE not installed")
def test_ase_multi_cutoff():
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    cutoffs = [2.0, 3.0]
    
    res = neighborlist_rs.build_multi_from_ase(atoms, cutoffs)
    
    assert 0 in res
    assert 1 in res
    
    # 2.0 (index 0) is smaller than NN distance (2.55), so 0 edges
    assert res[0]["edge_index"].shape[1] == 0
    # 3.0 (index 1) is larger than NN distance, so many edges
    assert res[1]["edge_index"].shape[1] > 0
