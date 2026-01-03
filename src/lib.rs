use pyo3::prelude::*;

pub mod cell;
pub mod search;

use crate::cell::Cell;
use nalgebra::{Matrix3, Vector3};

#[pyclass]
pub struct PyCell {
    pub(crate) inner: Cell,
}

#[pymethods]
impl PyCell {
    #[new]
    fn new(h: Vec<Vec<f64>>) -> PyResult<Self> {
        if h.len() != 3 || h.iter().any(|r| r.len() != 3) {
            return Err(pyo3::exceptions::PyValueError::new_err("Cell matrix must be 3x3"));
        }
        let h_mat = Matrix3::new(
            h[0][0], h[0][1], h[0][2],
            h[1][0], h[1][1], h[1][2],
            h[2][0], h[2][1], h[2][2],
        );
        let inner = Cell::new(h_mat).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyCell { inner })
    }

    fn wrap(&self, pos: [f64; 3]) -> [f64; 3] {
        let p = Vector3::from_column_slice(&pos);
        let wrapped = self.inner.wrap(&p);
        [wrapped.x, wrapped.y, wrapped.z]
    }
}

#[pymodule]
fn neighborlist_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCell>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
