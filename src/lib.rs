use pyo3::prelude::*;

pub mod cell;
pub mod search;

use crate::cell::Cell;
use crate::search::CellList;
use nalgebra::{Matrix3, Vector3};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray2};
use pyo3::types::PyDict;

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

#[pyfunction]
#[pyo3(signature = (cell, positions, cutoff, parallel=true))]
fn build_neighborlists<'py>(
    py: Python<'py>,
    cell: &PyCell,
    positions: PyReadonlyArray2<'_, f64>,
    cutoff: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let pos_view = positions.as_array();
    let n_atoms = pos_view.shape()[0];
    if pos_view.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Positions must be (N, 3)"));
    }

    // Convert NumPy array to Vec<Vector3<f64>>
    let mut pos_vec = Vec::with_capacity(n_atoms);
    for row in pos_view.rows() {
        pos_vec.push(Vector3::new(row[0], row[1], row[2]));
    }

    let cl = CellList::build(&cell.inner, &pos_vec, cutoff);
    let mut neighbors = if parallel {
        cl.par_search(&cell.inner, &pos_vec, cutoff)
    } else {
        cl.search(&cell.inner, &pos_vec, cutoff)
    };

    // Sort by i then j for determinism
    neighbors.sort_unstable_by_key(|&(i, j)| (i, j));

    // Prepare output arrays
    let n_edges = neighbors.len();
    let mut edge_i = Vec::with_capacity(n_edges);
    let mut edge_j = Vec::with_capacity(n_edges);
    let mut shifts = Vec::with_capacity(n_edges * 3);

    for (i, j) in neighbors {
        edge_i.push(i as i64);
        edge_j.push(j as i64);
        
        let (shift, _) = cell.inner.get_shift_and_displacement(&pos_vec[i], &pos_vec[j]);
        shifts.push(shift.x);
        shifts.push(shift.y);
        shifts.push(shift.z);
    }

    let dict = PyDict::new(py);
    let local = PyDict::new(py);

    local.set_item("edge_i", edge_i.into_pyarray(py))?;
    local.set_item("edge_j", edge_j.into_pyarray(py))?;
    
    let shifts_arr = numpy::PyArray1::from_vec(py, shifts)
        .reshape((n_edges, 3))?;
    local.set_item("shift", shifts_arr)?;

    dict.set_item("local", local)?;
    Ok(dict)
}

#[pymodule]
fn neighborlist_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCell>()?;
    m.add_function(wrap_pyfunction!(build_neighborlists, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
