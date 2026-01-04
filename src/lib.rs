use pyo3::prelude::*;

pub mod cell;
pub mod search;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use crate::cell::Cell;
use crate::search::CellList;
use nalgebra::{Matrix3, Vector3};
use numpy::{PyArrayMethods, PyReadonlyArray2};
use pyo3::types::PyDict;
use tracing_subscriber::EnvFilter;

#[pyclass]
pub struct PyCell {
    pub(crate) inner: Cell,
}

#[pymethods]
impl PyCell {
    #[new]
    fn new(h: Vec<Vec<f64>>) -> PyResult<Self> {
        if h.len() != 3 || h.iter().any(|r| r.len() != 3) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cell matrix must be 3x3",
            ));
        }
        let h_mat = Matrix3::new(
            h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
        );
        let inner =
            Cell::new(h_mat).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
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
    cell: Option<&PyCell>,
    positions: PyReadonlyArray2<'_, f64>,
    cutoff: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let pos_view = positions.as_array();
    let n_atoms = pos_view.shape()[0];
    if pos_view.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Positions must be (N, 3)",
        ));
    }

    let mut pos_vec = Vec::with_capacity(n_atoms);
    let mut min_bound = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut max_bound = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

    for row in pos_view.rows() {
        let p = Vector3::new(row[0], row[1], row[2]);
        if cell.is_none() {
            if p.x < min_bound.x { min_bound.x = p.x; }
            if p.y < min_bound.y { min_bound.y = p.y; }
            if p.z < min_bound.z { min_bound.z = p.z; }
            if p.x > max_bound.x { max_bound.x = p.x; }
            if p.y > max_bound.y { max_bound.y = p.y; }
            if p.z > max_bound.z { max_bound.z = p.z; }
        }
        pos_vec.push(p);
    }

    let cell_inner = if let Some(c) = cell {
        c.inner.clone()
    } else {
        // Infer cell from positions for non-PBC
        let margin = cutoff + 1.0;
        let span = max_bound - min_bound;
        let lx = span.x + 2.0 * margin;
        let ly = span.y + 2.0 * margin;
        let lz = span.z + 2.0 * margin;
        
        let h_mat = Matrix3::new(
            lx, 0.0, 0.0,
            0.0, ly, 0.0,
            0.0, 0.0, lz,
        );
        Cell::new(h_mat).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };

    let perp = cell_inner.perpendicular_widths();
    let min_width = perp.x.min(perp.y).min(perp.z);
    // Strict inequality to be safe with rounding? <= is typically fine for MIC logic, but < is safer.
    let mic_safe = cutoff * 2.0 < min_width;

    let (edge_i, edge_j, shifts) = if n_atoms < 500 && mic_safe {
        search::brute_force_search_full(&cell_inner, &pos_vec, cutoff)
    } else {
        let cl = CellList::build(&cell_inner, &pos_vec, cutoff);

        if parallel {
            cl.par_search_optimized(&cell_inner, cutoff)
        } else {
            let neighbors = cl.search(&cell_inner, &pos_vec, cutoff);
            let n_edges = neighbors.len();
            let mut ei = Vec::with_capacity(n_edges);
            let mut ej = Vec::with_capacity(n_edges);
            let mut s = Vec::with_capacity(n_edges * 3);
            for (i, j, sx, sy, sz) in neighbors {
                ei.push(i as i64);
                ej.push(j as i64);
                s.push(sx);
                s.push(sy);
                s.push(sz);
            }
            (ei, ej, s)
        }
    };

    let dict = PyDict::new(py);
    let local = PyDict::new(py);

    let n_edges = edge_i.len();
    local.set_item("edge_i", numpy::PyArray1::from_vec(py, edge_i))?;
    local.set_item("edge_j", numpy::PyArray1::from_vec(py, edge_j))?;

    let shifts_arr = numpy::PyArray1::from_vec(py, shifts).reshape((n_edges, 3))?;
    local.set_item("shift", shifts_arr)?;

    dict.set_item("local", local)?;
    Ok(dict)
}

#[pyfunction]
fn get_num_threads() -> usize {
    rayon::current_num_threads()
}

#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    // Note: build_global can only be called once.
    // If it fails, we ignore it (already initialized).
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global();
    Ok(())
}

#[pyfunction]
fn init_logging(level: Option<String>) {
    let filter = if let Some(l) = level {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(l))
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .with_thread_ids(true)
        .try_init();
}

#[pymodule]
fn neighborlist_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCell>()?;
    m.add_function(wrap_pyfunction!(build_neighborlists, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    Ok(())
}
