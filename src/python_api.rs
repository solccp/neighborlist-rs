use crate::batch;
use crate::cell::Cell;
use crate::config;
use crate::single;
use nalgebra::{Matrix3, Vector3};
use numpy::{
    PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
    ndarray,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::borrow::Cow;
use tracing_subscriber::EnvFilter;

#[pyclass]
pub struct PyCell {
    pub(crate) inner: Cell,
}

#[pymethods]
impl PyCell {
    #[new]
    #[pyo3(signature = (h, pbc=None))]
    fn new(h: Vec<Vec<f64>>, pbc: Option<[bool; 3]>) -> PyResult<Self> {
        if h.len() != 3 || h.iter().any(|r| r.len() != 3) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cell matrix must be 3x3",
            ));
        }
        let h_mat = Matrix3::new(
            h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
        );
        let pbc_vec = if let Some(p) = pbc {
            Vector3::new(p[0], p[1], p[2])
        } else {
            Vector3::new(true, true, true)
        };
        let inner = Cell::new(h_mat, pbc_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyCell { inner })
    }

    fn wrap(&self, pos: [f64; 3]) -> [f64; 3] {
        let p = Vector3::from_column_slice(&pos);
        let wrapped = self.inner.wrap(&p);
        [wrapped.x, wrapped.y, wrapped.z]
    }

    fn __repr__(&self) -> String {
        let h = self.inner.h();
        let p = self.inner.pbc();
        format!(
            "PyCell(h=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]], pbc=[{}, {}, {}])",
            h[(0, 0)],
            h[(0, 1)],
            h[(0, 2)],
            h[(1, 0)],
            h[(1, 1)],
            h[(1, 2)],
            h[(2, 0)],
            h[(2, 1)],
            h[(2, 2)],
            p.x,
            p.y,
            p.z
        )
    }

    #[staticmethod]
    fn from_ase(atoms: &Bound<'_, PyAny>) -> PyResult<Self> {
        let cell_obj = atoms.call_method0("get_cell")?;
        let cell_array_obj = cell_obj.call_method0("__array__")?;
        let cell_array: PyReadonlyArray2<f64> = cell_array_obj.extract()?;
        let c = cell_array.as_array();
        if c.shape() != [3, 3] {
            return Err(pyo3::exceptions::PyValueError::new_err("Cell must be 3x3"));
        }

        let pbc_obj = atoms.call_method0("get_pbc")?;
        let pbc: [bool; 3] = pbc_obj.extract()?;
        let pbc_vec = Vector3::new(pbc[0], pbc[1], pbc[2]);

        // Transpose ASE (row-major) to internal (column-major)
        let h_mat = Matrix3::new(
            c[[0, 0]],
            c[[1, 0]],
            c[[2, 0]],
            c[[0, 1]],
            c[[1, 1]],
            c[[2, 1]],
            c[[0, 2]],
            c[[1, 2]],
            c[[2, 2]],
        );
        let inner = Cell::new(h_mat, pbc_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyCell { inner })
    }
}

enum PositionData<'a> {
    Slice(&'a [Vector3<f64>]),
    Owned(Vec<Vector3<f64>>),
}

impl<'a> std::ops::Deref for PositionData<'a> {
    type Target = [Vector3<f64>];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Slice(s) => s,
            Self::Owned(v) => v,
        }
    }
}

fn get_positions<'a>(positions: &'a PyReadonlyArray2<'a, f64>) -> PyResult<PositionData<'a>> {
    if positions.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Positions must be (N, 3)",
        ));
    }

    if let Ok(slice) = positions.as_slice()
        && let Ok(cast_slice) = bytemuck::try_cast_slice(slice)
    {
        return Ok(PositionData::Slice(cast_slice));
    }

    // Fallback: Copy
    let pos_view = positions.as_array();
    let n_atoms = pos_view.shape()[0];
    let mut pos_vec = Vec::with_capacity(n_atoms);
    for row in pos_view.rows() {
        pos_vec.push(Vector3::new(row[0], row[1], row[2]));
    }
    Ok(PositionData::Owned(pos_vec))
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
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cutoff must be positive and finite",
        ));
    }
    let pos_data = get_positions(&positions)?;
    let n_atoms = pos_data.len();

    if n_atoms == 0 {
        let dict = PyDict::new(py);
        dict.set_item(
            "edge_index",
            numpy::PyArray1::from_vec(py, vec![0i64; 0]).reshape((2, 0))?,
        )?;
        dict.set_item(
            "shift",
            numpy::PyArray1::from_vec(py, vec![0i32; 0]).reshape((0, 3))?,
        )?;
        return Ok(dict);
    }

    let cell_info = cell.map(|c| (*c.inner.h(), *c.inner.pbc()));

    let (mut edge_i, edge_j, shifts) =
        single::search_single(&pos_data, cell_info, cutoff, parallel)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let dict = PyDict::new(py);

    let n_edges = edge_i.len();
    edge_i.extend(edge_j);
    let edge_index = numpy::PyArray1::from_vec(py, edge_i).reshape((2, n_edges))?;
    dict.set_item("edge_index", edge_index)?;

    let shifts_arr = numpy::PyArray1::from_vec(py, shifts).reshape((n_edges, 3))?;
    dict.set_item("shift", shifts_arr)?;

    Ok(dict)
}

fn extract_ase_data<'py>(
    atoms: &Bound<'py, PyAny>,
) -> PyResult<(PyReadonlyArray2<'py, f64>, Option<PyCell>)> {
    // 1. Extract positions
    let pos_obj = atoms.call_method0("get_positions")?;
    let positions: PyReadonlyArray2<f64> = pos_obj.extract()?;

    // 2. Extract PBC
    let pbc_obj = atoms.call_method0("get_pbc")?;
    let pbc: [bool; 3] = pbc_obj.extract()?;

    // 3. Handle PBC logic
    let any_periodic = pbc.iter().any(|&x| x);

    let py_cell = if any_periodic {
        Some(PyCell::from_ase(atoms)?)
    } else {
        None
    };

    Ok((positions, py_cell))
}

#[pyfunction]
#[pyo3(signature = (atoms, cutoff))]
fn build_from_ase<'py>(
    py: Python<'py>,
    atoms: &Bound<'py, PyAny>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (positions, py_cell) = extract_ase_data(atoms)?;
    build_neighborlists(py, py_cell.as_ref(), positions, cutoff, true)
}

#[pyfunction]
#[pyo3(signature = (atoms, cutoffs, labels=None, disjoint=false))]
fn build_multi_from_ase<'py>(
    py: Python<'py>,
    atoms: &Bound<'py, PyAny>,
    cutoffs: Vec<f64>,
    labels: Option<Vec<String>>,
    disjoint: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (positions, py_cell) = extract_ase_data(atoms)?;
    build_neighborlists_multi(py, py_cell.as_ref(), positions, cutoffs, labels, disjoint)
}

#[pyfunction]
#[pyo3(signature = (cell, positions, cutoffs, labels=None, disjoint=false))]
fn build_neighborlists_multi<'py>(
    py: Python<'py>,
    cell: Option<&PyCell>,
    positions: PyReadonlyArray2<'_, f64>,
    cutoffs: Vec<f64>,
    labels: Option<Vec<String>>,
    disjoint: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let pos_data = get_positions(&positions)?;
    let n_atoms = pos_data.len();

    if cutoffs.is_empty() {
        return Ok(PyDict::new(py));
    }
    if cutoffs.iter().any(|&c| !c.is_finite() || c <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All cutoffs must be positive and finite",
        ));
    }
    if labels.as_ref().is_some_and(|l| l.len() != cutoffs.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Length of labels must match length of cutoffs",
        ));
    }

    if n_atoms == 0 {
        let result_dict = PyDict::new(py);
        for (k, _) in cutoffs.iter().enumerate() {
            let cutoff_entry = PyDict::new(py);
            cutoff_entry.set_item(
                "edge_index",
                numpy::PyArray1::from_vec(py, vec![0i64; 0]).reshape((2, 0))?,
            )?;
            cutoff_entry.set_item(
                "shift",
                numpy::PyArray1::from_vec(py, vec![0i32; 0]).reshape((0, 3))?,
            )?;
            if let Some(ref l) = labels {
                result_dict.set_item(&l[k], cutoff_entry)?;
            } else {
                result_dict.set_item(k, cutoff_entry)?;
            }
        }
        return Ok(result_dict);
    }

    let cell_info = cell.map(|c| (*c.inner.h(), *c.inner.pbc()));

    let results = single::search_single_multi(&pos_data, cell_info, &cutoffs, disjoint)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let result_dict = PyDict::new(py);
    for (k, (mut edge_i, edge_j, shifts)) in results.into_iter().enumerate() {
        let cutoff_entry = PyDict::new(py);
        let n_edges = edge_i.len();

        // Create (2, N) edge_index
        edge_i.extend(edge_j);
        let edge_index = numpy::PyArray1::from_vec(py, edge_i).reshape((2, n_edges))?;
        cutoff_entry.set_item("edge_index", edge_index)?;

        let shifts_arr = numpy::PyArray1::from_vec(py, shifts).reshape((n_edges, 3))?;
        cutoff_entry.set_item("shift", shifts_arr)?;

        if let Some(ref l) = labels {
            result_dict.set_item(&l[k], cutoff_entry)?;
        } else {
            result_dict.set_item(k, cutoff_entry)?;
        }
    }

    Ok(result_dict)
}

fn is_zero_matrix(m: &ndarray::ArrayView2<f64>) -> bool {
    !m.iter().any(|&x| x != 0.0)
}

#[pyfunction]
#[pyo3(signature = (positions, batch, cells=None, cutoff=5.0, parallel=true))]
fn build_neighborlists_batch<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'_, f64>,
    batch: PyReadonlyArray1<'_, i32>,
    cells: Option<PyReadonlyArray3<'_, f64>>,
    cutoff: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyDict>> {
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cutoff must be positive and finite",
        ));
    }
    let pos_data = get_positions(&positions)?;
    let batch_view = batch.as_array();
    let n_total = pos_data.len();

    if batch_view.len() != n_total {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "positions and batch must have the same length",
        ));
    }

    if n_total == 0 {
        let dict = PyDict::new(py);
        dict.set_item(
            "edge_index",
            numpy::PyArray1::from_vec(py, vec![0i64; 0]).reshape((2, 0))?,
        )?;
        dict.set_item(
            "shift",
            numpy::PyArray1::from_vec(py, vec![0i32; 0]).reshape((0, 3))?,
        )?;
        return Ok(dict);
    }

    let batch_slice_cow: Cow<[i32]> = if let Some(s) = batch_view.as_slice() {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(batch_view.iter().copied().collect())
    };
    let batch_slice = batch_slice_cow.as_ref();

    let mut n_systems = 1;
    let mut current_batch_val = batch_slice[0];
    for &b in batch_slice.iter().skip(1) {
        if b < current_batch_val {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Batch IDs must be monotonic (non-decreasing)",
            ));
        }
        if b != current_batch_val {
            n_systems += 1;
            current_batch_val = b;
        }
    }

    let cell_matrices = if let Some(ref c) = cells {
        let cv = c.as_array();
        if cv.shape()[0] < n_systems {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected at least {} cells, but got {}",
                n_systems,
                cv.shape()[0]
            )));
        }
        let mut mats = Vec::with_capacity(n_systems);
        for i in 0..n_systems {
            let m = cv.index_axis(ndarray::Axis(0), i);
            if is_zero_matrix(&m) {
                mats.push(None);
            } else {
                // Transpose ASE (row-major) to internal (column-major)
                let h_mat = Matrix3::new(
                    m[[0, 0]],
                    m[[1, 0]],
                    m[[2, 0]],
                    m[[0, 1]],
                    m[[1, 1]],
                    m[[2, 1]],
                    m[[0, 2]],
                    m[[1, 2]],
                    m[[2, 2]],
                );
                mats.push(Some((h_mat, Vector3::new(true, true, true))));
            }
        }
        mats
    } else {
        vec![None; n_systems]
    };

    let (mut final_edge_i, final_edge_j, final_shift) =
        batch::search_batch(&pos_data, batch_slice, &cell_matrices, cutoff, parallel)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let total_edges = final_edge_i.len();
    let dict = PyDict::new(py);
    final_edge_i.extend(final_edge_j);
    let edge_index = numpy::PyArray1::from_vec(py, final_edge_i).reshape((2, total_edges))?;
    dict.set_item("edge_index", edge_index)?;
    dict.set_item(
        "shift",
        numpy::PyArray1::from_vec(py, final_shift).reshape((total_edges, 3))?,
    )?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (positions, batch, cells=None, cutoffs=vec![5.0], labels=None, disjoint=false))]
fn build_neighborlists_batch_multi<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'_, f64>,
    batch: PyReadonlyArray1<'_, i32>,
    cells: Option<PyReadonlyArray3<'_, f64>>,
    cutoffs: Vec<f64>,
    labels: Option<Vec<String>>,
    disjoint: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let pos_data = get_positions(&positions)?;
    let batch_view = batch.as_array();
    let n_total = pos_data.len();

    if batch_view.len() != n_total {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "positions and batch must have the same length",
        ));
    }

    if n_total == 0 || cutoffs.is_empty() {
        return Ok(PyDict::new(py));
    }

    if cutoffs.iter().any(|&c| !c.is_finite() || c <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All cutoffs must be positive and finite",
        ));
    }

    if labels.as_ref().is_some_and(|l| l.len() != cutoffs.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Length of labels must match length of cutoffs",
        ));
    }

    let batch_slice_cow: Cow<[i32]> = if let Some(s) = batch_view.as_slice() {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(batch_view.iter().copied().collect())
    };
    let batch_slice = batch_slice_cow.as_ref();

    let mut n_systems = 1;
    let mut current_batch_val = batch_slice[0];
    for &b in batch_slice.iter().skip(1) {
        if b < current_batch_val {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Batch IDs must be monotonic (non-decreasing)",
            ));
        }
        if b != current_batch_val {
            n_systems += 1;
            current_batch_val = b;
        }
    }

    let cell_matrices = if let Some(ref c) = cells {
        let cv = c.as_array();
        if cv.shape()[0] < n_systems {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected at least {} cells, but got {}",
                n_systems,
                cv.shape()[0]
            )));
        }
        let mut mats = Vec::with_capacity(n_systems);
        for i in 0..n_systems {
            let m = cv.index_axis(ndarray::Axis(0), i);
            if is_zero_matrix(&m) {
                mats.push(None);
            } else {
                // Transpose ASE (row-major) to internal (column-major)
                let h_mat = Matrix3::new(
                    m[[0, 0]],
                    m[[1, 0]],
                    m[[2, 0]],
                    m[[0, 1]],
                    m[[1, 1]],
                    m[[2, 1]],
                    m[[0, 2]],
                    m[[1, 2]],
                    m[[2, 2]],
                );
                mats.push(Some((h_mat, Vector3::new(true, true, true))));
            }
        }
        mats
    } else {
        vec![None; n_systems]
    };

    let results =
        batch::search_batch_multi(&pos_data, batch_slice, &cell_matrices, &cutoffs, disjoint)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let result_dict = PyDict::new(py);
    for (k, (mut edge_i, edge_j, shifts)) in results.into_iter().enumerate() {
        let n_edges = edge_i.len();

        let cutoff_entry = PyDict::new(py);

        // Create (2, N) edge_index
        edge_i.extend(edge_j);
        let edge_index = numpy::PyArray1::from_vec(py, edge_i).reshape((2, n_edges))?;
        cutoff_entry.set_item("edge_index", edge_index)?;

        cutoff_entry.set_item(
            "shift",
            numpy::PyArray1::from_vec(py, shifts).reshape((n_edges, 3))?,
        )?;

        if let Some(ref l) = labels {
            result_dict.set_item(&l[k], cutoff_entry)?;
        } else {
            result_dict.set_item(k, cutoff_entry)?;
        }
    }

    Ok(result_dict)
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
fn get_brute_force_threshold() -> usize {
    config::get_brute_force_threshold()
}

#[pyfunction]
fn set_brute_force_threshold(val: usize) {
    config::set_brute_force_threshold(val);
}

#[pyfunction]
fn get_parallel_threshold() -> usize {
    config::get_parallel_threshold()
}

#[pyfunction]
fn set_parallel_threshold(val: usize) {
    config::set_parallel_threshold(val);
}

#[pyfunction]
fn get_stack_threshold() -> usize {
    config::get_stack_threshold()
}

#[pyfunction]
fn set_stack_threshold(val: usize) {
    config::set_stack_threshold(val);
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
    m.add_function(wrap_pyfunction!(build_from_ase, m)?)?;
    m.add_function(wrap_pyfunction!(build_multi_from_ase, m)?)?;
    m.add_function(wrap_pyfunction!(build_neighborlists_multi, m)?)?;
    m.add_function(wrap_pyfunction!(build_neighborlists_batch, m)?)?;
    m.add_function(wrap_pyfunction!(build_neighborlists_batch_multi, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(get_brute_force_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(set_brute_force_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(get_parallel_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(set_parallel_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(get_stack_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(set_stack_threshold, m)?)?;
    Ok(())
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    #[test]
    fn test_python_api_basic() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let res = get_brute_force_threshold();
            assert!(res > 0);

            let threads = get_num_threads();
            assert!(threads > 0);
        });
    }

    #[test]
    fn test_build_neighborlists_empty() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let positions = numpy::PyArray2::<f64>::zeros(py, [0, 3], false);
            let cutoff = 1.0;
            let res = build_neighborlists(py, None, positions.readonly(), cutoff, false).unwrap();
            let edge_index = res.get_item("edge_index").unwrap().unwrap();
            let shift = res.get_item("shift").unwrap().unwrap();
            // Checking basic validity
            assert!(edge_index.extract::<numpy::PyReadonlyArray2<i64>>().is_ok());
            assert!(shift.extract::<numpy::PyReadonlyArray2<i32>>().is_ok());
        });
    }

    #[test]
    fn test_pycell_new() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let h = vec![
                vec![10.0, 0.0, 0.0],
                vec![0.0, 10.0, 0.0],
                vec![0.0, 0.0, 10.0],
            ];
            let cell = PyCell::new(h, None).unwrap();
            assert_eq!(cell.inner.h()[(0, 0)], 10.0);
        });
    }

    #[test]
    fn test_build_neighborlists_simple() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let positions_vec = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
            let positions = numpy::PyArray1::from_vec(py, positions_vec)
                .reshape((2, 3))
                .unwrap();
            let cutoff = 1.5;

            let res = build_neighborlists(py, None, positions.readonly(), cutoff, false).unwrap();

            let edge_index = res
                .get_item("edge_index")
                .unwrap()
                .unwrap()
                .extract::<numpy::PyReadonlyArray2<i64>>()
                .unwrap();
            let _shift = res
                .get_item("shift")
                .unwrap()
                .unwrap()
                .extract::<numpy::PyReadonlyArray2<i32>>()
                .unwrap();

            let edges = edge_index.as_array();
            // Should have 1 edge: 0->1 (half list, i < j)
            assert_eq!(edges.shape()[1], 1);
        });
    }

    #[test]
    fn test_build_neighborlists_multi() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let positions_vec = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
            let positions = numpy::PyArray1::from_vec(py, positions_vec)
                .reshape((2, 3))
                .unwrap();
            let cutoffs = vec![0.5, 1.5]; // 0.5 too small, 1.5 enough

            let res =
                build_neighborlists_multi(py, None, positions.readonly(), cutoffs, None, false)
                    .unwrap();

            let res0 = res.get_item(0).unwrap().unwrap(); // Cutoff 0.5
            let ei0 = res0
                .get_item("edge_index")
                .unwrap()
                .extract::<numpy::PyReadonlyArray2<i64>>()
                .unwrap();
            assert_eq!(ei0.as_array().shape()[1], 0);

            let res1 = res.get_item(1).unwrap().unwrap(); // Cutoff 1.5
            let ei1 = res1
                .get_item("edge_index")
                .unwrap()
                .extract::<numpy::PyReadonlyArray2<i64>>()
                .unwrap();
            assert_eq!(ei1.as_array().shape()[1], 1);
        });
    }

    #[test]
    fn test_build_neighborlists_batch() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let positions_vec = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10.0, 10.0, 10.0];
            let positions = numpy::PyArray1::from_vec(py, positions_vec)
                .reshape((3, 3))
                .unwrap();
            let batch = numpy::PyArray1::from_vec(py, vec![0, 0, 1]);
            let cutoff = 1.5;

            let res = build_neighborlists_batch(
                py,
                positions.readonly(),
                batch.readonly(),
                None,
                cutoff,
                false,
            )
            .unwrap();

            let edge_index = res
                .get_item("edge_index")
                .unwrap()
                .unwrap()
                .extract::<numpy::PyReadonlyArray2<i64>>()
                .unwrap();
            // System 0: (0,1) -> 1 edge. System 1: isolated -> 0 edges. Total 1.
            assert_eq!(edge_index.as_array().shape()[1], 1);
        });
    }

    #[test]
    fn test_pycell_methods() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let h = vec![
                vec![10.0, 0.0, 0.0],
                vec![0.0, 10.0, 0.0],
                vec![0.0, 0.0, 10.0],
            ];
            let cell = PyCell::new(h, None).unwrap();

            // test wrap
            let p = [15.0, 2.0, -1.0];
            let wrapped = cell.wrap(p);
            assert_eq!(wrapped, [5.0, 2.0, 9.0]);

            // test repr
            let r = cell.__repr__();
            assert!(r.contains("PyCell"));
            assert!(r.contains("pbc=[true, true, true]"));
        });
    }

    #[test]
    fn test_python_api_errors() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let positions = numpy::PyArray2::<f64>::zeros(py, [1, 3], false);

            // Invalid cutoff
            let res = build_neighborlists(py, None, positions.readonly(), -1.0, false);
            assert!(res.is_err());

            // Non-monotonic batch
            let batch = numpy::PyArray1::from_vec(py, vec![1, 0]);
            let pos2 = numpy::PyArray2::<f64>::zeros(py, [2, 3], false);
            let res_batch =
                build_neighborlists_batch(py, pos2.readonly(), batch.readonly(), None, 1.0, false);
            assert!(res_batch.is_err());
        });
    }

    #[test]
    fn test_global_configs() {
        pyo3::prepare_freethreaded_python();
        let _ = set_num_threads(2);
        assert!(get_num_threads() > 0);

        init_logging(Some("debug".to_string()));

        set_brute_force_threshold(500);
        assert_eq!(get_brute_force_threshold(), 500);

        set_parallel_threshold(100);
        assert_eq!(get_parallel_threshold(), 100);

        set_stack_threshold(400);
        assert_eq!(get_stack_threshold(), 400);
    }

    #[test]
    fn test_ase_integration_basic() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Create a simple ASE atoms object
            let ase = py.import("ase").unwrap();
            let atoms_cls = ase.getattr("Atoms").unwrap();

            let kwargs = PyDict::new(py);
            kwargs.set_item("symbols", "H2").unwrap();
            let pos = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
            kwargs.set_item("positions", pos).unwrap();

            let atoms = atoms_cls.call((), Some(&kwargs)).unwrap();

            let res = build_from_ase(py, &atoms, 1.5).unwrap();
            let ei = res
                .get_item("edge_index")
                .unwrap()
                .unwrap()
                .extract::<numpy::PyReadonlyArray2<i64>>()
                .unwrap();
            assert_eq!(ei.as_array().shape()[1], 1);
        });
    }

    #[test]
    fn test_build_neighborlists_batch_multi() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let positions_vec = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10.0, 10.0, 10.0];
            let positions = numpy::PyArray1::from_vec(py, positions_vec)
                .reshape((3, 3))
                .unwrap();
            let batch = numpy::PyArray1::from_vec(py, vec![0, 0, 1]);
            let cutoffs = vec![0.5, 1.5];

            let res = build_neighborlists_batch_multi(
                py,
                positions.readonly(),
                batch.readonly(),
                None,
                cutoffs,
                None,
                false,
            )
            .unwrap();

            let res1 = res.get_item(1).unwrap().unwrap(); // cutoff 1.5
            let ei1 = res1
                .get_item("edge_index")
                .unwrap()
                .extract::<numpy::PyReadonlyArray2<i64>>()
                .unwrap();
            assert_eq!(ei1.as_array().shape()[1], 1);
        });
    }

    #[test]
    fn test_build_neighborlists_batch_multi_disjoint() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let positions_vec = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.1, 0.0, 0.0];
            let positions = numpy::PyArray1::from_vec(py, positions_vec)
                .reshape((3, 3))
                .unwrap();
            let batch = numpy::PyArray1::from_vec(py, vec![0, 0, 0]);
            let cutoffs = vec![1.5, 3.0];
            let labels = vec!["short".to_string(), "long".to_string()];

            // disjoint = true
            let res = build_neighborlists_batch_multi(
                py,
                positions.readonly(),
                batch.readonly(),
                None,
                cutoffs,
                Some(labels),
                true,
            )
            .unwrap();

            let short = res.get_item("short").unwrap().unwrap();
            // Edges (0,1), (1,2). Total 2 (half list).
            assert_eq!(
                short
                    .get_item("edge_index")
                    .unwrap()
                    .extract::<numpy::PyReadonlyArray2<i64>>()
                    .unwrap()
                    .as_array()
                    .shape()[1],
                2
            );

            let long = res.get_item("long").unwrap().unwrap();
            // Edge (0,2) only. Total 1.
            assert_eq!(
                long.get_item("edge_index")
                    .unwrap()
                    .extract::<numpy::PyReadonlyArray2<i64>>()
                    .unwrap()
                    .as_array()
                    .shape()[1],
                1
            );
        });
    }

    #[test]
    fn test_batch_with_zero_cells() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let positions = numpy::PyArray2::<f64>::zeros(py, [2, 3], false);
            let batch = numpy::PyArray1::from_vec(py, vec![0, 1]);
            let cells = numpy::PyArray3::<f64>::zeros(py, [2, 3, 3], false);

            let res = build_neighborlists_batch(
                py,
                positions.readonly(),
                batch.readonly(),
                Some(cells.readonly()),
                1.0,
                false,
            )
            .unwrap();
            assert!(
                res.get_item("edge_index")
                    .unwrap()
                    .unwrap()
                    .extract::<numpy::PyReadonlyArray2<i64>>()
                    .unwrap()
                    .as_array()
                    .is_empty()
            );
        });
    }

    #[test]
    fn test_build_neighborlists_multi_labels() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let pos_vec = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
            let positions = numpy::PyArray1::from_vec(py, pos_vec)
                .reshape((2, 3))
                .unwrap();
            let cutoffs = vec![0.5, 1.5];
            let labels = vec!["a".to_string(), "b".to_string()];

            let res = build_neighborlists_multi(
                py,
                None,
                positions.readonly(),
                cutoffs,
                Some(labels),
                false,
            )
            .unwrap();
            assert!(res.contains("a").unwrap());
            assert!(res.contains("b").unwrap());
        });
    }

    #[test]
    fn test_extract_ase_data_isolated() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let ase = py.import("ase").unwrap();
            let atoms_cls = ase.getattr("Atoms").unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("symbols", "H").unwrap();
            kwargs
                .set_item("positions", vec![vec![0.0, 0.0, 0.0]])
                .unwrap();
            kwargs.set_item("pbc", vec![false, false, false]).unwrap();
            let atoms = atoms_cls.call((), Some(&kwargs)).unwrap();

            let (pos, cell) = extract_ase_data(&atoms).unwrap();
            assert_eq!(pos.shape(), [1, 3]);
            assert!(cell.is_none());
        });
    }
}
