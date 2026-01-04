use pyo3::prelude::*;

pub mod cell;
pub mod search;

// Tuning parameters
const BRUTE_FORCE_THRESHOLD: usize = 500;
const PARALLEL_THRESHOLD: usize = 20;
const AUTO_BOX_MARGIN: f64 = 1.0;

type NeighborListResult = (Vec<i64>, Vec<i64>, Vec<i32>);

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use crate::cell::Cell;
use crate::search::CellList;
use nalgebra::{Matrix3, Vector3};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ndarray};
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
            if p.x < min_bound.x {
                min_bound.x = p.x;
            }
            if p.y < min_bound.y {
                min_bound.y = p.y;
            }
            if p.z < min_bound.z {
                min_bound.z = p.z;
            }
            if p.x > max_bound.x {
                max_bound.x = p.x;
            }
            if p.y > max_bound.y {
                max_bound.y = p.y;
            }
            if p.z > max_bound.z {
                max_bound.z = p.z;
            }
        }
        pos_vec.push(p);
    }

    let cell_inner = if let Some(c) = cell {
        c.inner.clone()
    } else {
        // Infer cell from positions for non-PBC
        let margin = cutoff + AUTO_BOX_MARGIN;
        let span = max_bound - min_bound;
        let lx = span.x + 2.0 * margin;
        let ly = span.y + 2.0 * margin;
        let lz = span.z + 2.0 * margin;

        let h_mat = Matrix3::new(lx, 0.0, 0.0, 0.0, ly, 0.0, 0.0, 0.0, lz);
        Cell::new(h_mat).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };

    let perp = cell_inner.perpendicular_widths();
    let min_width = perp.x.min(perp.y).min(perp.z);
    // Strict inequality to be safe with rounding? <= is typically fine for MIC logic, but < is safer.
    let mic_safe = cutoff * 2.0 < min_width;

    let (edge_i, edge_j, shifts) = if n_atoms < BRUTE_FORCE_THRESHOLD && mic_safe {
        search::brute_force_search_full(&cell_inner, &pos_vec, cutoff)
    } else {
        let cl = CellList::build(&cell_inner, &pos_vec, cutoff);

        if parallel && n_atoms >= PARALLEL_THRESHOLD {
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

    let n_edges = edge_i.len();
    let mut edge_i = edge_i;
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
    let pbc: Vec<bool> = pbc_obj.extract()?;
    if pbc.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "PBC must be length 3",
        ));
    }

    // 3. Extract Cell
    let cell_obj = atoms.call_method0("get_cell")?;
    let cell_array_obj = cell_obj.call_method0("__array__")?;
    let cell_array: PyReadonlyArray2<f64> = cell_array_obj.extract()?;

    // 4. Handle PBC logic
    let all_periodic = pbc.iter().all(|&x| x);
    let none_periodic = pbc.iter().all(|&x| !x);

    let py_cell = if all_periodic {
        let c = cell_array.as_array();
        if c.shape() != [3, 3] {
            return Err(pyo3::exceptions::PyValueError::new_err("Cell must be 3x3"));
        }
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
        let inner =
            Cell::new(h_mat).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Some(PyCell { inner })
    } else if none_periodic {
        None
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Mixed PBC (e.g., [True, True, False]) is not currently supported. Only all-True or all-False is supported.",
        ));
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
    let pos_view = positions.as_array();
    let n_atoms = pos_view.shape()[0];
    if pos_view.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Positions must be (N, 3)",
        ));
    }
    if cutoffs.is_empty() {
        return Ok(PyDict::new(py));
    }
    if let Some(ref l) = labels {
        if l.len() != cutoffs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Length of labels must match length of cutoffs",
            ));
        }
    }

    let mut pos_vec = Vec::with_capacity(n_atoms);
    let mut min_bound = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut max_bound = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

    for row in pos_view.rows() {
        let p = Vector3::new(row[0], row[1], row[2]);
        if cell.is_none() {
            if p.x < min_bound.x {
                min_bound.x = p.x;
            }
            if p.y < min_bound.y {
                min_bound.y = p.y;
            }
            if p.z < min_bound.z {
                min_bound.z = p.z;
            }
            if p.x > max_bound.x {
                max_bound.x = p.x;
            }
            if p.y > max_bound.y {
                max_bound.y = p.y;
            }
            if p.z > max_bound.z {
                max_bound.z = p.z;
            }
        }
        pos_vec.push(p);
    }

    let max_cutoff = cutoffs.iter().cloned().fold(f64::NAN, f64::max);

    let cell_inner = if let Some(c) = cell {
        c.inner.clone()
    } else {
        // Infer cell from positions for non-PBC
        let margin = max_cutoff + AUTO_BOX_MARGIN;
        let span = max_bound - min_bound;
        let lx = span.x + 2.0 * margin;
        let ly = span.y + 2.0 * margin;
        let lz = span.z + 2.0 * margin;

        let h_mat = Matrix3::new(lx, 0.0, 0.0, 0.0, ly, 0.0, 0.0, 0.0, lz);
        Cell::new(h_mat).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };

    let perp = cell_inner.perpendicular_widths();
    let min_width = perp.x.min(perp.y).min(perp.z);
    let mic_safe = max_cutoff * 2.0 < min_width;

    let results = if n_atoms < BRUTE_FORCE_THRESHOLD && mic_safe {
        search::brute_force_search_multi(&cell_inner, &pos_vec, &cutoffs, disjoint)
    } else {
        let cl = CellList::build(&cell_inner, &pos_vec, max_cutoff);
        cl.par_search_multi(&cell_inner, &cutoffs, disjoint)
    };

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
    m.iter().all(|&x| x == 0.0)
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
    let pos_view = positions.as_array();
    let batch_view = batch.as_array();
    let n_total = pos_view.shape()[0];

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

    // 1. Determine system boundaries
    let mut system_indices = Vec::new();
    let mut current_start = 0;
    let mut current_batch_val = batch_view[0];

    for i in 1..n_total {
        if batch_view[i] != current_batch_val {
            system_indices.push((current_start, i, current_batch_val));
            current_start = i;
            current_batch_val = batch_view[i];
        }
    }
    system_indices.push((current_start, n_total, current_batch_val));
    let n_systems = system_indices.len();

    // 2. Extract cell matrices
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
                mats.push(Some(Matrix3::new(
                    m[[0, 0]],
                    m[[0, 1]],
                    m[[0, 2]],
                    m[[1, 0]],
                    m[[1, 1]],
                    m[[1, 2]],
                    m[[2, 0]],
                    m[[2, 1]],
                    m[[2, 2]],
                )));
            }
        }
        mats
    } else {
        vec![None; n_systems]
    };

    // 3. Parallel search over systems
    use rayon::prelude::*;
    let results: Result<Vec<NeighborListResult>, String> = system_indices
        .par_iter()
        .enumerate()
        .map(|(i, &(start, end, _batch_val))| {
            let pos_slice = pos_view.slice(ndarray::s![start..end, ..]);
            let n_atoms_local = end - start;
            let mut pos_vec = Vec::with_capacity(n_atoms_local);
            let mut min_bound = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut max_bound =
                Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

            for row in pos_slice.rows() {
                let p = Vector3::new(row[0], row[1], row[2]);
                if cell_matrices[i].is_none() {
                    if p.x < min_bound.x {
                        min_bound.x = p.x;
                    }
                    if p.y < min_bound.y {
                        min_bound.y = p.y;
                    }
                    if p.z < min_bound.z {
                        min_bound.z = p.z;
                    }
                    if p.x > max_bound.x {
                        max_bound.x = p.x;
                    }
                    if p.y > max_bound.y {
                        max_bound.y = p.y;
                    }
                    if p.z > max_bound.z {
                        max_bound.z = p.z;
                    }
                }
                pos_vec.push(p);
            }

            let cell_inner = if let Some(h_mat) = cell_matrices[i] {
                Cell::new(h_mat).map_err(|e| e.to_string())?
            } else {
                let margin = cutoff + AUTO_BOX_MARGIN;
                let span = max_bound - min_bound;
                let h_mat = Matrix3::new(
                    span.x + 2.0 * margin,
                    0.0,
                    0.0,
                    0.0,
                    span.y + 2.0 * margin,
                    0.0,
                    0.0,
                    0.0,
                    span.z + 2.0 * margin,
                );
                Cell::new(h_mat).map_err(|e| e.to_string())?
            };

            let perp = cell_inner.perpendicular_widths();
            let min_width = perp.x.min(perp.y).min(perp.z);
            let mic_safe = cutoff * 2.0 < min_width;

            let (ei, ej, s) = if n_atoms_local < BRUTE_FORCE_THRESHOLD && mic_safe {
                search::brute_force_search_full(&cell_inner, &pos_vec, cutoff)
            } else {
                let cl = CellList::build(&cell_inner, &pos_vec, cutoff);
                if parallel && n_atoms_local >= PARALLEL_THRESHOLD {
                    cl.par_search_optimized(&cell_inner, cutoff)
                } else {
                    let neighbors = cl.search(&cell_inner, &pos_vec, cutoff);
                    let mut ei = Vec::with_capacity(neighbors.len());
                    let mut ej = Vec::with_capacity(neighbors.len());
                    let mut s = Vec::with_capacity(neighbors.len() * 3);
                    for (u, v, sx, sy, sz) in neighbors {
                        ei.push(u as i64);
                        ej.push(v as i64);
                        s.push(sx);
                        s.push(sy);
                        s.push(sz);
                    }
                    (ei, ej, s)
                }
            };

            let offset = start as i64;
            let ei_global = ei.into_iter().map(|idx| idx + offset).collect();
            let ej_global = ej.into_iter().map(|idx| idx + offset).collect();

            Ok((ei_global, ej_global, s))
        })
        .collect();

    let results = results.map_err(pyo3::exceptions::PyValueError::new_err)?;

    let total_edges = results.iter().map(|r| r.0.len()).sum();
    let mut final_edge_i = Vec::with_capacity(total_edges);
    let mut final_edge_j = Vec::with_capacity(total_edges);
    let mut final_shift = Vec::with_capacity(total_edges * 3);

    for (ei, ej, s) in results {
        final_edge_i.extend(ei);
        final_edge_j.extend(ej);
        final_shift.extend(s);
    }

    let dict = PyDict::new(py);
    let mut final_edge_i = final_edge_i;
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
    let pos_view = positions.as_array();
    let batch_view = batch.as_array();
    let n_total = pos_view.shape()[0];

    if batch_view.len() != n_total {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "positions and batch must have the same length",
        ));
    }

    if n_total == 0 || cutoffs.is_empty() {
        return Ok(PyDict::new(py));
    }

    if let Some(ref l) = labels {
        if l.len() != cutoffs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Length of labels must match length of cutoffs",
            ));
        }
    }

    // 1. Determine system boundaries
    let mut system_indices = Vec::new();
    let mut current_start = 0;
    let mut current_batch_val = batch_view[0];

    for i in 1..n_total {
        if batch_view[i] != current_batch_val {
            system_indices.push((current_start, i, current_batch_val));
            current_start = i;
            current_batch_val = batch_view[i];
        }
    }
    system_indices.push((current_start, n_total, current_batch_val));
    let n_systems = system_indices.len();

    // 2. Extract cell matrices
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
                mats.push(Some(Matrix3::new(
                    m[[0, 0]],
                    m[[0, 1]],
                    m[[0, 2]],
                    m[[1, 0]],
                    m[[1, 1]],
                    m[[1, 2]],
                    m[[2, 0]],
                    m[[2, 1]],
                    m[[2, 2]],
                )));
            }
        }
        mats
    } else {
        vec![None; n_systems]
    };

    let max_cutoff = cutoffs.iter().cloned().fold(f64::NAN, f64::max);

    // 3. Parallel search over systems for multiple cutoffs
    use rayon::prelude::*;
    let results: Result<Vec<Vec<NeighborListResult>>, String> = system_indices
        .par_iter()
        .enumerate()
        .map(|(i, &(start, end, _batch_val))| {
            let pos_slice = pos_view.slice(ndarray::s![start..end, ..]);
            let n_atoms_local = end - start;
            let mut pos_vec = Vec::with_capacity(n_atoms_local);
            let mut min_bound = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut max_bound =
                Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

            for row in pos_slice.rows() {
                let p = Vector3::new(row[0], row[1], row[2]);
                if cell_matrices[i].is_none() {
                    if p.x < min_bound.x {
                        min_bound.x = p.x;
                    }
                    if p.y < min_bound.y {
                        min_bound.y = p.y;
                    }
                    if p.z < min_bound.z {
                        min_bound.z = p.z;
                    }
                    if p.x > max_bound.x {
                        max_bound.x = p.x;
                    }
                    if p.y > max_bound.y {
                        max_bound.y = p.y;
                    }
                    if p.z > max_bound.z {
                        max_bound.z = p.z;
                    }
                }
                pos_vec.push(p);
            }

            let cell_inner = if let Some(h_mat) = cell_matrices[i] {
                Cell::new(h_mat).map_err(|e| e.to_string())?
            } else {
                let margin = max_cutoff + AUTO_BOX_MARGIN;
                let span = max_bound - min_bound;
                let h_mat = Matrix3::new(
                    span.x + 2.0 * margin,
                    0.0,
                    0.0,
                    0.0,
                    span.y + 2.0 * margin,
                    0.0,
                    0.0,
                    0.0,
                    span.z + 2.0 * margin,
                );
                Cell::new(h_mat).map_err(|e| e.to_string())?
            };

            let perp = cell_inner.perpendicular_widths();
            let min_width = perp.x.min(perp.y).min(perp.z);
            let mic_safe = max_cutoff * 2.0 < min_width;

            let system_results = if n_atoms_local < BRUTE_FORCE_THRESHOLD && mic_safe {
                search::brute_force_search_multi(&cell_inner, &pos_vec, &cutoffs, disjoint)
            } else {
                let cl = CellList::build(&cell_inner, &pos_vec, max_cutoff);
                cl.par_search_multi(&cell_inner, &cutoffs, disjoint)
            };

            let offset = start as i64;
            let offset_results = system_results
                .into_iter()
                .map(|(ei, ej, s)| {
                    let ei_global = ei.into_iter().map(|idx| idx + offset).collect();
                    let ej_global = ej.into_iter().map(|idx| idx + offset).collect();
                    (ei_global, ej_global, s)
                })
                .collect();

            Ok(offset_results)
        })
        .collect();

    let results = results.map_err(pyo3::exceptions::PyValueError::new_err)?;

    // 4. Aggregation by cutoff
    let n_cutoffs = cutoffs.len();
    let result_dict = PyDict::new(py);

    for k in 0..n_cutoffs {
        let total_edges: usize = results.iter().map(|sys_res| sys_res[k].0.len()).sum();

        let mut final_edge_i = Vec::with_capacity(total_edges);
        let mut final_edge_j = Vec::with_capacity(total_edges);
        let mut final_shift = Vec::with_capacity(total_edges * 3);

        for sys_res in &results {
            let (ei, ej, s) = &sys_res[k];
            final_edge_i.extend(ei.iter().cloned());
            final_edge_j.extend(ej.iter().cloned());
            final_shift.extend(s.iter().cloned());
        }

        let cutoff_entry = PyDict::new(py);
        
        // Create (2, N) edge_index
        final_edge_i.extend(final_edge_j);
        let edge_index = numpy::PyArray1::from_vec(py, final_edge_i).reshape((2, total_edges))?;
        cutoff_entry.set_item("edge_index", edge_index)?;

        cutoff_entry.set_item(
            "shift",
            numpy::PyArray1::from_vec(py, final_shift).reshape((total_edges, 3))?,
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
    Ok(())
}
