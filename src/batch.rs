use crate::cell::Cell;
use crate::config;
use crate::search::{self, CellList, EdgeResult};
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

// Constants shared with lib.rs (or defined here and used there)
pub const AUTO_BOX_MARGIN: f64 = 1.0;

pub fn search_batch(
    positions: &[Vector3<f64>],
    batch: &[i32],
    cells: &[Option<(Matrix3<f64>, Vector3<bool>)>],
    cutoff: f64,
    parallel: bool,
) -> Result<EdgeResult, String> {
    let n_total = positions.len();
    if batch.len() != n_total {
        return Err("positions and batch must have the same length".to_string());
    }
    if n_total == 0 {
        return Ok((vec![], vec![], vec![]));
    }

    // 1. Determine system boundaries
    let mut system_indices = Vec::new();
    let mut current_start = 0;
    let mut current_batch_val = batch[0];

    for (i, &val) in batch.iter().enumerate().skip(1) {
        if val != current_batch_val {
            system_indices.push((current_start, i));
            current_start = i;
            current_batch_val = val;
        }
    }
    system_indices.push((current_start, n_total));
    let n_systems = system_indices.len();

    if cells.len() < n_systems {
        return Err(format!(
            "Expected at least {} cells, but got {}",
            n_systems,
            cells.len()
        ));
    }

    // 2. Parallel search over systems
    let results: Result<Vec<EdgeResult>, String> = system_indices
        .par_iter()
        .enumerate()
        .map(|(i, &(start, end))| {
            let pos_slice = &positions[start..end];
            let n_atoms_local = end - start;

            // Determine cell
            let cell_inner = if let Some((h_mat, pbc)) = cells[i] {
                Cell::new(h_mat, pbc).map_err(|e| e.to_string())?
            } else {
                let mut min_bound = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
                let mut max_bound =
                    Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for p in pos_slice {
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
                Cell::new(h_mat, Vector3::new(false, false, false)).map_err(|e| e.to_string())?
            };

            let perp = cell_inner.perpendicular_widths();
            let min_width = perp.x.min(perp.y).min(perp.z);
            let mic_safe = cutoff * 2.0 < min_width;

            let (ei, ej, s) = if n_atoms_local < config::get_brute_force_threshold() && mic_safe {
                search::brute_force_search_full(&cell_inner, pos_slice, cutoff)
            } else {
                let cl = CellList::build(&cell_inner, pos_slice, cutoff);
                if parallel && n_atoms_local >= config::get_parallel_threshold() {
                    cl.par_search_optimized(&cell_inner, cutoff)
                } else {
                    let neighbors = cl.search(&cell_inner, pos_slice, cutoff);
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

    let results = results?;

    let total_edges: usize = results.iter().map(|r| r.0.len()).sum();
    let mut final_edge_i = Vec::with_capacity(total_edges);
    let mut final_edge_j = Vec::with_capacity(total_edges);
    let mut final_shift = Vec::with_capacity(total_edges * 3);

    for (ei, ej, s) in results {
        final_edge_i.extend(ei);
        final_edge_j.extend(ej);
        final_shift.extend(s);
    }

    Ok((final_edge_i, final_edge_j, final_shift))
}

pub fn search_batch_multi(
    positions: &[Vector3<f64>],
    batch: &[i32],
    cells: &[Option<(Matrix3<f64>, Vector3<bool>)>],
    cutoffs: &[f64],
    disjoint: bool,
) -> Result<Vec<EdgeResult>, String> {
    let n_total = positions.len();
    if batch.len() != n_total {
        return Err("positions and batch must have the same length".to_string());
    }
    if n_total == 0 || cutoffs.is_empty() {
        return Ok(vec![(vec![], vec![], vec![]); cutoffs.len()]);
    }

    // 1. Determine system boundaries
    let mut system_indices = Vec::new();
    let mut current_start = 0;
    let mut current_batch_val = batch[0];

    for (i, &val) in batch.iter().enumerate().skip(1) {
        if val != current_batch_val {
            system_indices.push((current_start, i));
            current_start = i;
            current_batch_val = val;
        }
    }
    system_indices.push((current_start, n_total));
    let n_systems = system_indices.len();

    if cells.len() < n_systems {
        return Err(format!(
            "Expected at least {} cells, but got {}",
            n_systems,
            cells.len()
        ));
    }

    let max_cutoff = cutoffs.iter().cloned().fold(f64::NAN, f64::max);

    // 2. Parallel search
    let results: Result<Vec<Vec<EdgeResult>>, String> = system_indices
        .par_iter()
        .enumerate()
        .map(|(i, &(start, end))| {
            let pos_slice = &positions[start..end];
            let n_atoms_local = end - start;

            let cell_inner = if let Some((h_mat, pbc)) = cells[i] {
                Cell::new(h_mat, pbc).map_err(|e| e.to_string())?
            } else {
                let mut min_bound = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
                let mut max_bound =
                    Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for p in pos_slice {
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
                Cell::new(h_mat, Vector3::new(false, false, false)).map_err(|e| e.to_string())?
            };

            let perp = cell_inner.perpendicular_widths();
            let min_width = perp.x.min(perp.y).min(perp.z);
            let mic_safe = max_cutoff * 2.0 < min_width;

            let system_results = if n_atoms_local < config::get_brute_force_threshold() && mic_safe
            {
                search::brute_force_search_multi(&cell_inner, pos_slice, cutoffs, disjoint)
            } else {
                let cl = CellList::build(&cell_inner, pos_slice, max_cutoff);
                cl.par_search_multi(&cell_inner, cutoffs, disjoint)
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

    let results = results?;

    // 3. Aggregation
    let mut aggregated = Vec::with_capacity(cutoffs.len());
    for k in 0..cutoffs.len() {
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
        aggregated.push((final_edge_i, final_edge_j, final_shift));
    }

    Ok(aggregated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_batch_search_basic() {
        let h = Matrix3::identity() * 10.0;
        let cell = Some((h, Vector3::new(true, true, true)));

        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0), // System 0
            Vector3::new(1.2, 1.0, 1.0), // System 0
            Vector3::new(5.0, 5.0, 5.0), // System 1 (shifted idx 0)
            Vector3::new(5.2, 5.0, 5.0), // System 1 (shifted idx 1)
        ];
        let batch = vec![0, 0, 1, 1];
        let cells = vec![cell, cell];

        let (ei, ej, _) = search_batch(&positions, &batch, &cells, 0.5, true).unwrap();

        // Expected: (0, 1) for system 0
        // (2, 3) for system 1
        // Only i < j is returned

        assert_eq!(ei.len(), 2);
        assert_eq!(ej.len(), 2);

        let mut edges: Vec<(i64, i64)> = ei.iter().zip(ej.iter()).map(|(&i, &j)| (i, j)).collect();
        edges.sort();

        assert_eq!(edges, vec![(0, 1), (2, 3)]);
    }

    #[test]
    fn test_batch_search_multi_basic() {
        let h = Matrix3::identity() * 10.0;
        let cell = Some((h, Vector3::new(true, true, true)));

        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0), // System 0
            Vector3::new(1.2, 1.0, 1.0), // System 0
            Vector3::new(5.0, 5.0, 5.0), // System 1
            Vector3::new(5.5, 5.0, 5.0), // System 1 (dist 0.5)
        ];
        let batch = vec![0, 0, 1, 1];
        let cells = vec![cell, cell];
        let cutoffs = vec![0.3, 0.6];

        let results = search_batch_multi(&positions, &batch, &cells, &cutoffs, false).unwrap();

        assert_eq!(results.len(), 2);

        // Cutoff 0.3
        // System 0: (0,1) dist 0.2 -> match
        // System 1: (2,3) dist 0.5 -> no match
        let (ei0, _, _) = &results[0];
        assert_eq!(ei0.len(), 1); // (0,1)

        // Cutoff 0.6
        // System 0: match
        // System 1: match
        let (ei1, _, _) = &results[1];
        assert_eq!(ei1.len(), 2); // (0,1), (2,3)
    }

    #[test]
    fn test_batch_search_autobox() {
        // No cells provided
        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.2, 1.0, 1.0),
            Vector3::new(10.0, 10.0, 10.0),
            Vector3::new(10.2, 10.0, 10.0),
        ];
        let batch = vec![0, 0, 1, 1];
        let cells = vec![None, None];

        let (ei, ej, _) = search_batch(&positions, &batch, &cells, 0.5, false).unwrap();
        assert_eq!(ei.len(), 2);
        assert_eq!(ej.len(), 2);
    }

    #[test]
    fn test_batch_search_parallel() {
        // Trigger parallel threshold (20 atoms per system)
        let mut positions = Vec::new();
        let mut batch = Vec::new();
        for s in 0..2 {
            for i in 0..25 {
                positions.push(Vector3::new(i as f64, 0.0, 0.0));
                batch.push(s);
            }
        }
        let cells = vec![None, None];
        // This should trigger parallel path in search_batch
        let (ei, _, _) = search_batch(&positions, &batch, &cells, 1.1, true).unwrap();
        assert!(!ei.is_empty());
    }

    #[test]
    fn test_batch_search_multi_parallel() {
        let mut positions = Vec::new();
        let mut batch = Vec::new();
        for s in 0..2 {
            for i in 0..25 {
                positions.push(Vector3::new(i as f64, 0.0, 0.0));
                batch.push(s);
            }
        }
        let cells = vec![None, None];
        let cutoffs = vec![1.1, 2.1];
        let results = search_batch_multi(&positions, &batch, &cells, &cutoffs, false).unwrap();
        assert_eq!(results.len(), 2);
        assert!(!results[0].0.is_empty());
        assert!(results[1].0.len() > results[0].0.len());
    }

    #[test]
    fn test_batch_search_multi_disjoint() {
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.1, 0.0, 0.0),
        ];
        let batch = vec![0, 0, 0];
        let cells = vec![None];
        let cutoffs = vec![1.5, 3.0];

        // disjoint = true
        let results = search_batch_multi(&positions, &batch, &cells, &cutoffs, true).unwrap();
        // Cutoff 1.5: (0,1), (1,2)
        // Cutoff 3.0: (0,2) -- because (0,1) and (1,2) already in 1.5
        assert_eq!(results[0].0.len(), 2);
        assert_eq!(results[1].0.len(), 1);
    }

    #[test]
    fn test_batch_search_serial() {
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let batch = vec![0, 0];
        let cells = vec![None];
        // force serial path in search_batch (parallel=false)
        let (ei, _, _) = search_batch(&positions, &batch, &cells, 1.5, false).unwrap();
        assert_eq!(ei.len(), 1);
    }

    #[test]
    fn test_batch_search_large() {
        let mut positions = Vec::new();
        let mut batch = Vec::new();
        for s in 0..2 {
            for i in 0..505 {
                positions.push(Vector3::new(i as f64, 0.0, 0.0));
                batch.push(s);
            }
        }
        let cells = vec![None, None];
        let (ei, _, _) = search_batch(&positions, &batch, &cells, 1.1, true).unwrap();
        assert!(!ei.is_empty());
    }

    #[test]
    fn test_batch_search_empty() {
        let (ei, _, _) = search_batch(&[], &[], &[], 1.0, true).unwrap();
        assert!(ei.is_empty());
    }

    #[test]
    fn test_batch_search_errors() {
        let positions = vec![Vector3::new(0.0, 0.0, 0.0)];
        let batch = vec![0, 1]; // mismatch length
        let cells = vec![None];
        assert!(search_batch(&positions, &batch, &cells, 1.0, false).is_err());

        let batch2 = vec![0];
        let cells2 = vec![]; // mismatch n_systems
        assert!(search_batch(&positions, &batch2, &cells2, 1.0, false).is_err());
    }
}
