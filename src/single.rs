use crate::cell::Cell;
use crate::config;
use crate::search::{self, CellList, EdgeResult};
use nalgebra::{Matrix3, Vector3};

pub const AUTO_BOX_MARGIN: f64 = 1.0;

pub fn search_single(
    positions: &[Vector3<f64>],
    cell: Option<(Matrix3<f64>, Vector3<bool>)>,
    cutoff: f64,
    parallel: bool,
) -> Result<EdgeResult, String> {
    let n_atoms = positions.len();
    if n_atoms == 0 {
        return Ok((vec![], vec![], vec![]));
    }

    let cell_inner = if let Some((h_mat, pbc)) = cell {
        Cell::new(h_mat, pbc).map_err(|e| e.to_string())?
    } else {
        let mut min_bound = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max_bound = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in positions {
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

    if n_atoms < config::get_brute_force_threshold() && mic_safe {
        Ok(search::brute_force_search_full(
            &cell_inner,
            positions,
            cutoff,
        ))
    } else {
        let cl = CellList::build(&cell_inner, positions, cutoff);
        if parallel && n_atoms >= config::get_parallel_threshold() {
            Ok(cl.par_search_optimized(&cell_inner, cutoff))
        } else {
            let neighbors = cl.search(&cell_inner, positions, cutoff);
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
            Ok((ei, ej, s))
        }
    }
}

pub fn search_single_multi(
    positions: &[Vector3<f64>],
    cell: Option<(Matrix3<f64>, Vector3<bool>)>,
    cutoffs: &[f64],
    disjoint: bool,
) -> Result<Vec<EdgeResult>, String> {
    let n_atoms = positions.len();
    if n_atoms == 0 {
        return Ok(vec![(vec![], vec![], vec![]); cutoffs.len()]);
    }
    if cutoffs.is_empty() {
        return Ok(vec![]);
    }

    let max_cutoff = cutoffs.iter().cloned().fold(f64::NAN, f64::max);

    let cell_inner = if let Some((h_mat, pbc)) = cell {
        Cell::new(h_mat, pbc).map_err(|e| e.to_string())?
    } else {
        let mut min_bound = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max_bound = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in positions {
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

    if n_atoms < config::get_brute_force_threshold() && mic_safe {
        Ok(search::brute_force_search_multi(
            &cell_inner,
            positions,
            cutoffs,
            disjoint,
        ))
    } else {
        let cl = CellList::build(&cell_inner, positions, max_cutoff);
        Ok(cl.par_search_multi(&cell_inner, cutoffs, disjoint))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_threshold_logic() {
        let threshold = config::get_stack_threshold();
        // This test ensures the default is one of our expected profiles
        assert!(threshold == 1000 || threshold == 800 || threshold == 400);
    }

    #[test]
    fn test_search_single_basic() {
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let (ei, ej, _) = search_single(&positions, None, 1.5, false).unwrap();
        assert_eq!(ei, vec![0]);
        assert_eq!(ej, vec![1]);
    }

    #[test]
    fn test_search_single_multi_basic() {
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.5, 0.0, 0.0),
        ];
        let cutoffs = vec![1.5, 3.0];
        let results = search_single_multi(&positions, None, &cutoffs, false).unwrap();
        assert_eq!(results[0].0.len(), 1); // (0,1)
        assert_eq!(results[1].0.len(), 3); // (0,1), (1,2), (0,2)
    }

    #[test]
    fn test_search_single_multi_disjoint() {
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.1, 0.0, 0.0),
        ];
        let cutoffs = vec![1.5, 3.0];
        let results = search_single_multi(&positions, None, &cutoffs, true).unwrap();
        // Cutoff 1.5: (0,1), (1,2)
        // Cutoff 3.0: (0,2)
        assert_eq!(results[0].0.len(), 2);
        assert_eq!(results[1].0.len(), 1);
    }

    #[test]
    fn test_search_single_serial() {
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        // force serial path (parallel=false)
        let (ei, _, _) = search_single(&positions, None, 1.5, false).unwrap();
        assert_eq!(ei.len(), 1);
    }

    #[test]
    fn test_search_single_parallel() {
        let mut positions = Vec::new();
        for i in 0..25 {
            positions.push(Vector3::new(i as f64, 0.0, 0.0));
        }
        let (ei, _, _) = search_single(&positions, None, 1.1, true).unwrap();
        assert!(!ei.is_empty());
    }

    #[test]
    fn test_search_single_large() {
        // Trigger CellList path (> 500 atoms)
        let mut positions = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    positions.push(Vector3::new(i as f64 * 0.5, j as f64 * 0.5, k as f64 * 0.5));
                }
            }
        }
        let (ei, _, _) = search_single(&positions, None, 0.6, true).unwrap();
        assert!(!ei.is_empty());

        let results = search_single_multi(&positions, None, &[0.6, 1.1], false).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_single_empty() {
        let (ei, _, _) = search_single(&[], None, 1.0, true).unwrap();
        assert!(ei.is_empty());

        let res_multi = search_single_multi(&[], None, &[1.0], false).unwrap();
        assert!(res_multi[0].0.is_empty());

        assert!(
            search_single_multi(&[Vector3::zeros()], None, &[], false)
                .unwrap()
                .is_empty()
        );
    }
}
