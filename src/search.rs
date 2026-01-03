use rayon::prelude::*;
use crate::cell::Cell;
use nalgebra::Vector3;

const EMPTY: usize = usize::MAX;

pub struct CellList {
    head: Vec<usize>,
    next: Vec<usize>,
    num_bins: Vector3<usize>,
}

impl CellList {
    pub fn build(cell: &Cell, positions: &[Vector3<f64>], cutoff: f64) -> Self {
        let h = cell.h();
        let a = h.column(0).norm();
        let b = h.column(1).norm();
        let c = h.column(2).norm();

        let nx = (a / cutoff).floor() as usize;
        let ny = (b / cutoff).floor() as usize;
        let nz = (c / cutoff).floor() as usize;

        // Ensure at least 1 bin
        let num_bins = Vector3::new(nx.max(1), ny.max(1), nz.max(1));
        
        let total_bins = num_bins.x * num_bins.y * num_bins.z;
        let mut head = vec![EMPTY; total_bins];
        let mut next = vec![EMPTY; positions.len()];

        for (i, pos) in positions.iter().enumerate() {
            // Get fractional coordinates wrapped to [0, 1)
            let mut frac = cell.to_fractional(pos);
            frac.x -= frac.x.floor();
            frac.y -= frac.y.floor();
            frac.z -= frac.z.floor();

            let bx = ((frac.x * num_bins.x as f64) as usize).min(num_bins.x - 1);
            let by = ((frac.y * num_bins.y as f64) as usize).min(num_bins.y - 1);
            let bz = ((frac.z * num_bins.z as f64) as usize).min(num_bins.z - 1);

            let bin_idx = bx + num_bins.x * (by + num_bins.y * bz);

            next[i] = head[bin_idx];
            head[bin_idx] = i;
        }

        Self {
            head,
            next,
            num_bins,
        }
    }

    /// Helper to get all atoms in a specific bin (kx, ky, kz)
    pub fn get_atoms_in_bin(&self, bx: usize, by: usize, bz: usize) -> Vec<usize> {
        let mut atoms = Vec::new();
        if bx >= self.num_bins.x || by >= self.num_bins.y || bz >= self.num_bins.z {
            return atoms;
        }

        let bin_idx = bx + self.num_bins.x * (by + self.num_bins.y * bz);
        let mut curr = self.head[bin_idx];

        while curr != EMPTY {
            atoms.push(curr);
            curr = self.next[curr];
        }
        atoms
    }

    pub fn search(&self, cell: &Cell, positions: &[Vector3<f64>], cutoff: f64) -> Vec<(usize, usize)> {
        let mut neighbors = Vec::new();
        let cutoff_sq = cutoff * cutoff;
        let n_bins = self.num_bins;
        let nx = n_bins.x as i32;
        let ny = n_bins.y as i32;
        let nz = n_bins.z as i32;

        // Iterate over all bins
        for bx in 0..nx {
            for by in 0..ny {
                for bz in 0..nz {
                    self.search_bin(bx, by, bz, nx, ny, nz, n_bins, cell, positions, cutoff_sq, &mut neighbors);
                }
            }
        }
        neighbors
    }

    pub fn par_search(&self, cell: &Cell, positions: &[Vector3<f64>], cutoff: f64) -> Vec<(usize, usize)> {
        let cutoff_sq = cutoff * cutoff;
        let n_bins = self.num_bins;
        let nx = n_bins.x as i32;
        let ny = n_bins.y as i32;
        let nz = n_bins.z as i32;

        // Parallelize over the x-dimension
        (0..nx).into_par_iter().map(|bx| {
            let mut local_neighbors = Vec::new();
            for by in 0..ny {
                for bz in 0..nz {
                    self.search_bin(bx, by, bz, nx, ny, nz, n_bins, cell, positions, cutoff_sq, &mut local_neighbors);
                }
            }
            local_neighbors
        }).reduce(Vec::new, |mut a, b| {
            a.extend(b);
            a
        })
    }

    // Helper method to process a single bin
    fn search_bin(
        &self, 
        bx: i32, by: i32, bz: i32, 
        nx: i32, ny: i32, nz: i32, 
        n_bins: Vector3<usize>,
        cell: &Cell, 
        positions: &[Vector3<f64>], 
        cutoff_sq: f64,
        neighbors: &mut Vec<(usize, usize)>
    ) {
        let bin_idx = (bx as usize) + n_bins.x * ((by as usize) + n_bins.y * (bz as usize));
        let mut i = self.head[bin_idx];

        while i != EMPTY {
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let nbx = (bx + dx).rem_euclid(nx) as usize;
                        let nby = (by + dy).rem_euclid(ny) as usize;
                        let nbz = (bz + dz).rem_euclid(nz) as usize;
                        
                        let n_bin_idx = nbx + n_bins.x * (nby + n_bins.y * nbz);
                        let mut j = self.head[n_bin_idx];

                        while j != EMPTY {
                            if i < j {
                                let (_, disp) = cell.get_shift_and_displacement(&positions[i], &positions[j]);
                                if disp.norm_squared() < cutoff_sq {
                                    neighbors.push((i, j));
                                }
                            }
                            j = self.next[j];
                        }
                    }
                }
            }
            i = self.next[i];
        }
    }
}

pub fn brute_force_search(cell: &Cell, positions: &[Vector3<f64>], cutoff: f64) -> Vec<(usize, usize)> {
    let mut neighbors = Vec::new();
    let n = positions.len();
    let cutoff_sq = cutoff * cutoff;

    for i in 0..n {
        for j in (i + 1)..n {
            let (_, disp) = cell.get_shift_and_displacement(&positions[i], &positions[j]);
            if disp.norm_squared() < cutoff_sq {
                neighbors.push((i, j));
            }
        }
    }
    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    #[test]
    fn test_par_search() {
         let h = Matrix3::new(
            10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0,
        );
        let cell = Cell::new(h).unwrap();
        
        // Use enough atoms to likely trigger parallelism if configured, 
        // but verify logic correctness primarily.
        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.2, 1.2, 1.2), 
            Vector3::new(9.8, 9.8, 9.8),
            Vector3::new(5.0, 5.0, 5.0),
        ];

        let cutoff = 2.0;
        let cl = CellList::build(&cell, &positions, cutoff);
        
        let seq_result = cl.search(&cell, &positions, cutoff);
        let par_result = cl.par_search(&cell, &positions, cutoff);

        let mut seq_sorted = seq_result.clone();
        seq_sorted.sort();

        let mut par_sorted = par_result.clone();
        par_sorted.sort();

        assert_eq!(par_sorted, seq_sorted);
    }

    #[test]
    fn test_brute_force_reference() {
        let h = Matrix3::new(
            10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0,
        );
        let cell = Cell::new(h).unwrap();
        
        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0), // 0
            Vector3::new(1.0, 3.5, 1.0), // 1. Dist = 2.5 < 3.0 -> Match (0,1)
            Vector3::new(8.5, 1.0, 1.0), // 2. Dist(0,2) = 7.5. Wrapped diff = -2.5. Dist = 2.5 < 3.0 -> Match (0,2)
        ];

        let neighbors = brute_force_search(&cell, &positions, 3.0);
        
        assert_eq!(neighbors.len(), 2);
        // Sort for deterministic check
        let mut sorted = neighbors.clone();
        sorted.sort();
        
        assert_eq!(sorted[0], (0, 1));
        assert_eq!(sorted[1], (0, 2));
    }

    #[test]
    fn test_cell_list_structure() {
        let h = Matrix3::new(
            10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0,
        );
        let cell = Cell::new(h).unwrap();
        
        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0), // Bin [0, 0, 0] expected (1/10 < 1/3)
            Vector3::new(9.0, 9.0, 9.0), // Bin [2, 2, 2] expected (9/10 > 2/3)
            Vector3::new(1.1, 1.1, 1.1), // Bin [0, 0, 0]
        ];

        // Cutoff 3.0 -> N=3 (bins of size 3.33)
        let cl = CellList::build(&cell, &positions, 3.0);
        
        assert_eq!(cl.num_bins, Vector3::new(3, 3, 3));

        // Check Bin [0, 0, 0]
        let bin0 = cl.get_atoms_in_bin(0, 0, 0);
        assert_eq!(bin0.len(), 2);
        assert!(bin0.contains(&0));
        assert!(bin0.contains(&2));

        // Check Bin [2, 2, 2]
        let bin2 = cl.get_atoms_in_bin(2, 2, 2);
        assert_eq!(bin2.len(), 1);
        assert!(bin2.contains(&1));

        // Check empty bin
        let bin1 = cl.get_atoms_in_bin(1, 1, 1);
        assert!(bin1.is_empty());
    }

    #[test]
    fn test_wrapping_behavior() {
         let h = Matrix3::new(
            10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0,
        );
        let cell = Cell::new(h).unwrap();
        
        let positions = vec![
            Vector3::new(11.0, 1.0, 1.0), // Should wrap to 1.0 -> Bin [0, 0, 0]
            Vector3::new(-1.0, 1.0, 1.0), // Should wrap to 9.0 -> Bin [2, 0, 0] (cutoff 3.0 -> 3 bins)
        ];

        let cl = CellList::build(&cell, &positions, 3.0);
        
        let bin0 = cl.get_atoms_in_bin(0, 0, 0);
        assert!(bin0.contains(&0));

        let bin2 = cl.get_atoms_in_bin(2, 0, 0);
        assert!(bin2.contains(&1));
    }

    #[test]
    fn test_cell_list_search_vs_brute_force() {
        let h = Matrix3::new(
            10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0,
        );
        let cell = Cell::new(h).unwrap();
        
        // Random-ish positions
        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.2, 1.2, 1.2), // Neighbor to 0
            Vector3::new(9.8, 9.8, 9.8), // Neighbor to 0 via PBC
            Vector3::new(5.0, 5.0, 5.0), // Isolated
        ];

        let cutoff = 2.0;
        
        let expected = brute_force_search(&cell, &positions, cutoff);
        
        let cl = CellList::build(&cell, &positions, cutoff);
        let result = cl.search(&cell, &positions, cutoff);

        // Sort both for comparison
        let mut expected_sorted = expected.clone();
        expected_sorted.sort();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, expected_sorted);
    }
}
