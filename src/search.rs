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
    fn test_brute_force_reference() {
        let h = Matrix3::new(
            10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0,
        );
        let cell = Cell::new(h).unwrap();
        
        // Two atoms within cutoff (dist = 2.0), one outside (dist = 8.0 -> pbc 2.0?)
        // Let's make it clear. Cutoff = 3.0.
        // let positions = vec![
        //     Vector3::new(1.0, 1.0, 1.0), 
        //     Vector3::new(1.0, 3.0, 1.0), // Dist = 2.0 < 3.0 (Neighbor)
        //     Vector3::new(8.0, 1.0, 1.0), // Dist = 7.0. PBC -> -3.0. Dist = 3.0 (Boundary case? Let's make it 3.1)
        // ];
        // Wait, 8.0 to 1.0 is distance 7.0.
        // Image at -2.0. Dist to 1.0 is 3.0.
        // Let's use 8.1. Dist = 7.1. Img at -1.9. Dist to 1.0 is 2.9 (Neighbor via PBC).

        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0), // 0
            Vector3::new(1.0, 3.5, 1.0), // 1. Dist = 2.5 < 3.0 -> Match (0,1)
            Vector3::new(8.5, 1.0, 1.0), // 2. Dist(0,2) = 7.5. Wrapped diff = -2.5. Dist = 2.5 < 3.0 -> Match (0,2)
        ];

        let neighbors = brute_force_search(&cell, &positions, 3.0);
        
        // Expected pairs: (0,1), (0,2). (1,2) dist? 
        // 1=(1, 3.5, 1), 2=(8.5, 1, 1).
        // dx = 7.5 -> -2.5. dy = -2.5. dz = 0.
        // d^2 = 6.25 + 6.25 = 12.5. d = 3.53 > 3.0. No match.
        
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
}
