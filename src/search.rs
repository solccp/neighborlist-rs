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
        // 1. Calculate number of bins
        // We use the perpendicular widths of the cell to determine safe bin counts.
        // For a general cell, this is related to the lengths of the reciprocal vectors.
        // Simplified approach for Phase 1 (focusing on Orthorhombic-ish logic first, but generally safe):
        // N_i = floor(L_i / cutoff)
        
        // A robust way for general cells: N_i = floor(1 / |h_inv_i| / cutoff) ??
        // Actually, strictly: The distance between parallel planes defined by a lattice vector must be > cutoff.
        // d_1 = Vol / |a2 x a3|.
        // Let's stick to the simplest projection logic: 
        // We map fractional [0,1] to [0, N].
        // To satisfy the cell list condition, the "width" of a bin in Cartesian space must be >= cutoff.
        // width_i = |a_i| / N_i >= cutoff  => N_i <= |a_i| / cutoff.
        
        // Let's use the column vectors lengths for now. 
        // NOTE: For highly skewed triclinic cells, this needs strictly perpendicular widths.
        // For Phase 1, we assume we want to support general cells, so let's try to be somewhat correct.
        // If we just use lengths of cell vectors, we might violate the condition if the angle is small.
        // But for now, let's implement the standard logic:
        // num_bins[i] = floor(|a_i| / cutoff)
        
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

            // Calculate bin indices
            // Clamp to ensure 1.0 doesn't go out of bounds (though floor of 1.0-epsilon should be fine)
            // But floating point math is tricky.
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

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