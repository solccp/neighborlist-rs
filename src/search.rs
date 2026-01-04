use crate::cell::Cell;
use nalgebra::Vector3;
use rayon::prelude::*;
use tracing::info_span;

pub struct CellList {
    /// particles[sorted_idx] = original_idx
    particles: Vec<usize>,
    /// cell_starts[bin_rank] = start index in particles
    cell_starts: Vec<usize>,
    /// Maps linear bin index (bx + nx*(by + ny*bz)) to Morton rank
    bin_ranks: Vec<usize>,
    /// Wrapped positions in sorted order (index matches sorted_idx).
    pos_wrapped: Vec<Vector3<f64>>,
    /// Wrapping shifts in sorted order (index matches sorted_idx).
    atom_shifts: Vec<Vector3<i32>>,
    num_bins: Vector3<usize>,
    n_search: Vector3<i32>,
}

impl CellList {
    pub fn build(cell: &Cell, positions: &[Vector3<f64>], cutoff: f64) -> Self {
        let _span = info_span!("CellList::build", n_atoms = positions.len()).entered();
        let n_atoms = positions.len();

        // 1. Compute Z-order indices and original positions in parallel
        let mut atom_data: Vec<(u64, usize)> = {
            let _s = info_span!("compute_z_order").entered();
            positions
                .into_par_iter()
                .enumerate()
                .map(|(i, pos)| {
                    let mut frac = cell.to_fractional(pos);
                    frac.x -= frac.x.floor();
                    frac.y -= frac.y.floor();
                    frac.z -= frac.z.floor();
                    (compute_z_order(&frac), i)
                })
                .collect()
        };

        // 2. Sort atoms spatially by global Z-order
        {
            let _s = info_span!("spatial_sort").entered();
            atom_data.sort_unstable_by_key(|&(z, _)| z);
        }

        // 3. Setup bins and compute Morton ranks for bins
        let perp_widths = cell.perpendicular_widths();
        let nx = (perp_widths.x / cutoff).floor() as usize;
        let ny = (perp_widths.y / cutoff).floor() as usize;
        let nz = (perp_widths.z / cutoff).floor() as usize;
        let num_bins = Vector3::new(nx.max(1), ny.max(1), nz.max(1));
        let total_bins = num_bins.x * num_bins.y * num_bins.z;

        let mut bin_ranks = vec![0; total_bins];
        {
            let _s = info_span!("compute_bin_ranks").entered();
            let mut bin_morton: Vec<(u64, usize)> = (0..total_bins)
                .map(|i| {
                    let bx = i % num_bins.x;
                    let by = (i / num_bins.x) % num_bins.y;
                    let bz = i / (num_bins.x * num_bins.y);
                    (interleave_3(bx as u64) | (interleave_3(by as u64) << 1) | (interleave_3(bz as u64) << 2), i)
                })
                .collect();
            bin_morton.sort_unstable_by_key(|&(z, _)| z);
            for (rank, &(_z, linear_idx)) in bin_morton.iter().enumerate() {
                bin_ranks[linear_idx] = rank;
            }
        }

        let n_search = Vector3::new(
            (cutoff * num_bins.x as f64 / perp_widths.x).ceil() as i32,
            (cutoff * num_bins.y as f64 / perp_widths.y).ceil() as i32,
            (cutoff * num_bins.z as f64 / perp_widths.z).ceil() as i32,
        );

        let h_matrix = cell.h();

        // 4. Compute bin counts directly from atom_data
        let mut counts = vec![0; total_bins];
        for &(_z, original_idx) in &atom_data {
            let pos = positions[original_idx];
            let frac = cell.to_fractional(&pos);
            let ux = frac.x - frac.x.floor();
            let uy = frac.y - frac.y.floor();
            let uz = frac.z - frac.z.floor();
            let bx = ((ux * num_bins.x as f64) as usize).min(num_bins.x - 1);
            let by = ((uy * num_bins.y as f64) as usize).min(num_bins.y - 1);
            let bz = ((uz * num_bins.z as f64) as usize).min(num_bins.z - 1);
            let linear_idx = bx + num_bins.x * (by + num_bins.y * bz);
            let rank = bin_ranks[linear_idx];
            counts[rank] += 1;
        }

        let mut cell_starts = vec![0; total_bins + 1];
        let mut accum = 0;
        for i in 0..total_bins {
            cell_starts[i] = accum;
            accum += counts[i];
        }
        cell_starts[total_bins] = accum;

        let mut final_pos_wrapped = vec![Vector3::zeros(); n_atoms];
        let mut final_atom_shifts = vec![Vector3::zeros(); n_atoms];
        let mut final_particles = vec![0; n_atoms];
        let mut current_fill = cell_starts.clone();

        // 5. Fill final arrays
        {
            let _s = info_span!("bin_fill").entered();
            for &(_z, original_idx) in &atom_data {
                let pos = positions[original_idx];
                let frac = cell.to_fractional(&pos);
                let fx = frac.x.floor();
                let fy = frac.y.floor();
                let fz = frac.z.floor();

                let sx = -fx as i32;
                let sy = -fy as i32;
                let sz = -fz as i32;
                let atom_shift = Vector3::new(sx, sy, sz);

                let s_vec = h_matrix * Vector3::new(-sx as f64, -sy as f64, -sz as f64);
                let wrapped_pos = pos - s_vec;

                let ux = frac.x - fx;
                let uy = frac.y - fy;
                let uz = frac.z - fz;

                let bx = ((ux * num_bins.x as f64) as usize).min(num_bins.x - 1);
                let by = ((uy * num_bins.y as f64) as usize).min(num_bins.y - 1);
                let bz = ((uz * num_bins.z as f64) as usize).min(num_bins.z - 1);

                let linear_idx = bx + num_bins.x * (by + num_bins.y * bz);
                let rank = bin_ranks[linear_idx];
                
                let loc = current_fill[rank];
                final_pos_wrapped[loc] = wrapped_pos;
                final_atom_shifts[loc] = atom_shift;
                final_particles[loc] = original_idx;
                current_fill[rank] += 1;
            }
        }

        Self {
            particles: final_particles,
            cell_starts,
            bin_ranks,
            pos_wrapped: final_pos_wrapped,
            atom_shifts: final_atom_shifts,
            num_bins,
            n_search,
        }
    }

    pub fn get_atoms_in_bin(&self, bx: usize, by: usize, bz: usize) -> &[usize] {
        if bx >= self.num_bins.x || by >= self.num_bins.y || bz >= self.num_bins.z {
            return &[];
        }
        let linear_idx = bx + self.num_bins.x * (by + self.num_bins.y * bz);
        let rank = self.bin_ranks[linear_idx];
        &self.particles[self.cell_starts[rank]..self.cell_starts[rank + 1]]
    }

    pub fn search(
        &self,
        cell: &Cell,
        _positions: &[Vector3<f64>],
        cutoff: f64,
    ) -> Vec<(usize, usize, i32, i32, i32)> {
        let cutoff_sq = cutoff * cutoff;
        let n_atoms = self.particles.len();
        let mut neighbors = Vec::new();

        for i in 0..n_atoms {
            self.search_atom_neighbors(i, cell, cutoff_sq, &mut neighbors);
        }
        neighbors
    }

    pub fn par_search_optimized(&self, cell: &Cell, cutoff: f64) -> (Vec<i64>, Vec<i64>, Vec<i32>) {
        let _span = info_span!("CellList::par_search_optimized").entered();
        let cutoff_sq = cutoff * cutoff;
        let n_atoms = self.particles.len();

        let counts: Vec<u32> = {
            let _s = info_span!("count_pass").entered();
            (0..n_atoms)
                .into_par_iter()
                .map(|i| self.count_atom_neighbors(i, cell, cutoff_sq) as u32)
                .collect()
        };

        let mut offsets = Vec::with_capacity(n_atoms + 1);
        let mut total: usize = 0;
        for &c in &counts {
            offsets.push(total);
            total += c as usize;
        }
        offsets.push(total);

        let mut edge_i = vec![0i64; total];
        let mut edge_j = vec![0i64; total];
        let mut shifts = vec![0i32; total * 3];

        let edge_i_ptr = edge_i.as_mut_ptr() as usize;
        let edge_j_ptr = edge_j.as_mut_ptr() as usize;
        let shifts_ptr = shifts.as_mut_ptr() as usize;

        {
            let _s = info_span!("fill_pass").entered();
            (0..n_atoms).into_par_iter().for_each(|i| {
                let offset = offsets[i];
                unsafe {
                    let ei = std::slice::from_raw_parts_mut(edge_i_ptr as *mut i64, total);
                    let ej = std::slice::from_raw_parts_mut(edge_j_ptr as *mut i64, total);
                    let s = std::slice::from_raw_parts_mut(shifts_ptr as *mut i32, total * 3);

                    self.fill_atom_neighbors(i, cell, cutoff_sq, offset, ei, ej, s);
                }
            });
        }

        (edge_i, edge_j, shifts)
    }

    fn count_atom_neighbors(&self, i: usize, cell: &Cell, cutoff_sq: f64) -> usize {
        let pos_i_w = self.pos_wrapped[i];
        let h_matrix = cell.h();

        let frac_i = cell.to_fractional(&pos_i_w);
        let bx = ((frac_i.x * self.num_bins.x as f64) as i32).min(self.num_bins.x as i32 - 1);
        let by = ((frac_i.y * self.num_bins.y as f64) as i32).min(self.num_bins.y as i32 - 1);
        let bz = ((frac_i.z * self.num_bins.z as f64) as i32).min(self.num_bins.z as i32 - 1);

        let i_orig = self.particles[i];

        let mut count = 0;
        for dx in -self.n_search.x..=self.n_search.x {
            for dy in -self.n_search.y..=self.n_search.y {
                for dz in -self.n_search.z..=self.n_search.z {
                    let (nbx, sx) = div_mod(bx + dx, self.num_bins.x as i32);
                    let (nby, sy) = div_mod(by + dy, self.num_bins.y as i32);
                    let (nbz, sz) = div_mod(bz + dz, self.num_bins.z as i32);

                    let linear_idx = nbx + self.num_bins.x * (nby + self.num_bins.y * nbz);
                    let rank = self.bin_ranks[linear_idx];
                    let start_j = self.cell_starts[rank];
                    let end_j = self.cell_starts[rank + 1];

                    if start_j == end_j {
                        continue;
                    }

                    let offset_vec = h_matrix * Vector3::new(sx as f64, sy as f64, sz as f64);

                    for sorted_idx_j in start_j..end_j {
                        let j_orig = self.particles[sorted_idx_j];
                        if i_orig >= j_orig {
                            continue;
                        }
                        let disp: Vector3<f64> = (self.pos_wrapped[sorted_idx_j] - pos_i_w) + offset_vec;
                        if disp.norm_squared() < cutoff_sq {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    fn fill_atom_neighbors(
        &self,
        i: usize,
        cell: &Cell,
        cutoff_sq: f64,
        mut offset: usize,
        edge_i: &mut [i64],
        edge_j: &mut [i64],
        shifts: &mut [i32],
    ) {
        let pos_i_w = self.pos_wrapped[i];
        let s_i = self.atom_shifts[i];
        let h_matrix = cell.h();

        let frac_i = cell.to_fractional(&pos_i_w);
        let bx = ((frac_i.x * self.num_bins.x as f64) as i32).min(self.num_bins.x as i32 - 1);
        let by = ((frac_i.y * self.num_bins.y as f64) as i32).min(self.num_bins.y as i32 - 1);
        let bz = ((frac_i.z * self.num_bins.z as f64) as i32).min(self.num_bins.z as i32 - 1);

        let i_orig = self.particles[i];

        for dx in -self.n_search.x..=self.n_search.x {
            for dy in -self.n_search.y..=self.n_search.y {
                for dz in -self.n_search.z..=self.n_search.z {
                    let (nbx, sx) = div_mod(bx + dx, self.num_bins.x as i32);
                    let (nby, sy) = div_mod(by + dy, self.num_bins.y as i32);
                    let (nbz, sz) = div_mod(bz + dz, self.num_bins.z as i32);

                    let linear_idx = nbx + self.num_bins.x * (nby + self.num_bins.y * nbz);
                    let rank = self.bin_ranks[linear_idx];
                    let start_j = self.cell_starts[rank];
                    let end_j = self.cell_starts[rank + 1];

                    if start_j == end_j {
                        continue;
                    }

                    let offset_vec = h_matrix * Vector3::new(sx as f64, sy as f64, sz as f64);

                    for sorted_idx_j in start_j..end_j {
                        let j_orig = self.particles[sorted_idx_j];
                        if i_orig >= j_orig {
                            continue;
                        }
                        let disp: Vector3<f64> = (self.pos_wrapped[sorted_idx_j] - pos_i_w) + offset_vec;
                        if disp.norm_squared() < cutoff_sq {
                            edge_i[offset] = i_orig as i64;
                            edge_j[offset] = j_orig as i64;

                            let s_j = self.atom_shifts[sorted_idx_j];
                            shifts[offset * 3] = s_j.x - s_i.x + sx;
                            shifts[offset * 3 + 1] = s_j.y - s_i.y + sy;
                            shifts[offset * 3 + 2] = s_j.z - s_i.z + sz;

                            offset += 1;
                        }
                    }
                }
            }
        }
    }

    fn search_atom_neighbors(
        &self,
        i: usize,
        cell: &Cell,
        cutoff_sq: f64,
        neighbors: &mut Vec<(usize, usize, i32, i32, i32)>,
    ) {
        let pos_i_w = self.pos_wrapped[i];
        let s_i = self.atom_shifts[i];
        let h_matrix = cell.h();

        let frac_i = cell.to_fractional(&pos_i_w);
        let bx = ((frac_i.x * self.num_bins.x as f64) as i32).min(self.num_bins.x as i32 - 1);
        let by = ((frac_i.y * self.num_bins.y as f64) as i32).min(self.num_bins.y as i32 - 1);
        let bz = ((frac_i.z * self.num_bins.z as f64) as i32).min(self.num_bins.z as i32 - 1);

        let i_orig = self.particles[i];

        for dx in -self.n_search.x..=self.n_search.x {
            for dy in -self.n_search.y..=self.n_search.y {
                for dz in -self.n_search.z..=self.n_search.z {
                    let (nbx, sx) = div_mod(bx + dx, self.num_bins.x as i32);
                    let (nby, sy) = div_mod(by + dy, self.num_bins.y as i32);
                    let (nbz, sz) = div_mod(bz + dz, self.num_bins.z as i32);

                    let linear_idx = nbx + self.num_bins.x * (nby + self.num_bins.y * nbz);
                    let rank = self.bin_ranks[linear_idx];
                    let start_j = self.cell_starts[rank];
                    let end_j = self.cell_starts[rank + 1];

                    if start_j == end_j {
                        continue;
                    }

                    let offset_vec = h_matrix * Vector3::new(sx as f64, sy as f64, sz as f64);

                    for sorted_idx_j in start_j..end_j {
                        let j_orig = self.particles[sorted_idx_j];
                        if i_orig >= j_orig {
                            continue;
                        }
                        let disp: Vector3<f64> = (self.pos_wrapped[sorted_idx_j] - pos_i_w) + offset_vec;

                        if disp.norm_squared() < cutoff_sq {
                            let s_j = self.atom_shifts[sorted_idx_j];
                            let s_total_x = s_j.x - s_i.x + sx;
                            let s_total_y = s_j.y - s_i.y + sy;
                            let s_total_z = s_j.z - s_i.z + sz;
                            neighbors.push((i_orig, j_orig, s_total_x, s_total_y, s_total_z));
                        }
                    }
                }
            }
        }
    }
}

fn div_mod(val: i32, max: i32) -> (usize, i32) {
    let rem = val.rem_euclid(max);
    let shift = val.div_euclid(max);
    (rem as usize, shift)
}

/// Computes a 64-bit Morton (Z-order) index for fractional coordinates [0, 1).
fn compute_z_order(frac: &Vector3<f64>) -> u64 {
    let x = (frac.x.clamp(0.0, 0.999999) * (1u64 << 21) as f64) as u64;
    let y = (frac.y.clamp(0.0, 0.999999) * (1u64 << 21) as f64) as u64;
    let z = (frac.z.clamp(0.0, 0.999999) * (1u64 << 21) as f64) as u64;

    interleave_3(x) | (interleave_3(y) << 1) | (interleave_3(z) << 2)
}

fn interleave_3(mut x: u64) -> u64 {
    x &= 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffffu64;
    x = (x | x << 16) & 0x1f0000ff0000ffu64;
    x = (x | x << 8) & 0x100f00f00f00f00fu64;
    x = (x | x << 4) & 0x10c30c30c30c30c3u64;
    x = (x | x << 2) & 0x1249249249249249u64;
    x
}

pub fn brute_force_search(
    cell: &Cell,
    positions: &[Vector3<f64>],
    cutoff: f64,
) -> Vec<(usize, usize)> {
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
    fn test_z_order_calculation() {
        let p1 = Vector3::new(0.1, 0.1, 0.1);
        let p2 = Vector3::new(0.1, 0.1, 0.11);
        let p3 = Vector3::new(0.9, 0.9, 0.9);

        let z1 = compute_z_order(&p1);
        let z2 = compute_z_order(&p2);
        let z3 = compute_z_order(&p3);

        assert!(z1 < z2);
        assert!(z2 < z3);
        
        // Test clamping
        let p_out = Vector3::new(1.1, -0.1, 0.5);
        let z_out = compute_z_order(&p_out);
        assert!(z_out > 0);
    }

    #[test]
    fn test_spatial_reordering_correctness() {
        let h = Matrix3::identity() * 10.0;
        let cell = Cell::new(h).unwrap();
        
        // Atoms that are far in original index but close in space
        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0), // idx 0
            Vector3::new(9.0, 9.0, 9.0), // idx 1
            Vector3::new(1.1, 1.1, 1.1), // idx 2
        ];
        
        let cutoff = 2.0;
        let cl = CellList::build(&cell, &positions, cutoff);
        
        // Check that atoms 0 and 2 are in the same bin and thus contiguous in pos_wrapped
        let bin0 = cl.get_atoms_in_bin(0, 0, 0);
        assert_eq!(bin0.len(), 2);
        
        // Find indices of 0 and 2 in the sorted particles list
        let mut loc0 = None;
        let mut loc2 = None;
        for (i, &orig_idx) in cl.particles.iter().enumerate() {
            if orig_idx == 0 { loc0 = Some(i); }
            if orig_idx == 2 { loc2 = Some(i); }
        }
        
        let loc0 = loc0.unwrap();
        let loc2 = loc2.unwrap();
        
        // They should be adjacent in memory (in the same bin)
        assert!((loc0 as isize - loc2 as isize).abs() == 1);
        
        // Neighbor search should still find (0, 2)
        let neighbors = cl.search(&cell, &positions, cutoff);
        assert!(neighbors.iter().any(|&(i, j, _, _, _)| (i == 0 && j == 2) || (i == 2 && j == 0)));
    }

    #[test]
    fn test_brute_force_reference() {
        let h = Matrix3::new(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0);
        let cell = Cell::new(h).unwrap();

        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 3.5, 1.0),
            Vector3::new(8.5, 1.0, 1.0),
        ];

        let neighbors = brute_force_search(&cell, &positions, 3.0);

        assert_eq!(neighbors.len(), 2);
        let mut sorted = neighbors.clone();
        sorted.sort();

        assert_eq!(sorted[0], (0, 1));
        assert_eq!(sorted[1], (0, 2));
    }

    #[test]
    fn test_cell_list_structure() {
        let h = Matrix3::new(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0);
        let cell = Cell::new(h).unwrap();

        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0), // Bin [0, 0, 0]
            Vector3::new(9.0, 9.0, 9.0), // Bin [2, 2, 2]
            Vector3::new(1.1, 1.1, 1.1), // Bin [0, 0, 0]
        ];

        let cl = CellList::build(&cell, &positions, 3.0);

        assert_eq!(cl.num_bins, Vector3::new(3, 3, 3));

        let bin0 = cl.get_atoms_in_bin(0, 0, 0);
        assert_eq!(bin0.len(), 2);
        assert!(bin0.contains(&0));
        assert!(bin0.contains(&2));

        let bin2 = cl.get_atoms_in_bin(2, 2, 2);
        assert_eq!(bin2.len(), 1);
        assert!(bin2.contains(&1));

        let bin1 = cl.get_atoms_in_bin(1, 1, 1);
        assert!(bin1.is_empty());
    }

    #[test]
    fn test_cell_list_search_vs_brute_force() {
        let h = Matrix3::new(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0);
        let cell = Cell::new(h).unwrap();

        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.2, 1.2, 1.2), // Neighbor to 0
            Vector3::new(9.8, 9.8, 9.8), // Neighbor to 0 via PBC
            Vector3::new(5.0, 5.0, 5.0), // Isolated
        ];

        let cutoff = 2.0;

        let expected = brute_force_search(&cell, &positions, cutoff);

        let cl = CellList::build(&cell, &positions, cutoff);
        let mut result: Vec<(usize, usize)> = cl
            .search(&cell, &positions, cutoff)
            .into_iter()
            .map(|(i, j, _, _, _)| (i, j))
            .collect();

        let mut expected_sorted = expected.clone();
        expected_sorted.sort();

        result.sort();

        assert_eq!(result, expected_sorted);
    }

    #[cfg(feature = "dhat-heap")]
    #[test]
    fn test_memory_profile() {
        let _profiler = dhat::Profiler::new_heap();

        let h = Matrix3::identity() * 20.0;
        let cell = Cell::new(h).unwrap();

        let mut positions = Vec::new();
        for i in 0..100 {
            for j in 0..100 {
                positions.push(Vector3::new(i as f64 * 0.2, j as f64 * 0.2, 0.0));
            }
        }

        let cutoff = 3.0;
        let cl = CellList::build(&cell, &positions, cutoff);
        let _ = cl.par_search_optimized(&cell, cutoff);
    }

    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_search_correctness(
                box_size in 10.0..20.0,
                cutoff in 1.0..3.0,
                positions_data in prop::collection::vec(prop::collection::vec(0.0..20.0, 3), 2..50)
            ) {
                let h = Matrix3::identity() * box_size;
                let cell = Cell::new(h).unwrap();

                let mut positions = Vec::new();
                for p in positions_data {
                    positions.push(Vector3::new(
                        p[0] % box_size,
                        p[1] % box_size,
                        p[2] % box_size,
                    ));
                }

                let expected = brute_force_search(&cell, &positions, cutoff);
                let cl = CellList::build(&cell, &positions, cutoff);
                let mut result: Vec<(usize, usize)> = cl.search(&cell, &positions, cutoff).into_iter().map(|(i, j, _, _, _)| (i, j)).collect();

                let mut expected_sorted = expected.clone();
                expected_sorted.sort();
                result.sort();

                assert_eq!(result, expected_sorted);
            }
        }
    }
}