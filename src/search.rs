use crate::cell::Cell;
use nalgebra::Vector3;
use rayon::prelude::*;
use tracing::info_span;

// Internal tuning parameters
const BRUTE_FORCE_CAPACITY_FACTOR: usize = 10;
const Z_ORDER_BITS: u32 = 21;
const Z_ORDER_CLAMP_MAX: f64 = 0.999999;
const PARALLEL_TASKS_PER_THREAD: usize = 64;

pub type EdgeResult = (Vec<i64>, Vec<i64>, Vec<i32>);

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
                    (
                        interleave_3(bx as u64)
                            | (interleave_3(by as u64) << 1)
                            | (interleave_3(bz as u64) << 2),
                        i,
                    )
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

    pub fn par_search_optimized(&self, cell: &Cell, cutoff: f64) -> EdgeResult {
        let _span = info_span!("CellList::par_search_optimized").entered();
        let cutoff_sq = cutoff * cutoff;
        let n_atoms = self.particles.len();

        let num_threads = rayon::current_num_threads();
        let min_len = (n_atoms / (num_threads * PARALLEL_TASKS_PER_THREAD)).max(1);

        let results: Vec<(i64, i64, i32, i32, i32)> = (0..n_atoms)
            .into_par_iter()
            .with_min_len(min_len)
            .flat_map(|i| {
                let mut local_neighbors = Vec::new();
                self.search_atom_neighbors_collect(i, cell, cutoff_sq, &mut local_neighbors);
                local_neighbors
            })
            .collect();

        let n_edges = results.len();
        let mut edge_i = Vec::with_capacity(n_edges);
        let mut edge_j = Vec::with_capacity(n_edges);
        let mut shifts = Vec::with_capacity(n_edges * 3);

        for (i, j, sx, sy, sz) in results {
            edge_i.push(i);
            edge_j.push(j);
            shifts.push(sx);
            shifts.push(sy);
            shifts.push(sz);
        }

        (edge_i, edge_j, shifts)
    }

    pub fn par_search_multi(
        &self,
        cell: &Cell,
        cutoffs: &[f64],
        disjoint: bool,
    ) -> Vec<EdgeResult> {
        let _span = info_span!("CellList::par_search_multi").entered();
        let n_atoms = self.particles.len();
        let n_cutoffs = cutoffs.len();

        let num_threads = rayon::current_num_threads();
        let min_len = (n_atoms / (num_threads * PARALLEL_TASKS_PER_THREAD)).max(1);

        let results: Vec<Vec<Vec<(i64, i64, i32, i32, i32)>>> = (0..n_atoms)
            .into_par_iter()
            .with_min_len(min_len)
            .map(|i| {
                let mut atom_results = vec![Vec::new(); n_cutoffs];
                self.search_atom_neighbors_multi_collect(
                    i,
                    cell,
                    cutoffs,
                    disjoint,
                    &mut atom_results,
                );
                atom_results
            })
            .collect();

        let mut final_results = Vec::with_capacity(n_cutoffs);
        for k in 0..n_cutoffs {
            let total_at_k: usize = results.iter().map(|atom_res| atom_res[k].len()).sum();
            let mut edge_i = Vec::with_capacity(total_at_k);
            let mut edge_j = Vec::with_capacity(total_at_k);
            let mut shifts = Vec::with_capacity(total_at_k * 3);

            for atom_res in &results {
                for (i, j, sx, sy, sz) in &atom_res[k] {
                    edge_i.push(*i);
                    edge_j.push(*j);
                    shifts.push(*sx);
                    shifts.push(*sy);
                    shifts.push(*sz);
                }
            }
            final_results.push((edge_i, edge_j, shifts));
        }
        final_results
    }

    fn search_atom_neighbors_multi_collect(
        &self,
        i: usize,
        cell: &Cell,
        cutoffs: &[f64],
        disjoint: bool,
        results: &mut [Vec<(i64, i64, i32, i32, i32)>],
    ) {
        let n_cutoffs = cutoffs.len();
        let pos_i_w = self.pos_wrapped[i];
        let s_i = self.atom_shifts[i];
        let h_matrix = cell.h();

        let mut sorted_indices: Vec<usize> = (0..n_cutoffs).collect();
        sorted_indices.sort_by(|&a, &b| cutoffs[a].partial_cmp(&cutoffs[b]).unwrap());
        let sorted_cutoffs_sq: Vec<f64> = sorted_indices
            .iter()
            .map(|&k| cutoffs[k] * cutoffs[k])
            .collect();
        let max_cutoff_sq = sorted_cutoffs_sq.iter().cloned().next_back().unwrap_or(0.0);

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

                    let offset_vec = h_matrix * Vector3::new(sx as f64, sy as f64, sz as f64);

                    for sorted_idx_j in start_j..end_j {
                        let j_orig = self.particles[sorted_idx_j];
                        if i_orig >= j_orig {
                            continue;
                        }
                        let disp: Vector3<f64> =
                            (self.pos_wrapped[sorted_idx_j] - pos_i_w) + offset_vec;
                        let dist_sq = disp.norm_squared();

                        if dist_sq < max_cutoff_sq {
                            let mut computed_shifts = false;
                            let mut sx_diff = 0;
                            let mut sy_diff = 0;
                            let mut sz_diff = 0;

                            for (k, &rc_sq) in sorted_cutoffs_sq.iter().enumerate() {
                                if dist_sq < rc_sq {
                                    if !computed_shifts {
                                        let s_j = self.atom_shifts[sorted_idx_j];
                                        sx_diff = s_j.x - s_i.x + sx;
                                        sy_diff = s_j.y - s_i.y + sy;
                                        sz_diff = s_j.z - s_i.z + sz;
                                        computed_shifts = true;
                                    }

                                    results[sorted_indices[k]].push((
                                        i_orig as i64,
                                        j_orig as i64,
                                        sx_diff,
                                        sy_diff,
                                        sz_diff,
                                    ));

                                    if disjoint {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn search_atom_neighbors_collect(
        &self,
        i: usize,
        cell: &Cell,
        cutoff_sq: f64,
        neighbors: &mut Vec<(i64, i64, i32, i32, i32)>,
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
                        let disp: Vector3<f64> =
                            (self.pos_wrapped[sorted_idx_j] - pos_i_w) + offset_vec;
                        if disp.norm_squared() < cutoff_sq {
                            let s_j = self.atom_shifts[sorted_idx_j];
                            neighbors.push((
                                i_orig as i64,
                                j_orig as i64,
                                s_j.x - s_i.x + sx,
                                s_j.y - s_i.y + sy,
                                s_j.z - s_i.z + sz,
                            ));
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
                        let disp: Vector3<f64> =
                            (self.pos_wrapped[sorted_idx_j] - pos_i_w) + offset_vec;

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
    let x = (frac.x.clamp(0.0, Z_ORDER_CLAMP_MAX) * (1u64 << Z_ORDER_BITS) as f64) as u64;
    let y = (frac.y.clamp(0.0, Z_ORDER_CLAMP_MAX) * (1u64 << Z_ORDER_BITS) as f64) as u64;
    let z = (frac.z.clamp(0.0, Z_ORDER_CLAMP_MAX) * (1u64 << Z_ORDER_BITS) as f64) as u64;

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

pub fn brute_force_search_full(
    cell: &Cell,
    positions: &[Vector3<f64>],
    cutoff: f64,
) -> EdgeResult {
    let n = positions.len();
    let cutoff_sq = cutoff * cutoff;
    // Estimate capacity: avg 50 neighbors? for small system maybe less.
    let capacity = n * BRUTE_FORCE_CAPACITY_FACTOR;
    let mut edge_i = Vec::with_capacity(capacity);
    let mut edge_j = Vec::with_capacity(capacity);
    let mut shifts = Vec::with_capacity(capacity * 3);

    for i in 0..n {
        for j in (i + 1)..n {
            let (shift, disp) = cell.get_shift_and_displacement(&positions[i], &positions[j]);
            if disp.norm_squared() < cutoff_sq {
                edge_i.push(i as i64);
                edge_j.push(j as i64);
                shifts.push(shift.x);
                shifts.push(shift.y);
                shifts.push(shift.z);
            }
        }
    }
    (edge_i, edge_j, shifts)
}

pub fn brute_force_search_multi(
    cell: &Cell,
    positions: &[Vector3<f64>],
    cutoffs: &[f64],
    disjoint: bool,
) -> Vec<EdgeResult> {
    let n = positions.len();
    let n_cutoffs = cutoffs.len();

    // Sort cutoffs to handle disjoint correctly
    let mut sorted_indices: Vec<usize> = (0..n_cutoffs).collect();
    sorted_indices.sort_by(|&a, &b| cutoffs[a].partial_cmp(&cutoffs[b]).unwrap());

    let sorted_cutoffs_sq: Vec<f64> = sorted_indices
        .iter()
        .map(|&i| cutoffs[i] * cutoffs[i])
        .collect();
    let max_cutoff_sq = sorted_cutoffs_sq.iter().cloned().next_back().unwrap_or(0.0);

    let capacity = n * BRUTE_FORCE_CAPACITY_FACTOR;

    let mut edge_i_vecs: Vec<Vec<i64>> = vec![Vec::with_capacity(capacity); n_cutoffs];
    let mut edge_j_vecs: Vec<Vec<i64>> = vec![Vec::with_capacity(capacity); n_cutoffs];
    let mut shifts_vecs: Vec<Vec<i32>> = vec![Vec::with_capacity(capacity * 3); n_cutoffs];

    for i in 0..n {
        for j in (i + 1)..n {
            let (shift, disp) = cell.get_shift_and_displacement(&positions[i], &positions[j]);
            let d2 = disp.norm_squared();

            if d2 < max_cutoff_sq {
                for (k, &rc_sq) in sorted_cutoffs_sq.iter().enumerate() {
                    if d2 < rc_sq {
                        edge_i_vecs[k].push(i as i64);
                        edge_j_vecs[k].push(j as i64);
                        shifts_vecs[k].push(shift.x);
                        shifts_vecs[k].push(shift.y);
                        shifts_vecs[k].push(shift.z);
                        if disjoint {
                            break;
                        }
                    }
                }
            }
        }
    }

    let mut final_results = vec![(Vec::new(), Vec::new(), Vec::new()); n_cutoffs];
    for (rank, &orig_idx) in sorted_indices.iter().enumerate() {
        final_results[orig_idx] = (
            std::mem::take(&mut edge_i_vecs[rank]),
            std::mem::take(&mut edge_j_vecs[rank]),
            std::mem::take(&mut shifts_vecs[rank]),
        );
    }
    final_results
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
            if orig_idx == 0 {
                loc0 = Some(i);
            }
            if orig_idx == 2 {
                loc2 = Some(i);
            }
        }

        let loc0 = loc0.unwrap();
        let loc2 = loc2.unwrap();

        // They should be adjacent in memory (in the same bin)
        assert!((loc0 as isize - loc2 as isize).abs() == 1);

        // Neighbor search should still find (0, 2)
        let neighbors = cl.search(&cell, &positions, cutoff);
        assert!(
            neighbors
                .iter()
                .any(|&(i, j, _, _, _)| (i == 0 && j == 2) || (i == 2 && j == 0))
        );
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

    #[test]
    fn test_par_search_optimized_consistency() {
        let h = Matrix3::identity() * 10.0;
        let cell = Cell::new(h).unwrap();

        // Create enough atoms to trigger parallel threshold if we were using the python binding,
        // but here we call it directly so it doesn't matter, but good for stress test.
        let mut positions = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    positions.push(Vector3::new(i as f64, j as f64, k as f64));
                }
            }
        }

        let cutoff = 1.5;
        let cl = CellList::build(&cell, &positions, cutoff);

        // Serial reference
        let mut serial_results: Vec<(usize, usize)> = cl
            .search(&cell, &positions, cutoff)
            .into_iter()
            .map(|(i, j, _, _, _)| (i, j))
            .collect();
        serial_results.sort();

        // Parallel
        let (ei, ej, _) = cl.par_search_optimized(&cell, cutoff);
        let mut par_results: Vec<(usize, usize)> = ei
            .iter()
            .zip(ej.iter())
            .map(|(&i, &j)| (i as usize, j as usize))
            .collect();
        par_results.sort();

        assert_eq!(serial_results, par_results);
    }

    #[test]
    fn test_par_search_multi_consistency() {
        let h = Matrix3::identity() * 10.0;
        let cell = Cell::new(h).unwrap();
        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.2, 1.0, 1.0), // dist 0.2
            Vector3::new(1.5, 1.0, 1.0), // dist 0.5 from 0, 0.3 from 1
        ];

        let cutoffs = vec![0.3, 0.6];
        let cl = CellList::build(&cell, &positions, 0.6);

        let results = cl.par_search_multi(&cell, &cutoffs, false);

        // Verify it matches individual searches.
        for (k, cutoff) in cutoffs.iter().enumerate() {
            let (ei, ej, _) = &results[k];
            let mut par_res: Vec<(usize, usize)> = ei
                .iter()
                .zip(ej.iter())
                .map(|(&i, &j)| (i as usize, j as usize))
                .collect();
            par_res.sort();

            let mut serial_res: Vec<(usize, usize)> = cl
                .search(&cell, &positions, *cutoff)
                .into_iter()
                .map(|(i, j, _, _, _)| (i, j))
                .collect();
            serial_res.sort();

            assert_eq!(par_res, serial_res, "Mismatch at cutoff {}", cutoff);
        }
    }

    #[test]
    fn test_brute_force_full_consistency() {
        let h = Matrix3::identity() * 10.0;
        let cell = Cell::new(h).unwrap();
        let positions = vec![Vector3::new(1.0, 1.0, 1.0), Vector3::new(1.2, 1.0, 1.0)];
        let cutoff = 0.5;

        let (ei, ej, _) = brute_force_search_full(&cell, &positions, cutoff);
        let mut bf_res: Vec<(usize, usize)> = ei
            .iter()
            .zip(ej.iter())
            .map(|(&i, &j)| (i as usize, j as usize))
            .collect();
        bf_res.sort();

        let cl = CellList::build(&cell, &positions, cutoff);
        let mut cl_res: Vec<(usize, usize)> = cl
            .search(&cell, &positions, cutoff)
            .into_iter()
            .map(|(i, j, _, _, _)| (i, j))
            .collect();
        cl_res.sort();

        assert_eq!(bf_res, cl_res);
    }

    #[test]
    fn test_brute_force_multi_consistency() {
        let h = Matrix3::identity() * 10.0;
        let cell = Cell::new(h).unwrap();
        let positions = vec![
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.2, 1.0, 1.0),
            Vector3::new(1.5, 1.0, 1.0),
        ];

        let cutoffs = vec![0.3, 0.6];
        // Brute force multi
        let results = brute_force_search_multi(&cell, &positions, &cutoffs, false);

        for (k, cutoff) in cutoffs.iter().enumerate() {
            let (ei, ej, _) = &results[k];
            let mut bf_res: Vec<(usize, usize)> = ei
                .iter()
                .zip(ej.iter())
                .map(|(&i, &j)| (i as usize, j as usize))
                .collect();
            bf_res.sort();

            let cl = CellList::build(&cell, &positions, *cutoff);
            let mut serial_res: Vec<(usize, usize)> = cl
                .search(&cell, &positions, *cutoff)
                .into_iter()
                .map(|(i, j, _, _, _)| (i, j))
                .collect();
            serial_res.sort();

            assert_eq!(bf_res, serial_res, "Mismatch at cutoff {}", cutoff);
        }
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