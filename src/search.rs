use crate::cell::Cell;
use crate::config;
use nalgebra::Vector3;
use rayon::prelude::*;
use tracing::info_span;
use wide::*;

// Internal tuning parameters
const BRUTE_FORCE_CAPACITY_FACTOR: usize = 10;
const Z_ORDER_BITS: u32 = 21;
const Z_ORDER_CLAMP_MAX: f64 = 0.999999;
const PARALLEL_TASKS_PER_THREAD: usize = 64;

pub type EdgeResult = (Vec<i64>, Vec<i64>, Vec<i32>);

/// Helper to avoid small Vec allocations in hot loops.
/// Stores offset data: (bin_index, shift_vector, shift_integer).
struct OffsetList {
    stack: [(usize, Vector3<f64>, i32); 16],
    heap: Vec<(usize, Vector3<f64>, i32)>,
    use_stack: bool,
    len: usize,
}

impl OffsetList {
    fn new() -> Self {
        Self {
            stack: [(0, Vector3::zeros(), 0); 16],
            heap: Vec::new(),
            use_stack: true,
            len: 0,
        }
    }

    fn prepare(&mut self, center_bin: i32, n_search: i32, num_bins: i32, h_col: Vector3<f64>) {
        let required = (2 * n_search + 1) as usize;
        if required > 16 {
            self.use_stack = false;
            self.heap.clear();
            self.heap.reserve(required);
        } else {
            self.use_stack = true;
            self.len = 0;
        }

        for d in -n_search..=n_search {
            let (nb, shift) = div_mod(center_bin + d, num_bins);
            let vec = h_col * (shift as f64);
            let item = (nb, vec, shift);
            if self.use_stack {
                self.stack[self.len] = item;
                self.len += 1;
            } else {
                self.heap.push(item);
            }
        }
    }

    fn as_slice(&self) -> &[(usize, Vector3<f64>, i32)] {
        if self.use_stack {
            &self.stack[..self.len]
        } else {
            &self.heap
        }
    }
}

pub struct CellList {
    /// particles[sorted_idx] = original_idx
    particles: Vec<usize>,
    /// cell_starts[bin_rank] = start index in particles
    cell_starts: Vec<usize>,
    /// Maps linear bin index (bx + nx*(by + ny*bz)) to Morton rank
    bin_ranks: Vec<usize>,
    /// Wrapped positions in sorted order (index matches sorted_idx).
    pos_x: Vec<f64>,
    pos_y: Vec<f64>,
    pos_z: Vec<f64>,
    /// Wrapping shifts in sorted order (index matches sorted_idx).
    atom_shifts_x: Vec<i32>,
    atom_shifts_y: Vec<i32>,
    atom_shifts_z: Vec<i32>,
    num_bins: Vector3<usize>,
    n_search: Vector3<i32>,
    // Optimization for pruning
    bin_basis: [Vector3<f64>; 3],
    bin_origin: Vector3<f64>,
    bin_bounding_radius: f64,
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

        let mut final_pos_x = vec![0.0; n_atoms];
        let mut final_pos_y = vec![0.0; n_atoms];
        let mut final_pos_z = vec![0.0; n_atoms];
        let mut final_atom_shifts_x = vec![0; n_atoms];
        let mut final_atom_shifts_y = vec![0; n_atoms];
        let mut final_atom_shifts_z = vec![0; n_atoms];
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
                final_pos_x[loc] = wrapped_pos.x;
                final_pos_y[loc] = wrapped_pos.y;
                final_pos_z[loc] = wrapped_pos.z;
                final_atom_shifts_x[loc] = sx;
                final_atom_shifts_y[loc] = sy;
                final_atom_shifts_z[loc] = sz;
                final_particles[loc] = original_idx;
                current_fill[rank] += 1;
            }
        }

        // 6. Precompute bin geometry for pruning
        let vx = h_matrix * Vector3::new(1.0 / num_bins.x as f64, 0.0, 0.0);
        let vy = h_matrix * Vector3::new(0.0, 1.0 / num_bins.y as f64, 0.0);
        let vz = h_matrix * Vector3::new(0.0, 0.0, 1.0 / num_bins.z as f64);
        let bin_origin = h_matrix
            * Vector3::new(
                0.5 / num_bins.x as f64,
                0.5 / num_bins.y as f64,
                0.5 / num_bins.z as f64,
            );

        // Compute max radius (half of max diagonal)
        let d1 = (vx + vy + vz).norm_squared();
        let d2 = (vx + vy - vz).norm_squared();
        let d3 = (vx - vy + vz).norm_squared();
        let d4 = (vx - vy - vz).norm_squared();
        let max_diag_sq = d1.max(d2).max(d3).max(d4);
        let bin_bounding_radius = 0.5 * max_diag_sq.sqrt();

        Self {
            particles: final_particles,
            cell_starts,
            bin_ranks,
            pos_x: final_pos_x,
            pos_y: final_pos_y,
            pos_z: final_pos_z,
            atom_shifts_x: final_atom_shifts_x,
            atom_shifts_y: final_atom_shifts_y,
            atom_shifts_z: final_atom_shifts_z,
            num_bins,
            n_search,
            bin_basis: [vx, vy, vz],
            bin_origin,
            bin_bounding_radius,
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

        let counts: Vec<u32> = {
            let _s = info_span!("count_pass").entered();
            (0..n_atoms)
                .into_par_iter()
                .with_min_len(min_len)
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
            (0..n_atoms)
                .into_par_iter()
                .with_min_len(min_len)
                .for_each(|i| {
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

    pub fn par_search_multi(
        &self,
        cell: &Cell,
        cutoffs: &[f64],
        disjoint: bool,
    ) -> Vec<EdgeResult> {
        let _span = info_span!("CellList::par_search_multi").entered();
        let n_atoms = self.particles.len();
        let n_cutoffs = cutoffs.len();

        let mut sorted_indices: Vec<usize> = (0..n_cutoffs).collect();
        sorted_indices.sort_by(|&a, &b| cutoffs[a].partial_cmp(&cutoffs[b]).unwrap());

        let sorted_cutoffs_sq: Vec<f64> = sorted_indices
            .iter()
            .map(|&i| cutoffs[i] * cutoffs[i])
            .collect();
        let max_cutoff_sq = sorted_cutoffs_sq.iter().cloned().next_back().unwrap_or(0.0);

        let num_threads = rayon::current_num_threads();
        let min_len = (n_atoms / (num_threads * PARALLEL_TASKS_PER_THREAD)).max(1);

        // 1. Count pass
        let mut counts = vec![0u32; n_atoms * n_cutoffs];
        {
            let _s = info_span!("count_pass_multi").entered();
            counts
                .par_chunks_mut(n_cutoffs)
                .enumerate()
                .for_each(|(i, atom_counts)| {
                    self.count_atom_neighbors_multi(
                        i,
                        cell,
                        &sorted_cutoffs_sq,
                        max_cutoff_sq,
                        atom_counts,
                        disjoint,
                    );
                });
        }

        // 2. Offsets
        let mut offsets = vec![0usize; n_atoms * n_cutoffs];
        let mut totals = vec![0usize; n_cutoffs];

        for k in 0..n_cutoffs {
            let mut accum = 0;
            for i in 0..n_atoms {
                offsets[i * n_cutoffs + k] = accum;
                accum += counts[i * n_cutoffs + k] as usize;
            }
            totals[k] = accum;
        }

        // 3. Allocate results
        let mut result_edge_i: Vec<Vec<i64>> = totals.iter().map(|&t| vec![0; t]).collect();
        let mut result_edge_j: Vec<Vec<i64>> = totals.iter().map(|&t| vec![0; t]).collect();
        let mut result_shifts: Vec<Vec<i32>> = totals.iter().map(|&t| vec![0; t * 3]).collect();

        let ptrs: Vec<(usize, usize, usize)> = (0..n_cutoffs)
            .map(|k| {
                (
                    result_edge_i[k].as_mut_ptr() as usize,
                    result_edge_j[k].as_mut_ptr() as usize,
                    result_shifts[k].as_mut_ptr() as usize,
                )
            })
            .collect();

        // 4. Fill pass
        {
            let _s = info_span!("fill_pass_multi").entered();
            (0..n_atoms)
                .into_par_iter()
                .with_min_len(min_len)
                .for_each(|i| {
                    let atom_offsets = &offsets[i * n_cutoffs..(i + 1) * n_cutoffs];

                    unsafe {
                        self.fill_atom_neighbors_multi(
                            i,
                            cell,
                            &sorted_cutoffs_sq,
                            max_cutoff_sq,
                            atom_offsets,
                            &ptrs,
                            disjoint,
                        );
                    }
                });
        }

        // Assemble return - Reorder back to original input cutoff order
        let mut final_results = vec![(Vec::new(), Vec::new(), Vec::new()); n_cutoffs];
        for (rank, &orig_idx) in sorted_indices.iter().enumerate() {
            final_results[orig_idx] = (
                std::mem::take(&mut result_edge_i[rank]),
                std::mem::take(&mut result_edge_j[rank]),
                std::mem::take(&mut result_shifts[rank]),
            );
        }
        final_results
    }

    fn count_atom_neighbors(&self, i: usize, cell: &Cell, cutoff_sq: f64) -> usize {
        let pos_i_x = self.pos_x[i];
        let pos_i_y = self.pos_y[i];
        let pos_i_z = self.pos_z[i];
        let h_matrix = cell.h();

        let frac_i = cell.to_fractional(&Vector3::new(pos_i_x, pos_i_y, pos_i_z));
        let bx = ((frac_i.x * self.num_bins.x as f64) as i32).min(self.num_bins.x as i32 - 1);
        let by = ((frac_i.y * self.num_bins.y as f64) as i32).min(self.num_bins.y as i32 - 1);
        let bz = ((frac_i.z * self.num_bins.z as f64) as i32).min(self.num_bins.z as i32 - 1);

        let i_orig = self.particles[i];
        let mut count = 0;

        // Pre-calculate shift vectors using OffsetList
        let mut sx_list = OffsetList::new();
        sx_list.prepare(
            bx,
            self.n_search.x,
            self.num_bins.x as i32,
            h_matrix.column(0).into(),
        );

        let mut sy_list = OffsetList::new();
        sy_list.prepare(
            by,
            self.n_search.y,
            self.num_bins.y as i32,
            h_matrix.column(1).into(),
        );

        let mut sz_list = OffsetList::new();
        sz_list.prepare(
            bz,
            self.n_search.z,
            self.num_bins.z as i32,
            h_matrix.column(2).into(),
        );

        let cutoff_sq_v = f64x4::from(cutoff_sq);
        let prune_limit = cutoff_sq.sqrt() + self.bin_bounding_radius;
        let prune_limit_sq = prune_limit * prune_limit;
        let pos_i = Vector3::new(pos_i_x, pos_i_y, pos_i_z);

        for (nbx, ox, sx) in sx_list.as_slice() {
            let center_x = self.bin_origin + self.bin_basis[0] * (*nbx as f64);
            for (nby, oy, sy) in sy_list.as_slice() {
                let center_xy = center_x + self.bin_basis[1] * (*nby as f64);
                for (nbz, oz, sz) in sz_list.as_slice() {
                    let target_center = center_xy + self.bin_basis[2] * (*nbz as f64);
                    let offset_vec = ox + oy + oz;

                    // Pruning check
                    let center_shifted = target_center + offset_vec;
                    let diff = center_shifted - pos_i;
                    if diff.norm_squared() > prune_limit_sq {
                        continue;
                    }

                    let linear_idx = nbx + self.num_bins.x * (nby + self.num_bins.y * nbz);
                    let rank = self.bin_ranks[linear_idx];
                    let start_j = self.cell_starts[rank];
                    let end_j = self.cell_starts[rank + 1];

                    if start_j == end_j {
                        continue;
                    }

                    let offset_vec = ox + oy + oz;
                    let is_shifted = *sx != 0 || *sy != 0 || *sz != 0;
                    let ox_v = f64x4::from(offset_vec.x);
                    let oy_v = f64x4::from(offset_vec.y);
                    let oz_v = f64x4::from(offset_vec.z);
                    let pix_v = f64x4::from(pos_i_x);
                    let piy_v = f64x4::from(pos_i_y);
                    let piz_v = f64x4::from(pos_i_z);

                    let mut sj = start_j;
                    while sj + 4 <= end_j {
                        let j_orig_v = [
                            self.particles[sj],
                            self.particles[sj + 1],
                            self.particles[sj + 2],
                            self.particles[sj + 3],
                        ];
                        let pjx_v =
                            f64x4::from(<[f64; 4]>::try_from(&self.pos_x[sj..sj + 4]).unwrap());
                        let pjy_v =
                            f64x4::from(<[f64; 4]>::try_from(&self.pos_y[sj..sj + 4]).unwrap());
                        let pjz_v =
                            f64x4::from(<[f64; 4]>::try_from(&self.pos_z[sj..sj + 4]).unwrap());

                        let dx = (pjx_v - pix_v) + ox_v;
                        let dy = (pjy_v - piy_v) + oy_v;
                        let dz = (pjz_v - piz_v) + oz_v;
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        let mask = dist_sq.simd_lt(cutoff_sq_v);
                        let m_array: [u64; 4] = bytemuck::cast(mask);
                        for k in 0..4 {
                            if m_array[k] != 0
                                && (i_orig < j_orig_v[k] || (i_orig == j_orig_v[k] && is_shifted))
                            {
                                count += 1;
                            }
                        }
                        sj += 4;
                    }

                    for sj_tail in sj..end_j {
                        let j_orig = self.particles[sj_tail];
                        if i_orig > j_orig {
                            continue;
                        }
                        if i_orig == j_orig && !is_shifted {
                            continue;
                        }
                        let dx = (self.pos_x[sj_tail] - pos_i_x) + offset_vec.x;
                        let dy = (self.pos_y[sj_tail] - pos_i_y) + offset_vec.y;
                        let dz = (self.pos_z[sj_tail] - pos_i_z) + offset_vec.z;
                        if dx * dx + dy * dy + dz * dz < cutoff_sq {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    #[allow(clippy::too_many_arguments)]
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
        let pos_i_x = self.pos_x[i];
        let pos_i_y = self.pos_y[i];
        let pos_i_z = self.pos_z[i];
        let s_i_x = self.atom_shifts_x[i];
        let s_i_y = self.atom_shifts_y[i];
        let s_i_z = self.atom_shifts_z[i];
        let h_matrix = cell.h();

        let frac_i = cell.to_fractional(&Vector3::new(pos_i_x, pos_i_y, pos_i_z));
        let bx = ((frac_i.x * self.num_bins.x as f64) as i32).min(self.num_bins.x as i32 - 1);
        let by = ((frac_i.y * self.num_bins.y as f64) as i32).min(self.num_bins.y as i32 - 1);
        let bz = ((frac_i.z * self.num_bins.z as f64) as i32).min(self.num_bins.z as i32 - 1);

        let i_orig = self.particles[i];

        // Pre-calculate shift vectors using OffsetList
        let mut sx_list = OffsetList::new();
        sx_list.prepare(
            bx,
            self.n_search.x,
            self.num_bins.x as i32,
            h_matrix.column(0).into(),
        );

        let mut sy_list = OffsetList::new();
        sy_list.prepare(
            by,
            self.n_search.y,
            self.num_bins.y as i32,
            h_matrix.column(1).into(),
        );

        let mut sz_list = OffsetList::new();
        sz_list.prepare(
            bz,
            self.n_search.z,
            self.num_bins.z as i32,
            h_matrix.column(2).into(),
        );

        let cutoff_sq_v = f64x4::from(cutoff_sq);
        let prune_limit = cutoff_sq.sqrt() + self.bin_bounding_radius;
        let prune_limit_sq = prune_limit * prune_limit;
        let pos_i = Vector3::new(pos_i_x, pos_i_y, pos_i_z);

        for (nbx, ox, sx) in sx_list.as_slice() {
            let center_x = self.bin_origin + self.bin_basis[0] * (*nbx as f64);
            for (nby, oy, sy) in sy_list.as_slice() {
                let center_xy = center_x + self.bin_basis[1] * (*nby as f64);
                for (nbz, oz, sz) in sz_list.as_slice() {
                    let target_center = center_xy + self.bin_basis[2] * (*nbz as f64);
                    let offset_vec = ox + oy + oz;

                    // Pruning check
                    let center_shifted = target_center + offset_vec;
                    let diff = center_shifted - pos_i;
                    if diff.norm_squared() > prune_limit_sq {
                        continue;
                    }

                    let linear_idx = nbx + self.num_bins.x * (nby + self.num_bins.y * nbz);
                    let rank = self.bin_ranks[linear_idx];
                    let start_j = self.cell_starts[rank];
                    let end_j = self.cell_starts[rank + 1];

                    if start_j == end_j {
                        continue;
                    }

                    let offset_vec = ox + oy + oz;
                    let is_shifted = *sx != 0 || *sy != 0 || *sz != 0;
                    let ox_v = f64x4::from(offset_vec.x);
                    let oy_v = f64x4::from(offset_vec.y);
                    let oz_v = f64x4::from(offset_vec.z);
                    let pix_v = f64x4::from(pos_i_x);
                    let piy_v = f64x4::from(pos_i_y);
                    let piz_v = f64x4::from(pos_i_z);

                    let mut sj = start_j;
                    while sj + 4 <= end_j {
                        let j_orig_v = [
                            self.particles[sj],
                            self.particles[sj + 1],
                            self.particles[sj + 2],
                            self.particles[sj + 3],
                        ];
                        let pjx_v =
                            f64x4::from(<[f64; 4]>::try_from(&self.pos_x[sj..sj + 4]).unwrap());
                        let pjy_v =
                            f64x4::from(<[f64; 4]>::try_from(&self.pos_y[sj..sj + 4]).unwrap());
                        let pjz_v =
                            f64x4::from(<[f64; 4]>::try_from(&self.pos_z[sj..sj + 4]).unwrap());

                        let dx = (pjx_v - pix_v) + ox_v;
                        let dy = (pjy_v - piy_v) + oy_v;
                        let dz = (pjz_v - piz_v) + oz_v;
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        let mask = dist_sq.simd_lt(cutoff_sq_v);
                        let m_array: [u64; 4] = bytemuck::cast(mask);
                        for k in 0..4 {
                            if m_array[k] != 0
                                && (i_orig < j_orig_v[k] || (i_orig == j_orig_v[k] && is_shifted))
                            {
                                edge_i[offset] = i_orig as i64;
                                edge_j[offset] = j_orig_v[k] as i64;
                                shifts[offset * 3] = self.atom_shifts_x[sj + k] - s_i_x + sx;
                                shifts[offset * 3 + 1] = self.atom_shifts_y[sj + k] - s_i_y + sy;
                                shifts[offset * 3 + 2] = self.atom_shifts_z[sj + k] - s_i_z + sz;
                                offset += 1;
                            }
                        }
                        sj += 4;
                    }

                    for sj_tail in sj..end_j {
                        let j_orig = self.particles[sj_tail];
                        if i_orig > j_orig {
                            continue;
                        }
                        if i_orig == j_orig && !is_shifted {
                            continue;
                        }
                        let dx = (self.pos_x[sj_tail] - pos_i_x) + offset_vec.x;
                        let dy = (self.pos_y[sj_tail] - pos_i_y) + offset_vec.y;
                        let dz = (self.pos_z[sj_tail] - pos_i_z) + offset_vec.z;
                        if dx * dx + dy * dy + dz * dz < cutoff_sq {
                            edge_i[offset] = i_orig as i64;
                            edge_j[offset] = j_orig as i64;
                            shifts[offset * 3] = self.atom_shifts_x[sj_tail] - s_i_x + sx;
                            shifts[offset * 3 + 1] = self.atom_shifts_y[sj_tail] - s_i_y + sy;
                            shifts[offset * 3 + 2] = self.atom_shifts_z[sj_tail] - s_i_z + sz;
                            offset += 1;
                        }
                    }
                }
            }
        }
    }

    fn count_atom_neighbors_multi(
        &self,
        i: usize,
        cell: &Cell,
        sorted_cutoffs_sq: &[f64],
        max_cutoff_sq: f64,
        counts: &mut [u32],
        disjoint: bool,
    ) {
        let pos_i_x = self.pos_x[i];
        let pos_i_y = self.pos_y[i];
        let pos_i_z = self.pos_z[i];
        let h_matrix = cell.h();

        let frac_i = cell.to_fractional(&Vector3::new(pos_i_x, pos_i_y, pos_i_z));
        let bx = ((frac_i.x * self.num_bins.x as f64) as i32).min(self.num_bins.x as i32 - 1);
        let by = ((frac_i.y * self.num_bins.y as f64) as i32).min(self.num_bins.y as i32 - 1);
        let bz = ((frac_i.z * self.num_bins.z as f64) as i32).min(self.num_bins.z as i32 - 1);

        let i_orig = self.particles[i];

        // Pre-calculate shift vectors using OffsetList
        let mut sx_list = OffsetList::new();
        sx_list.prepare(
            bx,
            self.n_search.x,
            self.num_bins.x as i32,
            h_matrix.column(0).into(),
        );

        let mut sy_list = OffsetList::new();
        sy_list.prepare(
            by,
            self.n_search.y,
            self.num_bins.y as i32,
            h_matrix.column(1).into(),
        );

        let mut sz_list = OffsetList::new();
        sz_list.prepare(
            bz,
            self.n_search.z,
            self.num_bins.z as i32,
            h_matrix.column(2).into(),
        );

        let max_cutoff_sq_v = f64x4::from(max_cutoff_sq);
        let prune_limit = max_cutoff_sq.sqrt() + self.bin_bounding_radius;
        let prune_limit_sq = prune_limit * prune_limit;
        let pos_i = Vector3::new(pos_i_x, pos_i_y, pos_i_z);

        for (nbx, ox, sx) in sx_list.as_slice() {
            let center_x = self.bin_origin + self.bin_basis[0] * (*nbx as f64);
            for (nby, oy, sy) in sy_list.as_slice() {
                let center_xy = center_x + self.bin_basis[1] * (*nby as f64);
                for (nbz, oz, sz) in sz_list.as_slice() {
                    let target_center = center_xy + self.bin_basis[2] * (*nbz as f64);
                    let offset_vec = ox + oy + oz;

                    // Pruning check
                    let center_shifted = target_center + offset_vec;
                    let diff = center_shifted - pos_i;
                    if diff.norm_squared() > prune_limit_sq {
                        continue;
                    }

                    let linear_idx = nbx + self.num_bins.x * (nby + self.num_bins.y * nbz);
                    let rank = self.bin_ranks[linear_idx];
                    let start_j = self.cell_starts[rank];
                    let end_j = self.cell_starts[rank + 1];

                    if start_j == end_j {
                        continue;
                    }

                    let offset_vec = ox + oy + oz;
                    let is_shifted = *sx != 0 || *sy != 0 || *sz != 0;
                    let ox_v = f64x4::from(offset_vec.x);
                    let oy_v = f64x4::from(offset_vec.y);
                    let oz_v = f64x4::from(offset_vec.z);
                    let pix_v = f64x4::from(pos_i_x);
                    let piy_v = f64x4::from(pos_i_y);
                    let piz_v = f64x4::from(pos_i_z);

                    let mut sj = start_j;
                    while sj + 4 <= end_j {
                        let j_orig_v = [
                            self.particles[sj],
                            self.particles[sj + 1],
                            self.particles[sj + 2],
                            self.particles[sj + 3],
                        ];
                        let pjx_v =
                            f64x4::from(<[f64; 4]>::try_from(&self.pos_x[sj..sj + 4]).unwrap());
                        let pjy_v =
                            f64x4::from(<[f64; 4]>::try_from(&self.pos_y[sj..sj + 4]).unwrap());
                        let pjz_v =
                            f64x4::from(<[f64; 4]>::try_from(&self.pos_z[sj..sj + 4]).unwrap());

                        let dx = (pjx_v - pix_v) + ox_v;
                        let dy = (pjy_v - piy_v) + oy_v;
                        let dz = (pjz_v - piz_v) + oz_v;
                        let dist_sq_v = dx * dx + dy * dy + dz * dz;

                        let mask = dist_sq_v.simd_lt(max_cutoff_sq_v);
                        let m_array: [u64; 4] = bytemuck::cast(mask);
                        let dist_sq_array: [f64; 4] = dist_sq_v.into();

                        for k in 0..4 {
                            if m_array[k] != 0
                                && (i_orig < j_orig_v[k] || (i_orig == j_orig_v[k] && is_shifted))
                            {
                                let d2 = dist_sq_array[k];
                                for (cutoff_idx, &rc_sq) in sorted_cutoffs_sq.iter().enumerate() {
                                    if d2 < rc_sq {
                                        counts[cutoff_idx] += 1;
                                        if disjoint {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        sj += 4;
                    }

                    for sj_tail in sj..end_j {
                        let j_orig = self.particles[sj_tail];
                        if i_orig > j_orig {
                            continue;
                        }
                        if i_orig == j_orig && !is_shifted {
                            continue;
                        }
                        let dx = (self.pos_x[sj_tail] - pos_i_x) + offset_vec.x;
                        let dy = (self.pos_y[sj_tail] - pos_i_y) + offset_vec.y;
                        let dz = (self.pos_z[sj_tail] - pos_i_z) + offset_vec.z;
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        if dist_sq < max_cutoff_sq {
                            for (k, &rc_sq) in sorted_cutoffs_sq.iter().enumerate() {
                                if dist_sq < rc_sq {
                                    counts[k] += 1;
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

    // SAFETY: The caller must ensure that atom_offsets and ptrs are valid and imply exclusive mutable access
    // to the regions of the output vectors assigned to this atom.
    #[allow(clippy::too_many_arguments)]
    unsafe fn fill_atom_neighbors_multi(
        &self,
        i: usize,
        cell: &Cell,
        sorted_cutoffs_sq: &[f64],
        max_cutoff_sq: f64,
        atom_offsets: &[usize],         // offsets for this atom for each rank
        ptrs: &[(usize, usize, usize)], // global pointers for each rank
        disjoint: bool,
    ) {
        let pos_i_x = self.pos_x[i];
        let pos_i_y = self.pos_y[i];
        let pos_i_z = self.pos_z[i];
        let s_i_x = self.atom_shifts_x[i];
        let s_i_y = self.atom_shifts_y[i];
        let s_i_z = self.atom_shifts_z[i];
        let h_matrix = cell.h();

        let frac_i = cell.to_fractional(&Vector3::new(pos_i_x, pos_i_y, pos_i_z));
        let bx = ((frac_i.x * self.num_bins.x as f64) as i32).min(self.num_bins.x as i32 - 1);
        let by = ((frac_i.y * self.num_bins.y as f64) as i32).min(self.num_bins.y as i32 - 1);
        let bz = ((frac_i.z * self.num_bins.z as f64) as i32).min(self.num_bins.z as i32 - 1);

        let i_orig = self.particles[i];
        let mut local_counts = vec![0usize; sorted_cutoffs_sq.len()];

        // Pre-calculate shift vectors using OffsetList
        let mut sx_list = OffsetList::new();
        sx_list.prepare(
            bx,
            self.n_search.x,
            self.num_bins.x as i32,
            h_matrix.column(0).into(),
        );

        let mut sy_list = OffsetList::new();
        sy_list.prepare(
            by,
            self.n_search.y,
            self.num_bins.y as i32,
            h_matrix.column(1).into(),
        );

        let mut sz_list = OffsetList::new();
        sz_list.prepare(
            bz,
            self.n_search.z,
            self.num_bins.z as i32,
            h_matrix.column(2).into(),
        );

        let max_cutoff_sq_v = f64x4::from(max_cutoff_sq);
        let prune_limit = max_cutoff_sq.sqrt() + self.bin_bounding_radius;
        let prune_limit_sq = prune_limit * prune_limit;
        let pos_i = Vector3::new(pos_i_x, pos_i_y, pos_i_z);

        for (nbx, ox, sx) in sx_list.as_slice() {
            let center_x = self.bin_origin + self.bin_basis[0] * (*nbx as f64);
            for (nby, oy, sy) in sy_list.as_slice() {
                let center_xy = center_x + self.bin_basis[1] * (*nby as f64);
                for (nbz, oz, sz) in sz_list.as_slice() {
                    let target_center = center_xy + self.bin_basis[2] * (*nbz as f64);
                    let offset_vec = ox + oy + oz;

                    // Pruning check
                    let center_shifted = target_center + offset_vec;
                    let diff = center_shifted - pos_i;
                    if diff.norm_squared() > prune_limit_sq {
                        continue;
                    }

                    let linear_idx = nbx + self.num_bins.x * (nby + self.num_bins.y * nbz);
                    let rank = self.bin_ranks[linear_idx];
                    let start_j = self.cell_starts[rank];
                    let end_j = self.cell_starts[rank + 1];

                    if start_j == end_j {
                        continue;
                    }

                    let offset_vec = ox + oy + oz;
                    let is_shifted = *sx != 0 || *sy != 0 || *sz != 0;
                    let ox_v = f64x4::from(offset_vec.x);
                    let oy_v = f64x4::from(offset_vec.y);
                    let oz_v = f64x4::from(offset_vec.z);
                    let pix_v = f64x4::from(pos_i_x);
                    let piy_v = f64x4::from(pos_i_y);
                    let piz_v = f64x4::from(pos_i_z);

                    let mut sj = start_j;
                    while sj + 4 <= end_j {
                        let j_orig_v = [
                            self.particles[sj],
                            self.particles[sj + 1],
                            self.particles[sj + 2],
                            self.particles[sj + 3],
                        ];
                        let pjx_v =
                            f64x4::from(<[f64; 4]>::try_from(&self.pos_x[sj..sj + 4]).unwrap());
                        let pjy_v =
                            f64x4::from(<[f64; 4]>::try_from(&self.pos_y[sj..sj + 4]).unwrap());
                        let pjz_v =
                            f64x4::from(<[f64; 4]>::try_from(&self.pos_z[sj..sj + 4]).unwrap());

                        let dx = (pjx_v - pix_v) + ox_v;
                        let dy = (pjy_v - piy_v) + oy_v;
                        let dz = (pjz_v - piz_v) + oz_v;
                        let dist_sq_v = dx * dx + dy * dy + dz * dz;

                        let mask = dist_sq_v.simd_lt(max_cutoff_sq_v);
                        let m_array: [u64; 4] = bytemuck::cast(mask);
                        let dist_sq_array: [f64; 4] = dist_sq_v.into();

                        for k in 0..4 {
                            if m_array[k] != 0
                                && (i_orig < j_orig_v[k] || (i_orig == j_orig_v[k] && is_shifted))
                            {
                                let d2 = dist_sq_array[k];
                                for (cutoff_idx, &rc_sq) in sorted_cutoffs_sq.iter().enumerate() {
                                    if d2 < rc_sq {
                                        let write_idx =
                                            atom_offsets[cutoff_idx] + local_counts[cutoff_idx];
                                        let (ptr_i, ptr_j, ptr_s) = ptrs[cutoff_idx];
                                        let p_i = ptr_i as *mut i64;
                                        let p_j = ptr_j as *mut i64;
                                        let p_s = ptr_s as *mut i32;

                                        unsafe {
                                            *p_i.add(write_idx) = i_orig as i64;
                                            *p_j.add(write_idx) = j_orig_v[k] as i64;
                                            let ps_base = p_s.add(write_idx * 3);
                                            *ps_base = self.atom_shifts_x[sj + k] - s_i_x + sx;
                                            *ps_base.add(1) =
                                                self.atom_shifts_y[sj + k] - s_i_y + sy;
                                            *ps_base.add(2) =
                                                self.atom_shifts_z[sj + k] - s_i_z + sz;
                                        }

                                        local_counts[cutoff_idx] += 1;
                                        if disjoint {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        sj += 4;
                    }

                    for sj_tail in sj..end_j {
                        let j_orig = self.particles[sj_tail];
                        if i_orig > j_orig {
                            continue;
                        }
                        if i_orig == j_orig && !is_shifted {
                            continue;
                        }
                        let dx = (self.pos_x[sj_tail] - pos_i_x) + offset_vec.x;
                        let dy = (self.pos_y[sj_tail] - pos_i_y) + offset_vec.y;
                        let dz = (self.pos_z[sj_tail] - pos_i_z) + offset_vec.z;
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        if dist_sq < max_cutoff_sq {
                            for (k, &rc_sq) in sorted_cutoffs_sq.iter().enumerate() {
                                if dist_sq < rc_sq {
                                    let write_idx = atom_offsets[k] + local_counts[k];
                                    let (ptr_i, ptr_j, ptr_s) = ptrs[k];
                                    let p_i = ptr_i as *mut i64;
                                    let p_j = ptr_j as *mut i64;
                                    let p_s = ptr_s as *mut i32;

                                    unsafe {
                                        *p_i.add(write_idx) = i_orig as i64;
                                        *p_j.add(write_idx) = j_orig as i64;
                                        let ps_base = p_s.add(write_idx * 3);
                                        *ps_base = self.atom_shifts_x[sj_tail] - s_i_x + sx;
                                        *ps_base.add(1) = self.atom_shifts_y[sj_tail] - s_i_y + sy;
                                        *ps_base.add(2) = self.atom_shifts_z[sj_tail] - s_i_z + sz;
                                    }

                                    local_counts[k] += 1;
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

    fn search_atom_neighbors(
        &self,
        i: usize,
        cell: &Cell,
        cutoff_sq: f64,
        neighbors: &mut Vec<(usize, usize, i32, i32, i32)>,
    ) {
        let pos_i_x = self.pos_x[i];
        let pos_i_y = self.pos_y[i];
        let pos_i_z = self.pos_z[i];
        let s_i_x = self.atom_shifts_x[i];
        let s_i_y = self.atom_shifts_y[i];
        let s_i_z = self.atom_shifts_z[i];
        let h_matrix = cell.h();

        let frac_i = cell.to_fractional(&Vector3::new(pos_i_x, pos_i_y, pos_i_z));
        let bx = ((frac_i.x * self.num_bins.x as f64) as i32).min(self.num_bins.x as i32 - 1);
        let by = ((frac_i.y * self.num_bins.y as f64) as i32).min(self.num_bins.y as i32 - 1);
        let bz = ((frac_i.z * self.num_bins.z as f64) as i32).min(self.num_bins.z as i32 - 1);

        let i_orig = self.particles[i];

        let prune_limit = cutoff_sq.sqrt() + self.bin_bounding_radius;
        let prune_limit_sq = prune_limit * prune_limit;
        let pos_i = Vector3::new(pos_i_x, pos_i_y, pos_i_z);

        for dx in -self.n_search.x..=self.n_search.x {
            for dy in -self.n_search.y..=self.n_search.y {
                for dz in -self.n_search.z..=self.n_search.z {
                    let (nbx, sx) = div_mod(bx + dx, self.num_bins.x as i32);
                    let (nby, sy) = div_mod(by + dy, self.num_bins.y as i32);
                    let (nbz, sz) = div_mod(bz + dz, self.num_bins.z as i32);

                    let offset_vec = h_matrix * Vector3::new(sx as f64, sy as f64, sz as f64);

                    // Pruning check
                    let target_center = self.bin_origin
                        + self.bin_basis[0] * (nbx as f64)
                        + self.bin_basis[1] * (nby as f64)
                        + self.bin_basis[2] * (nbz as f64);
                    let center_shifted = target_center + offset_vec;
                    let diff = center_shifted - pos_i;
                    if diff.norm_squared() > prune_limit_sq {
                        continue;
                    }

                    let linear_idx = nbx + self.num_bins.x * (nby + self.num_bins.y * nbz);
                    let rank = self.bin_ranks[linear_idx];
                    let start_j = self.cell_starts[rank];
                    let end_j = self.cell_starts[rank + 1];

                    if start_j == end_j {
                        continue;
                    }

                    let is_shifted = sx != 0 || sy != 0 || sz != 0;

                    for sj in start_j..end_j {
                        let j_orig = self.particles[sj];
                        if i_orig > j_orig {
                            continue;
                        }
                        if i_orig == j_orig && !is_shifted {
                            continue;
                        }
                        let dx = (self.pos_x[sj] - pos_i_x) + offset_vec.x;
                        let dy = (self.pos_y[sj] - pos_i_y) + offset_vec.y;
                        let dz = (self.pos_z[sj] - pos_i_z) + offset_vec.z;

                        if dx * dx + dy * dy + dz * dz < cutoff_sq {
                            let s_total_x = self.atom_shifts_x[sj] - s_i_x + sx;
                            let s_total_y = self.atom_shifts_y[sj] - s_i_y + sy;
                            let s_total_z = self.atom_shifts_z[sj] - s_i_z + sz;
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

pub fn brute_force_search_simd(positions: &[Vector3<f64>], cutoff: f64) -> EdgeResult {
    let n = positions.len();

    let cutoff_sq = cutoff * cutoff;
    let cutoff_sq_v = f64x4::from(cutoff_sq);

    // Estimate capacity: avg 50 neighbors? for small system maybe less.
    let capacity = n * BRUTE_FORCE_CAPACITY_FACTOR;
    let mut edge_i = Vec::with_capacity(capacity);
    let mut edge_j = Vec::with_capacity(capacity);
    let mut shifts = Vec::with_capacity(capacity * 3);

    // Stack-allocated scratchpad for SoA conversion
    let (pos_x, pos_y, pos_z) = if n <= config::get_stack_threshold() {
        let mut x = [0.0; config::MAX_STACK_SIZE];
        let mut y = [0.0; config::MAX_STACK_SIZE];
        let mut z = [0.0; config::MAX_STACK_SIZE];
        for (i, p) in positions.iter().enumerate() {
            x[i] = p.x;
            y[i] = p.y;
            z[i] = p.z;
        }
        (
            PositionSoA::Stack(x),
            PositionSoA::Stack(y),
            PositionSoA::Stack(z),
        )
    } else {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        for p in positions {
            x.push(p.x);
            y.push(p.y);
            z.push(p.z);
        }
        (
            PositionSoA::Heap(x),
            PositionSoA::Heap(y),
            PositionSoA::Heap(z),
        )
    };

    let px = match &pos_x {
        PositionSoA::Stack(s) => &s[..n],
        PositionSoA::Heap(v) => &v[..n],
    };
    let py = match &pos_y {
        PositionSoA::Stack(s) => &s[..n],
        PositionSoA::Heap(v) => &v[..n],
    };
    let pz = match &pos_z {
        PositionSoA::Stack(s) => &s[..n],
        PositionSoA::Heap(v) => &v[..n],
    };

    for i in 0..n {
        let pix = f64x4::from(px[i]);
        let piy = f64x4::from(py[i]);
        let piz = f64x4::from(pz[i]);

        let mut j = i + 1;
        while j + 4 <= n {
            let pjx = f64x4::from(<[f64; 4]>::try_from(&px[j..j + 4]).unwrap());
            let pjy = f64x4::from(<[f64; 4]>::try_from(&py[j..j + 4]).unwrap());
            let pjz = f64x4::from(<[f64; 4]>::try_from(&pz[j..j + 4]).unwrap());

            let dx = pjx - pix;
            let dy = pjy - piy;
            let dz = pjz - piz;

            let d2 = dx * dx + dy * dy + dz * dz;
            let mask = d2.simd_lt(cutoff_sq_v);

            if mask.any() {
                let m_array: [u64; 4] = bytemuck::cast(mask);
                for (k, &m) in m_array.iter().enumerate() {
                    if m != 0 {
                        edge_i.push(i as i64);
                        edge_j.push((j + k) as i64);
                        shifts.push(0);
                        shifts.push(0);
                        shifts.push(0);
                    }
                }
            }
            j += 4;
        }

        // Tail loop
        for k in j..n {
            let dx = px[k] - px[i];
            let dy = py[k] - py[i];
            let dz = pz[k] - pz[i];
            if dx * dx + dy * dy + dz * dz < cutoff_sq {
                edge_i.push(i as i64);
                edge_j.push(k as i64);
                shifts.push(0);
                shifts.push(0);
                shifts.push(0);
            }
        }
    }

    (edge_i, edge_j, shifts)
}

#[allow(clippy::large_enum_variant)]
enum PositionSoA {
    Stack([f64; config::MAX_STACK_SIZE]),
    Heap(Vec<f64>),
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

pub fn brute_force_search_full(cell: &Cell, positions: &[Vector3<f64>], cutoff: f64) -> EdgeResult {
    // Optimization: Use SIMD kernel for fully isolated systems
    let pbc = cell.pbc();
    if !pbc.x && !pbc.y && !pbc.z {
        return brute_force_search_simd(positions, cutoff);
    }

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
    fn test_brute_force_simd_vs_serial() {
        let mut positions = Vec::new();
        let n = 100;
        let mut rng = 42u64;
        let mut next_rand = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng as f64) / (u64::MAX as f64)
        };

        for _ in 0..n {
            positions.push(Vector3::new(
                next_rand() * 10.0,
                next_rand() * 10.0,
                next_rand() * 10.0,
            ));
        }

        let cutoff = 2.5;

        // Serial result
        let mut edge_i_serial = Vec::new();
        let mut edge_j_serial = Vec::new();
        let mut shifts_serial = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let dist_sq = (positions[i] - positions[j]).norm_squared();
                if dist_sq < cutoff * cutoff {
                    edge_i_serial.push(i as i64);
                    edge_j_serial.push(j as i64);
                    shifts_serial.push(0);
                    shifts_serial.push(0);
                    shifts_serial.push(0);
                }
            }
        }

        // SIMD result
        let (ei_simd, ej_simd, s_simd) = brute_force_search_simd(&positions, cutoff);

        assert_eq!(ei_simd, edge_i_serial);
        assert_eq!(ej_simd, edge_j_serial);
        assert_eq!(s_simd, shifts_serial);
    }

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
        let cell = Cell::new(h, Vector3::new(true, true, true)).unwrap();

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
        let cell = Cell::new(h, Vector3::new(true, true, true)).unwrap();

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
        let cell = Cell::new(h, Vector3::new(true, true, true)).unwrap();

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
        let cell = Cell::new(h, Vector3::new(true, true, true)).unwrap();

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
        let cell = Cell::new(h, Vector3::new(true, true, true)).unwrap();

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
        let cell = Cell::new(h, Vector3::new(true, true, true)).unwrap();

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
        let cell = Cell::new(h, Vector3::new(true, true, true)).unwrap();
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
        let cell = Cell::new(h, Vector3::new(true, true, true)).unwrap();
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
        let cell = Cell::new(h, Vector3::new(true, true, true)).unwrap();
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
                let cell = Cell::new(h, Vector3::new(true, true, true)).unwrap();

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

            #[test]
            fn test_brute_force_isolated_correctness(
                cutoff in 1.0..5.0,
                positions_data in prop::collection::vec(prop::collection::vec(-10.0..10.0, 3), 2..100)
            ) {
                let positions: Vec<Vector3<f64>> = positions_data.into_iter().map(|p| Vector3::new(p[0], p[1], p[2])).collect();

                // Create a cell large enough that no PBC image can ever be within cutoff
                let h = Matrix3::identity() * 1000.0;
                let cell = Cell::new(h, Vector3::new(false, false, false)).unwrap();

                // Reference: simple O(N^2) without any Cell logic
                let mut expected = Vec::new();
                for i in 0..positions.len() {
                    for j in (i+1)..positions.len() {
                        let dist = (positions[i] - positions[j]).norm();
                        if dist < cutoff {
                            expected.push((i as i64, j as i64));
                        }
                    }
                }
                expected.sort();

                let (ei, ej, shifts) = brute_force_search_full(&cell, &positions, cutoff);
                let mut results: Vec<(i64, i64)> = ei.into_iter().zip(ej.into_iter()).collect();
                results.sort();

                assert_eq!(results, expected);
                for s in shifts {
                    assert_eq!(s, 0, "Shift must be zero for isolated system");
                }
            }
        }
    }
}
