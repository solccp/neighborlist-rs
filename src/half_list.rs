//! Half neighbor list for large cutoffs, written straight into its output.
//!
//! The cell-list kernels in `search` use bins at least one cutoff wide, so when
//! the cutoff is comparable to or larger than the cell (e.g. D3 at 60 Bohr)
//! they scan ~6x more candidates than they keep. Here:
//!
//! - bins are sized independently of the cell, and neighbor bins (images
//!   included) come from a precomputed stencil of bin offsets, pruned per atom;
//! - each unordered pair is scanned once: from the atom with the lower sorted
//!   index, or for a self-image, only for its positive shift;
//! - a branchless count pass sizes each atom's slot range, and the fill pass
//!   writes into the final arrays exactly once. At these list sizes the output
//!   (tens of MB) costs more than the distances, so intermediate buffers and a
//!   concatenation would dominate.

use crate::cell::Cell;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::mem::MaybeUninit;

/// Half list: pairs `(i[e], j[e])` with `i < j`, or `i == j` and a shift whose
/// first non-zero component is positive. `r = x_j - x_i + shifts[e] · cell`
/// (shift as a row vector, cell rows as lattice vectors).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HalfList {
    pub i: Vec<u32>,
    pub j: Vec<u32>,
    pub shifts: Vec<[i32; 3]>,
}

impl HalfList {
    pub fn len(&self) -> usize {
        self.i.len()
    }

    pub fn is_empty(&self) -> bool {
        self.i.is_empty()
    }
}

/// Target bin width as a fraction of the cutoff, and its floor in length units.
/// Smaller bins prune more candidates but cost more bin visits; per-atom
/// pruning makes wide bins cheap. Tuned on 384-atom solids (SnO2, Si, Al slab)
/// at 6, 12 and 31.75 A cutoffs.
const BIN_FRACTION: f64 = 0.33;
const MIN_BIN_WIDTH: f64 = 6.0;
/// Upper bound on bins per atom; bins are widened beyond the target if needed.
const MAX_BINS_PER_ATOM: usize = 4;

/// Candidates compacted per block in the fill pass.
const BLOCK: usize = 64;

struct Axis {
    periodic: bool,
    /// Fractional origin and bin size along this lattice vector.
    lo: f64,
    df: f64,
    nb: i32,
    /// Stencil reach in bins.
    reach: i32,
}

impl Axis {
    fn bin(&self, f: f64) -> i32 {
        (((f - self.lo) / self.df).floor() as i32).clamp(0, self.nb - 1)
    }

    /// Target bin and image shift for bin `b` moved by `d`, or `None` if it
    /// leaves the box along a non-periodic axis.
    fn step(&self, b: i32, d: i32) -> Option<(i32, i32)> {
        let t = b + d;
        if self.periodic {
            Some((t.rem_euclid(self.nb), t.div_euclid(self.nb)))
        } else if (0..self.nb).contains(&t) {
            Some((t, 0))
        } else {
            None
        }
    }
}

struct StencilEntry {
    d: [i32; 3],
    /// Cartesian offset between the centers of the two bins.
    delta: Vector3<f64>,
}

/// Atoms sorted by bin, with the geometry needed to scan their neighbors.
struct Grid {
    rows: [Vector3<f64>; 3],
    axes: [Axis; 3],
    stencil: Vec<StencilEntry>,
    /// `start[b]..start[b + 1]` are the sorted atoms in linear bin `b`.
    start: Vec<usize>,
    /// Per sorted atom: wrapped position (SoA), wrap shift, original index,
    /// bin, and offset from its bin's center.
    px: Vec<f64>,
    py: Vec<f64>,
    pz: Vec<f64>,
    wrap: Vec<[i32; 3]>,
    orig: Vec<u32>,
    bin: Vec<[i32; 3]>,
    from_center: Vec<Vector3<f64>>,
    cut2: f64,
    prune2: f64,
}

/// One neighbor bin of an atom: sorted atoms `lo..hi` (already restricted to
/// `j > i`), their image shift, and `offset = shift · cell - x_i`, so that
/// `r_ij = x_j + offset`. `self_image` is set when the atom's own image at
/// `shift` is a pair to emit.
struct Visit {
    lo: usize,
    hi: usize,
    shift: [i32; 3],
    offset: Vector3<f64>,
    self_image: bool,
}

impl Grid {
    fn new(cell: &Cell, positions: &[Vector3<f64>], cutoff: f64) -> Self {
        let n = positions.len();
        let h = cell.h();
        let pbc = cell.pbc();
        let rows: [Vector3<f64>; 3] = std::array::from_fn(|k| h.column(k).into_owned());
        let perp = cell.perpendicular_widths();

        // Fractional coordinates, wrapped along periodic axes only.
        let mut frac: Vec<Vector3<f64>> = positions.iter().map(|p| cell.to_fractional(p)).collect();
        let mut wrap = vec![[0i32; 3]; n];
        for (f, w) in frac.iter_mut().zip(wrap.iter_mut()) {
            for k in 0..3 {
                if pbc[k] {
                    let fl = f[k].floor();
                    f[k] -= fl;
                    w[k] = -(fl as i32);
                }
            }
        }

        // Fractional origin and span per axis: the whole cell along periodic
        // axes, the atoms' extent along non-periodic ones.
        let extent: [(f64, f64); 3] = std::array::from_fn(|k| {
            if pbc[k] {
                (0.0, 1.0)
            } else {
                let lo = frac.iter().map(|f| f[k]).fold(f64::INFINITY, f64::min);
                let hi = frac.iter().map(|f| f[k]).fold(f64::NEG_INFINITY, f64::max);
                // Pad so the top atom does not sit on the last bin's upper face.
                (lo, (hi - lo).max(1e-12) * (1.0 + 1e-9))
            }
        });
        let width: [f64; 3] = std::array::from_fn(|k| perp[k] * extent[k].1);

        // Cap the bin count so sparse systems (e.g. molecules spread over a
        // large box) do not allocate far more bins than atoms.
        let max_bins = (MAX_BINS_PER_ATOM * n).max(64) as f64;
        let bins_for = |t: f64| -> f64 { width.iter().map(|w| (w / t).round().max(1.0)).product() };
        let mut target = (cutoff * BIN_FRACTION).max(MIN_BIN_WIDTH);
        while bins_for(target) > max_bins {
            target *= 1.25;
        }

        let axes: [Axis; 3] = std::array::from_fn(|k| {
            let (lo, span) = extent[k];
            let nb = ((width[k] / target).round() as i32).max(1);
            let reach = (cutoff * nb as f64 / width[k]).ceil();
            Axis {
                periodic: pbc[k],
                lo,
                df: span / nb as f64,
                nb,
                // Along a non-periodic axis no offset beyond the box can hit a
                // bin; clamping also keeps a zero-extent axis (planar or single
                // atom) from producing an unbounded stencil.
                reach: if pbc[k] {
                    reach as i32
                } else {
                    reach.min((nb - 1) as f64) as i32
                },
            }
        });

        // Bin basis vectors and the bounding radius of one bin.
        let v: [Vector3<f64>; 3] = std::array::from_fn(|k| rows[k] * axes[k].df);
        let r_bin = 0.5
            * [
                v[0] + v[1] + v[2],
                v[0] + v[1] - v[2],
                v[0] - v[1] + v[2],
                v[0] - v[1] - v[2],
            ]
            .iter()
            .map(|d| d.norm())
            .fold(0.0, f64::max);

        // Stencil: bin offsets whose boxes can hold points closer than the
        // cutoff (two points in bins with centers `delta` apart are at least
        // |delta| - 2 r_bin apart).
        let mut stencil = Vec::new();
        let keep2 = (cutoff + 2.0 * r_bin).powi(2);
        for dx in -axes[0].reach..=axes[0].reach {
            for dy in -axes[1].reach..=axes[1].reach {
                for dz in -axes[2].reach..=axes[2].reach {
                    let delta = v[0] * dx as f64 + v[1] * dy as f64 + v[2] * dz as f64;
                    if delta.norm_squared() < keep2 {
                        stencil.push(StencilEntry {
                            d: [dx, dy, dz],
                            delta,
                        });
                    }
                }
            }
        }

        // Counting sort of atoms by bin.
        let nbins = (axes[0].nb * axes[1].nb * axes[2].nb) as usize;
        let bin_of: Vec<[i32; 3]> = frac
            .iter()
            .map(|f| std::array::from_fn(|k| axes[k].bin(f[k])))
            .collect();
        let linear = |b: [i32; 3]| (b[0] + axes[0].nb * (b[1] + axes[1].nb * b[2])) as usize;
        let mut start = vec![0usize; nbins + 1];
        for b in &bin_of {
            start[linear(*b) + 1] += 1;
        }
        for k in 0..nbins {
            start[k + 1] += start[k];
        }
        let mut fill = start.clone();
        let mut order = vec![0usize; n];
        for (a, b) in bin_of.iter().enumerate() {
            let l = linear(*b);
            order[fill[l]] = a;
            fill[l] += 1;
        }

        // Wrapped positions as x + w · cell, so atoms already inside the cell
        // keep their exact input coordinates.
        let wrapped = |a: usize| {
            let w = wrap[a];
            positions[a] + rows[0] * w[0] as f64 + rows[1] * w[1] as f64 + rows[2] * w[2] as f64
        };
        let pos: Vec<Vector3<f64>> = order.iter().map(|&a| wrapped(a)).collect();
        let center = |b: [i32; 3]| -> Vector3<f64> {
            (0..3).fold(Vector3::zeros(), |acc, k| {
                acc + rows[k] * (axes[k].lo + (b[k] as f64 + 0.5) * axes[k].df)
            })
        };

        Grid {
            px: pos.iter().map(|p| p.x).collect(),
            py: pos.iter().map(|p| p.y).collect(),
            pz: pos.iter().map(|p| p.z).collect(),
            wrap: order.iter().map(|&a| wrap[a]).collect(),
            orig: order.iter().map(|&a| a as u32).collect(),
            bin: order.iter().map(|&a| bin_of[a]).collect(),
            from_center: order
                .iter()
                .zip(&pos)
                .map(|(&a, p)| p - center(bin_of[a]))
                .collect(),
            rows,
            axes,
            stencil,
            start,
            cut2: cutoff * cutoff,
            prune2: (cutoff + r_bin).powi(2),
        }
    }

    fn len(&self) -> usize {
        self.orig.len()
    }

    /// Calls `f` for every neighbor bin of sorted atom `i` that can hold a pair
    /// to emit. Count and fill passes share this, so they see the same
    /// candidates in the same order.
    fn visits(&self, i: usize, mut f: impl FnMut(&Visit)) {
        let b = self.bin[i];
        let u = self.from_center[i];
        let pi = Vector3::new(self.px[i], self.py[i], self.pz[i]);
        let [ax, ay, az] = &self.axes;
        for e in &self.stencil {
            // Bin center farther than cutoff + bin radius from the atom.
            if (e.delta - u).norm_squared() >= self.prune2 {
                continue;
            }
            let Some((tx, sx)) = ax.step(b[0], e.d[0]) else {
                continue;
            };
            let Some((ty, sy)) = ay.step(b[1], e.d[1]) else {
                continue;
            };
            let Some((tz, sz)) = az.step(b[2], e.d[2]) else {
                continue;
            };
            let l = (tx + ax.nb * (ty + ay.nb * tz)) as usize;
            let (s0, s1) = (self.start[l], self.start[l + 1]);
            if s1 <= i {
                continue;
            }
            let shift = [sx, sy, sz];
            let image =
                self.rows[0] * sx as f64 + self.rows[1] * sy as f64 + self.rows[2] * sz as f64;
            // Self-image: the wrap shifts cancel, so the pair's shift is
            // `shift`. +S and -S are the same pair; keep the positive one.
            let self_image = s0 <= i && shift > [0, 0, 0] && image.norm_squared() < self.cut2;
            f(&Visit {
                lo: s0.max(i + 1),
                hi: s1,
                shift,
                offset: image - pi,
                self_image,
            });
        }
    }

    #[inline(always)]
    fn dist2(&self, j: usize, o: &Vector3<f64>) -> f64 {
        let dx = self.px[j] + o.x;
        let dy = self.py[j] + o.y;
        let dz = self.pz[j] + o.z;
        dx * dx + dy * dy + dz * dz
    }

    fn count(&self, i: usize) -> usize {
        let mut c = 0;
        self.visits(i, |v| {
            c += v.self_image as usize;
            // Branchless: about half the candidates are kept, so a branch
            // here mispredicts constantly.
            c += (v.lo..v.hi)
                .map(|j| (self.dist2(j, &v.offset) < self.cut2) as usize)
                .sum::<usize>();
        });
        c
    }

    /// Writes sorted atom `i`'s pairs into `out_*`, which must have exactly
    /// `count(i)` slots. Returns the number written.
    fn fill(
        &self,
        i: usize,
        out_i: &mut [MaybeUninit<u32>],
        out_j: &mut [MaybeUninit<u32>],
        out_s: &mut [MaybeUninit<[i32; 3]>],
    ) -> usize {
        let oi = self.orig[i];
        let wi = self.wrap[i];
        let mut k = 0;
        let mut buf = [0usize; BLOCK];
        self.visits(i, |v| {
            if v.self_image {
                out_i[k].write(oi);
                out_j[k].write(oi);
                out_s[k].write(v.shift);
                k += 1;
            }
            let mut j0 = v.lo;
            while j0 < v.hi {
                let j1 = (j0 + BLOCK).min(v.hi);
                // Branchless compaction of kept candidates into `buf`.
                let mut m = 0;
                for j in j0..j1 {
                    buf[m] = j;
                    m += (self.dist2(j, &v.offset) < self.cut2) as usize;
                }
                for &j in &buf[..m] {
                    let oj = self.orig[j];
                    let wj = self.wrap[j];
                    let t: [i32; 3] = std::array::from_fn(|a| v.shift[a] + wj[a] - wi[a]);
                    // Report as (min, max); reversing the pair negates the shift.
                    let sign = if oi < oj { 1 } else { -1 };
                    out_i[k].write(oi.min(oj));
                    out_j[k].write(oi.max(oj));
                    out_s[k].write(t.map(|c| c * sign));
                    k += 1;
                }
                j0 = j1;
            }
        });
        k
    }
}

/// Builds the half list. `cell` holds the lattice vectors as the columns of
/// its internal matrix (see [`Cell`]). Runs on rayon's current pool; the
/// output order is independent of the thread count.
pub fn build(cell: &Cell, positions: &[Vector3<f64>], cutoff: f64) -> HalfList {
    if positions.is_empty() {
        return HalfList::default();
    }
    let grid = Grid::new(cell, positions, cutoff);
    let n = grid.len();

    let counts: Vec<usize> = (0..n).into_par_iter().map(|i| grid.count(i)).collect();
    let total: usize = counts.iter().sum();

    // Uninitialized output: zero-filling would map and fault in fresh pages on
    // every call, which costs more than writing them once.
    let mut out = HalfList {
        i: Vec::with_capacity(total),
        j: Vec::with_capacity(total),
        shifts: Vec::with_capacity(total),
    };
    {
        let mut rest = (
            &mut out.i.spare_capacity_mut()[..total],
            &mut out.j.spare_capacity_mut()[..total],
            &mut out.shifts.spare_capacity_mut()[..total],
        );
        let mut slots = Vec::with_capacity(n);
        for (a, &c) in counts.iter().enumerate() {
            let (si, ri) = std::mem::take(&mut rest.0).split_at_mut(c);
            let (sj, rj) = std::mem::take(&mut rest.1).split_at_mut(c);
            let (ss, rs) = std::mem::take(&mut rest.2).split_at_mut(c);
            rest = (ri, rj, rs);
            slots.push((a, si, sj, ss));
        }
        slots.into_par_iter().for_each(|(a, si, sj, ss)| {
            let written = grid.fill(a, si, sj, ss);
            // Both passes evaluate identical expressions on the same
            // candidates, so the counts agree exactly.
            assert_eq!(written, si.len(), "count and fill passes disagree");
        });
    }
    // SAFETY: the slot ranges partition 0..total, and every fill call wrote
    // all of its slots (checked by the assert above).
    unsafe {
        out.i.set_len(total);
        out.j.set_len(total);
        out.shifts.set_len(total);
    }
    out
}
