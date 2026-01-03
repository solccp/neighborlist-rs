# Rust Neighborlist Library — Requirements & Specification

**Project Codename:** `rs-neighborlist`
**Primary Consumer:** Python/PyTorch MLIP training + inference
**Secondary Consumer:** MD runtime (custom engine or external integrator)
**Language:** Rust (stable)
**Python Binding:** PyO3 + maturin (primary), optional C-ABI (secondary)

---

## 1. Introduction

### 1.1 Background and Goals
This library provides high-performance neighborlist construction for atomistic systems, designed to be consumed by equivariant GNNs (e.g., e3nn/cuequivariance) which require correct displacement vectors under Periodic Boundary Conditions (PBC).

Key capabilities:
- **System Support:** Non-periodic (molecules) and Periodic (crystals/PBC) systems.
- **Multi-Cutoff:** Concurrent generation of lists for different interactions (e.g., local GNN vs dispersion vs Coulomb).
- **MD Support:** Optional skin and rebuild logic for dynamic molecular dynamics simulations.

**Critical Requirement:** PBC image shift vectors must be provided per edge to allow reconstruction of the exact displacement vector in the consumer application.

### 1.2 Scope

#### Phase 1: Core Features
1. **Neighbor Search:**
   - Molecules (no PBC)
   - Crystals (PBC with cell)
2. **Output Format:**
   - `edge_i`, `edge_j` (int64)
   - `edge_shift` (int32[E, 3]) for PBC (zero for non-PBC)
3. **Multiple Neighborlists (Single Call):**
   - `local` list (GNN cutoff)
   - `dispersion` list (DFTD3 cutoff)
   - `coul_real` list (optional)
4. **Binding:**
   - Python binding returning NumPy arrays without unnecessary copies.
5. **Determinism:**
   - Output order stability controlled by policy.

#### Phase 2: MD-Oriented Features
1. **Dynamic Management:**
   - Skin buffer (`r_skin`) logic.
   - Reference position tracking.
   - Rebuild triggers (`max_displacement > skin/2`).
2. **Optimization:**
   - Cached cell-list bins and incremental updates (optional).
   - Separate skins per list type.

### 1.3 Out of Scope (Non-Goals)
- Autograd / gradients.
- Energy/force evaluation.
- Long-range electrostatics (PME reciprocal) — handled by external tools like `torch-pme`.
- Dispersion physics — handled by `DFTD3`.
- GPU neighborlist construction.
- Bonded exclusions (1–2, 1–3, 1–4).
- Constraints / rigid bodies.
- MPI parallelization (domain decomposition).

---

## 2. Definitions

- **Cutoff (`r_cut`)**: Maximum interaction radius for a given neighborlist type.
- **Skin (`r_skin`)**: Buffer radius for dynamic lists to avoid rebuilding every step.
  - `r_list = r_cut + r_skin`
  - Rebuild condition: `max_i ||r_i - r_i_ref|| > r_skin/2`
- **Edge**: Directed pair `(i, j)` for message passing.
- **PBC Shift (`s_ij`)**: Integer vector `[sx, sy, sz]` such that:
  - `r_j_image = r_j + s_ij * cell` (matrix multiplication)
  - `d_ij = r_j_image - r_i`

---

## 3. Functional Requirements

### 3.1 System Types
- **Non-PBC**: No cell defined. `edge_shift` is always zero.
- **PBC**: Full 3x3 cell matrix with per-dimension flags `pbc = [bool; 3]`.

### 3.2 Coordinate Conventions
- **Positions**: Cartesian coordinates in Å (preferred).
- **Cell Matrix (`H`)**: Shape (3, 3).
  - Convention: Column vectors `a, b, c` as columns of `H`.

### 3.3 Neighborlist Types
The library must support producing multiple lists in a single traversal:

1.  **Local List**
    - Cutoff: `r_local`
    - Usage: GNN message passing.
    - Default: Directed edges.
2.  **Dispersion List**
    - Cutoff: `r_disp`
    - Usage: DFTD3 pair enumeration.
    - Default: Undirected pairs (to avoid double counting).
3.  **Coulomb Real-Space List** (Optional)
    - Cutoff: `r_coul_real`

### 3.4 Output Options
- **Directedness (`directed: bool`):**
  - `true`: Include `(i, j)` and `(j, i)` (or canonical equivalents).
  - `false`: Output only one pair, typically `i < j`.
- **Self-Interaction:** `include_self` (default: `false`).
- **Sort Policy:**
  - `None`: Fastest, order depends on internal bin traversal.
  - `ByIThenJ`: Stable output for reproducibility.

### 3.5 Correctness & PBC
- **Minimum Image:** Must generate correct minimum-image conventions or explicit image shifts.
- **Triclinic Cells:** Must handle non-orthogonal cells correctly.
- **Validation:** `edge_shift` must yield a displacement `||d_ij|| <= r_cut`.

---

## 4. Technical Architecture

### 4.1 Search Algorithm
- **Method:** Cell Lists / Binning.
- **Bin Size:** `r_max = max(r_local, r_disp, r_coul_real)`.
- **Process:**
  1. Map atoms to integer bin coordinates.
  2. Search candidate bins (handling PBC wrapping).
  3. Compute squared distance `d^2`.
  4. Compare `d^2` against all active cutoffs (`r_local^2`, `r_disp^2`, etc.) and append to relevant lists. This avoids re-traversing neighbors for different lists.

### 4.2 PBC Implementation
- **Triclinic Support:** Use fractional coordinates for image finding.
  - `d = r_j - r_i`
  - `f = H^{-1} d`
  - Apply minimum image to `f` (round to nearest integer).
  - `shift = -round(f)`
- **Convention:** Explicitly define sign convention and verify against Python reconstruction.

### 4.3 Performance Requirements
- **Complexity:** O(N) for typical densities.
- **Memory:** Minimize allocations via heuristics (estimated neighbors per atom).
- **Safety:** No panics on valid input. Handle sparse/dense extremes gracefully.

---

## 5. API Specification

### 5.1 Rust Core API

```rust
pub struct NeighborConfig {
    pub pbc: [bool; 3],
    pub r_local: f64,
    pub r_disp: f64,
    pub r_coul_real: Option<f64>,
    pub directed_local: bool,
    pub directed_disp: bool,
    pub sort_policy: SortPolicy,
}

pub enum SortPolicy { None, ByIThenJ }

pub struct NeighborList {
    pub edge_i: Vec<i64>,
    pub edge_j: Vec<i64>,
    pub shift: Vec<[i32; 3]>,
}

pub struct NeighborLists {
    pub local: NeighborList,
    pub disp: NeighborList,
    pub coul_real: Option<NeighborList>,
}

pub fn build_neighborlists(
    positions: &[[f64; 3]],
    cell: Option<[[f64; 3]; 3]>,
    config: &NeighborConfig,
) -> NeighborLists;
```

### 5.2 Python Binding (PyO3)

**Signature:**
```python
def build_neighborlists(
    positions: np.ndarray,          # (N, 3) float64
    cell: Optional[np.ndarray],     # (3, 3) float64
    pbc: Tuple[bool, bool, bool],
    r_local: float,
    r_disp: float,
    r_coul_real: Optional[float] = None,
    directed_local: bool = True,
    directed_disp: bool = False,
    sort_policy: str = "none"       # "none" | "ij"
) -> Dict[str, Any]: ...
```

**Return Object:**
```python
{
    "local": {
        "edge_i": np.ndarray, # int64
        "edge_j": np.ndarray, # int64
        "shift":  np.ndarray  # int32, shape (E, 3)
    },
    "disp": { ... },
    "coul_real": { ... } # Or None
}
```
**Zero-Copy:** Bindings should prefer returning NumPy arrays backed by Rust memory where possible, or efficient copies if ownership complexity is too high.

### 5.3 MD Dynamic API (Phase 2)

```rust
pub struct DynamicNeighborManager {
    // State tracking for rebuilds
}

impl DynamicNeighborManager {
    pub fn new(config: NeighborConfig, skins: SkinConfig) -> Self;
    pub fn update(&mut self, positions: &[[f64; 3]], cell: Option<[[f64; 3]; 3]>);
    pub fn needs_rebuild(&self, positions: &[[f64; 3]]) -> bool;
    pub fn get(&self) -> &NeighborLists;
}
```

---

## 6. Testing and Validation

### 6.1 Unit Tests (Rust)
- **Non-PBC:** Compare against brute-force ($O(N^2)$) search.
- **PBC Orthorhombic:** Compare against brute-force with minimum image.
- **PBC Triclinic:** Validate shift reconstruction `r_j + shift * cell - r_i` is within cutoff.

### 6.2 Property-Based Tests (`proptest`)
- Random invertible cells.
- Random point clouds.
- Invariants:
  - All edges returned are within cutoff.
  - No missing edges compared to brute force.

### 6.3 Python Integration Tests
- Validate shapes and dtypes.
- Benchmark for typical N (1k - 10k atoms).
- Verify reconstruction in PyTorch/NumPy matches expectations.

---

## 7. Deliverables & Acceptance

### 7.1 Phase 1 Deliverables
- [ ] Rust crate `rs-neighborlist`.
- [ ] PyO3 module `rs_neighborlist`.
- [ ] Core `build_neighborlists` implementation.
- [ ] Unit tests and basic benchmarks.
- [ ] Documentation and Examples.

### 7.2 Phase 2 Deliverables
- [ ] `DynamicNeighborManager` with skin logic.
- [ ] MD loop benchmarks.

### 7.3 Acceptance Criteria
1. **Correctness:** Matches brute-force for Molecules, Orthorhombic, and Triclinic systems.
2. **Performance:** Linear scaling $O(N)$ observed for 10k+ atoms.
3. **Integration:** Python bindings successfully produce edge indices and shifts usable by `torch`.