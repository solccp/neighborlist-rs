# Specification: Batched Neighbor List Processing

## 1. Overview
This feature allows processing a batch of atomic structures in a single Rust call. It is designed to maximize CPU utilization by parallelizing across different systems in the batch, which is especially beneficial for datasets containing many small molecules where single-system parallelism is limited by overhead.

## 2. API Changes

### Rust (`lib.rs`)
*   **New Function:** `build_neighborlists_batch`
    *   **Arguments:**
        *   `py`: Python token.
        *   `positions`: `PyReadonlyArray2<f64>` (N_total, 3) - Concatenated positions of all atoms in the batch.
        *   `batch`: `PyReadonlyArray1<i32>` (N_total) - System index for each atom.
        *   `cells`: `Option<PyReadonlyArray3<f64>>` (B, 3, 3) - Array of cell matrices for each system.
        *   `cutoff`: `f64` - Scalar cutoff radius.
        *   `parallel`: `bool` (default: true)
    *   **Returns:** `PyResult<Bound<PyDict>>`
        *   A dictionary containing concatenated results:
            *   `edge_i`: (M_total,) Global indices of source atoms.
            *   `edge_j`: (M_total,) Global indices of target atoms.
            *   `shift`: (M_total, 3) Periodic shifts.

### Python
*   Exposed as `neighborlist_rs.build_neighborlists_batch(positions, batch, cells=None, cutoff=5.0, parallel=True)`.

*   **New Function:** `build_neighborlists_batch_multi`
    *   **Arguments:**
        *   `py`: Python token.
        *   `positions`: `PyReadonlyArray2<f64>` (N_total, 3)
        *   `batch`: `PyReadonlyArray1<i32>` (N_total)
        *   `cells`: `Option<PyReadonlyArray3<f64>>` (B, 3, 3)
        *   `cutoffs`: `Vec<f64>` (List of cutoffs)
        *   `parallel`: `bool` (default: true)
    *   **Returns:** `PyResult<Bound<PyDict>>`
        *   A dictionary mapping each cutoff to a concatenated result dictionary (`edge_i`, `edge_j`, `shift`).

## 3. Core Logic

### Data Splitting
1.  Verify `positions` and `batch` lengths match.
2.  Determine system boundaries from the `batch` array (assuming sorted or grouped by system).
3.  Extract per-system slices of positions.

### Parallel Search
1.  Iterate over systems in parallel using Rayon.
2.  For each system:
    *   Initialize `CellList` or use brute force fallback.
    *   Perform search using the existing single-system kernels.
    *   Collect local edges.

### Aggregation
1.  Apply atom index offsets to local edges to convert them to global indices.
2.  Concatenate vectors from all systems into final output arrays.

## 4. Testing Plan
1.  **Correctness:**
    *   Construct a batch of 3 diverse systems (e.g. water, ethanol, silicon).
    *   Call `build_neighborlists_batch`.
    *   Call `build_neighborlists` for each individually.
    *   Verify concatenated results match.
2.  **Edge Cases:**
    *   Single system in batch.
    *   Empty batch.
    *   Systems with varying sizes.
3.  **Performance:**
    *   Benchmark 128 small molecules (e.g. water) in a batch vs. Python loop.
