# Specification: Multi-Cutoff Support

## 1. Overview
This feature enables `neighborlist-rs` to compute neighbor lists for multiple cutoff radii in a single pass. This is optimized for molecular mechanics and ML potentials that require different interaction ranges (e.g., short-range bonded vs. long-range electrostatic).

## 2. API Changes

### Rust (`lib.rs`)
*   **New Function:** `build_neighborlists_multi`
    *   **Arguments:**
        *   `py`: Python token.
        *   `cell`: `Option<&PyCell>`
        *   `positions`: `PyReadonlyArray2<f64>` (N, 3)
        *   `cutoffs`: `Vec<f64>` (List of cutoffs)
        *   `parallel`: `bool` (default: true)
    *   **Returns:** `PyResult<Bound<PyDict>>`
        *   The dictionary keys will be the cutoffs (as floats).
        *   The values will be the standard result dictionaries (`edge_i`, `edge_j`, `shift`).

### Python
*   Exposed as `neighborlist_rs.build_neighborlists_multi(cell, positions, cutoffs, parallel=True)`.

## 3. Core Logic (`search.rs`)

### `CellList::build`
*   No major changes needed. The Cell List is constructed based on the **maximum** cutoff found in the `cutoffs` vector.
*   $R_{grid} = \max(cutoffs)$.

### `CellList::par_search_multi`
*   New method similar to `par_search_optimized`.
*   **Input:** `cutoffs: &[f64]`.
*   **Storage:** Instead of a single `edge_i` vector, maintain `Vec<Vec<i64>>` (one vector per cutoff).
*   **Search Loop:**
    *   Calculate $d^2$.
    *   Iterate `for (k, &rc) in cutoffs.iter().enumerate()`:
        *   If $d^2 < rc^2$, append to the $k$-th result lists.
*   **Optimization:**
    *   Sort `cutoffs` internally? (Might complicate returning the dict if keys need to match input order. Better to just loop. N_cutoffs is small, usually < 5).
    *   Use the max cutoff for the spatial grid traversal loop limits.

## 4. Testing Plan
*   **Correctness:**
    *   Generate random system.
    *   Call `multi([3.0, 5.0])`.
    *   Call `single(3.0)` and `single(5.0)`.
    *   Assert results are identical (sorted).
*   **Performance:**
    *   Benchmark `multi` vs `sum(singles)` for [6.0, 14.0, 20.0].

## 5. Constraints
*   The memory overhead of storing multiple neighbor lists simultaneously might be high for large systems.
*   We assume `cutoffs` list is small.
