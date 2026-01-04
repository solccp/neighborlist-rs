# Track: Multi-Cutoff Support

**Status:** Planned
**Owner:** Sol
**Created:** 2026-01-04

## Context
In modern ML potentials (like JmolNet), different interactions often require different cutoffs:
*   **Short-range (MLP):** ~6.0 Å
*   **Dispersion:** ~14.0 Å
*   **Electrostatics/Long-range:** ~20.0 Å+

Currently, obtaining these requires calling `neighborlist-rs` three times. This is inefficient because:
1.  **Redundant Work:** Cell lists are built and sorted three times.
2.  **Redundant Search:** The spatial traversal for the largest cutoff (20.0 Å) implicitly visits all pairs needed for the smaller cutoffs.

## Objective
Implement a `build_neighborlists_multi` function that accepts a list of cutoffs and returns neighbor lists for all of them in a single, efficient pass.

## Requirements

### 1. API Design
*   **Input:** `cutoffs`: List of floats, e.g., `[6.0, 14.0, 20.0]`.
*   **Output:** A dictionary mapping each cutoff to its neighbor list result (edge indices and shifts).
    *   Example: `{ 6.0: {"edge_i": ...}, 14.0: {"edge_i": ...} }`

### 2. Implementation Strategy
*   **Grid Construction:** Build the Cell List using the **largest** provided cutoff ($R_{max}$).
*   **Single Traversal:** Iterate through atoms and neighboring cells based on $R_{max}$.
*   **Bucketing:** For each pair $(i, j)$ with distance $d$:
    *   Iterate through the sorted cutoffs.
    *   If $d < R_k$, add the pair to the result list for $R_k$.
    *   (Optimization: Since $R_1 < R_2 < R_3$, if $d < R_1$, it is also $< R_2$ and $< R_3$. We simply add to all applicable lists).

### 3. Performance Goals
*   The cost of `multi_pass([r1, r2, r3])` should be only marginally higher than `single_pass(r3)` (due to extra memory writes), and significantly faster than `single_pass(r1) + single_pass(r2) + single_pass(r3)`.

## Phases

### Phase 1: API & Refactoring
*   Define `build_neighborlists_multi` in `lib.rs`.
*   Update `search.rs` to support a `MultiCellList` struct or extend the existing one to handle `Vec<f64>` cutoffs.

### Phase 2: Core Implementation
*   Implement the fused search kernel.
*   Ensure thread-safety and proper parallel reduction (each thread maintains vectors for all cutoffs).

### Phase 3: Testing & Validation
*   **Correctness:** Verify that `multi([r1, r2])` returns identical results to calling `single(r1)` and `single(r2)` separately.
*   **Edge Cases:** Verify behavior when cutoffs are identical or unsorted.

### Phase 4: Benchmarking
*   Measure speedup on standard datasets (Ethanol, etc.) for the [6.0, 14.0, 20.0] scenario.

## Risks & Mitigations
*   **Memory Bandwidth:** Writing to 3x output vectors might saturate bandwidth.
    *   *Mitigation:* The compute cost of geometry usually outweighs write cost. If needed, we can optimize memory layout (structure of arrays vs array of structures).