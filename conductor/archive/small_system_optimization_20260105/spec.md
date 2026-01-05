# Specification: Small System Optimization (< 1000 atoms)

## 1. Overview
The current implementation uses a brute-force $O(N^2)$ path for systems with $N < 500$ atoms. While efficient, it lacks SIMD acceleration and relies on heap-allocated vectors for temporary results. This track aims to implement a high-performance SIMD kernel and stack-based memory management to minimize overhead and beat the `vesin` baseline for small, non-PBC systems.

## 2. Functional Requirements
### 2.1 SIMD Brute-Force Kernel
- Implement an explicit SIMD interaction loop using the `wide` crate (`f64x4`).
- Target non-PBC (isolated) systems first, as they are the primary use case for this optimization.
- The kernel should calculate distance-squared and compare against the cutoff without per-iteration heap allocations.

### 2.2 Stack-Allocated Scratchpad
- Use stack-allocated arrays (e.g., `[f64; 1024]`) for atom positions and temporary data when $N$ is below a safety threshold.
- Implement a fallback to heap allocation if $N$ exceeds the stack threshold (e.g., $N > 1024$) to prevent stack overflow.

### 2.3 Strategy Crossover Tuning
- Re-evaluate the `BRUTE_FORCE_THRESHOLD` (currently 500) once the SIMD kernel is implemented.
- Ensure the transition between brute-force and cell-lists remains seamless and correct.

## 3. Non-Functional Requirements
- **Performance:** `neighborlist-rs` should be measurably faster than `vesin` for 100 and 1000 atom systems in `benchmarks/comprehensive_benchmark.py`.
- **Safety:** Use `MaybeUninit` or similar primitives carefully if stack arrays are used, ensuring no UB is introduced.
- **Determinism:** Results must remain identical to the existing implementation.

## 4. Acceptance Criteria
- `pytest tests/` passes.
- `cargo test` passes.
- `benchmarks/comprehensive_benchmark.py` shows:
    - Speedup for "100 (non-PBC)".
    - Speedup for "1,000 (non-PBC)".
    - `neighborlist-rs` timing < `vesin` timing for these cases.

## 5. Out of Scope
- Optimizing PBC systems for very high atom counts (this is handled by the cell-list logic).
- Changing the public Python API signatures.
