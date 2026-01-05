# Plan: Small System Optimization (< 1000 atoms)

## Phase 1: Baseline & Regression Testing [checkpoint: 003930f]
- [x] Task: Create granular micro-benchmarks for systems of size 50, 100, 200, 500, 1000 atoms in `benchmarks/small_systems.py`. [3d2a40e]
- [x] Task: Write property tests in `src/search.rs` specifically for isolated systems to ensure edge-case correctness (e.g., all atoms at origin, atoms very far apart). [752cceb]
- [ ] Task: Conductor - User Manual Verification 'Baseline & Regression Testing' (Protocol in workflow.md)

## Phase 2: SIMD Brute-Force Kernel [checkpoint: ceb0a04]
- [x] Task: Implement `brute_force_search_simd` in `src/search.rs` using `f64x4` from the `wide` crate for isolated systems. [ceb0a04]
- [x] Task: Integrate the SIMD kernel into the `brute_force_search_full` entry point. [ceb0a04]
- [x] Task: Write unit tests to verify `brute_force_search_simd` produces bit-identical results to the serial implementation. [ceb0a04]
- [ ] Task: Conductor - User Manual Verification 'SIMD Brute-Force Kernel' (Protocol in workflow.md)

## Phase 3: Memory & Threshold Optimization
- [ ] Task: Implement a stack-allocated position buffer (using `ArrayVec` or a local array with a safety threshold of $N=512$) in `search_single`.
- [ ] Task: Add a safety fallback to heap allocation for systems exceeding the stack threshold.
- [ ] Task: Benchmark the crossover point and update `BRUTE_FORCE_THRESHOLD` based on new performance data.
- [ ] Task: Conductor - User Manual Verification 'Memory & Threshold Optimization' (Protocol in workflow.md)

## Phase 4: Final Verification
- [ ] Task: Run `benchmarks/comprehensive_benchmark.py` and verify `neighborlist-rs` < `vesin` for the 100 and 1000 atom cases.
- [ ] Task: Audit `src/lib.rs` for any remaining redundant allocations in the non-PBC path.
- [ ] Task: Conductor - User Manual Verification 'Final Verification' (Protocol in workflow.md)
