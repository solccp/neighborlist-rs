# Plan: Small System Optimization (< 1000 atoms)

## Phase 1: Baseline & Regression Testing [checkpoint: 003930f]
- [x] Task: Create granular micro-benchmarks for systems of size 50, 100, 200, 500, 1000 atoms in `benchmarks/small_systems.py`. [3d2a40e]
- [x] Task: Write property tests in `src/search.rs` specifically for isolated systems to ensure edge-case correctness (e.g., all atoms at origin, atoms very far apart). [752cceb]
- [x] Task: Conductor - User Manual Verification 'Baseline & Regression Testing' (Protocol in workflow.md) [003930f]

## Phase 2: SIMD Brute-Force Kernel [checkpoint: ceb0a04]
- [x] Task: Implement `brute_force_search_simd` in `src/search.rs` using `f64x4` from the `wide` crate for isolated systems. [ceb0a04]
- [x] Task: Integrate the SIMD kernel into the `brute_force_search_full` entry point. [ceb0a04]
- [x] Task: Write unit tests to verify `brute_force_search_simd` produces bit-identical results to the serial implementation. [ceb0a04]
- [x] Task: Conductor - User Manual Verification 'SIMD Brute-Force Kernel' (Protocol in workflow.md) [ceb0a04]

## Phase 3: Memory & Threshold Optimization [checkpoint: 1573576]
- [x] Task: Implement a stack-allocated position buffer (using `ArrayVec` or a local array with a safety threshold of $N=512$) in `search_single`. [1573576]
- [x] Task: Add a safety fallback to heap allocation for systems exceeding the stack threshold. [1573576]
- [x] Task: Benchmark the crossover point and update `BRUTE_FORCE_THRESHOLD` based on new performance data. [1573576]
- [x] Task: Implement dynamic configuration system (`src/config.rs`) to replace compile-time auto-tuning. [dad953f]
- [x] Task: Conductor - User Manual Verification 'Memory & Threshold Optimization' (Protocol in workflow.md) [1573576]

## Phase 4: Final Verification [checkpoint: 89b423d]
- [x] Task: Run `benchmarks/comprehensive_benchmark.py` and verify `neighborlist-rs` < `vesin` for the 100 and 1000 atom cases. [89b423d]
- [x] Task: Audit `src/lib.rs` for any remaining redundant allocations in the non-PBC path. [89b423d]
- [x] Task: Conductor - User Manual Verification 'Final Verification' (Protocol in workflow.md) [89b423d]
