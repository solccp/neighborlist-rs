# Track Plan: Holistic Codebase Optimization and Large-Scale Refactor

## Phase 1: Profiling & Baseline
- [x] Task: Instrument code with `tracing` or `perf` hooks to measure bin-search vs overhead 2bfddf0
- [x] Task: Implement a detailed memory profiler (e.g., using `dhat` or `heaptrack` in tests) 0c1a379
- [ ] Task: Establish 10k/20k atom baseline benchmarks in a dedicated benchmark folder
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Profiling & Baseline' (Protocol in workflow.md)

## Phase 2: Spatial Sorting (Cache Locality)
- [ ] Task: Implement Z-order (Morton) index calculation for atom positions
- [ ] Task: Refactor `CellList::build` to reorder the `pos_wrapped` and `atom_shifts` arrays based on spatial index
- [ ] Task: Write tests to verify that spatial reordering doesn't break neighbor indexing
- [ ] Task: Verify cache hit rate improvement using `perf stat` or equivalent
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Spatial Sorting (Cache Locality)' (Protocol in workflow.md)

## Phase 3: Buffer Management & Allocation Optimization
- [ ] Task: Refactor `par_search_optimized` to use a more compact storage for intermediate neighbor counts
- [ ] Task: Explore using `smallvec` or in-place sorting to reduce peak heap usage during the "Fill" pass
- [ ] Task: Minimize Python/Rust conversion overhead by optimizing the dictionary construction in `lib.rs`
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Buffer Management & Allocation Optimization' (Protocol in workflow.md)

## Phase 4: Parallelization Tuning
- [ ] Task: Implement adaptive chunk sizing for Rayon based on system size and CPU count
- [ ] Task: Optimize the "Pass 1" counting phase to be more work-stealing friendly
- [ ] Task: Benchmark scaling on 1, 2, 4, 8, 16, 20+ threads to find the saturation point
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Parallelization Tuning' (Protocol in workflow.md)

## Phase 5: Final Validation
- [ ] Task: Run full `proptest` suite with high iteration count (1000+) to ensure robustness
- [ ] Task: Final comparison against `vesin` across all cutoffs (6.0, 14.0, 20.0)
- [ ] Task: Update `README.md` with new performance metrics and scaling charts
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Final Validation' (Protocol in workflow.md)
