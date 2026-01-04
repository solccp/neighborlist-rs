# Track Plan: Holistic Codebase Optimization and Large-Scale Refactor

## Phase 1: Profiling & Baseline [checkpoint: 20c916c]
- [x] Task: Instrument code with `tracing` or `perf` hooks to measure bin-search vs overhead 2bfddf0
- [x] Task: Implement a detailed memory profiler (e.g., using `dhat` or `heaptrack` in tests) 0c1a379
- [x] Task: Establish 10k/20k atom baseline benchmarks in a dedicated benchmark folder c7f4815
- [x] Task: Conductor - User Manual Verification 'Phase 1: Profiling & Baseline' (Protocol in workflow.md) 20c916c

## Phase 2: Spatial Sorting (Cache Locality) [checkpoint: bf687dd]
- [x] Task: Implement Z-order (Morton) index calculation for atom positions 76b7cdf
- [x] Task: Refactor `CellList::build` to reorder the `pos_wrapped` and `atom_shifts` arrays based on spatial index 830ad06
- [x] Task: Write tests to verify that spatial reordering doesn't break neighbor indexing fa9266b
- [x] Task: Verify cache hit rate improvement using `perf stat` or equivalent 49bcfcf
- [x] Task: Conductor - User Manual Verification 'Phase 2: Spatial Sorting (Cache Locality)' (Protocol in workflow.md) bf687dd

## Phase 3: Buffer Management & Allocation Optimization [checkpoint: 77c2ad0]
- [x] Task: Refactor `par_search_optimized` to use a more compact storage for intermediate neighbor counts cbbb8ca
- [x] Task: Explore using `smallvec` or in-place sorting to reduce peak heap usage during the "Fill" pass ee573cf
- [x] Task: Minimize Python/Rust conversion overhead by optimizing the dictionary construction in `lib.rs` ee573cf
- [x] Task: Conductor - User Manual Verification 'Phase 3: Buffer Management & Allocation Optimization' (Protocol in workflow.md) 77c2ad0

## Phase 4: Parallelization Tuning [checkpoint: 9dcfd95]
- [x] Task: Implement adaptive chunk sizing for Rayon based on system size and CPU count 3d56f66
- [x] Task: Optimize the "Pass 1" counting phase to be more work-stealing friendly 3d56f66
- [x] Task: Benchmark scaling on 1, 2, 4, 8, 16, 20+ threads to find the saturation point 3d56f66
- [x] Task: Conductor - User Manual Verification 'Phase 4: Parallelization Tuning' (Protocol in workflow.md) 9dcfd95

## Phase 5: Final Validation [checkpoint: ffbfb49]
- [x] Task: Run full `proptest` suite with high iteration count (1000+) to ensure robustness c485462
- [x] Task: Final comparison against `vesin` across all cutoffs (6.0, 14.0, 20.0) f77c87f
- [x] Task: Update `README.md` with new performance metrics and scaling charts 4328150
- [x] Task: Conductor - User Manual Verification 'Phase 5: Final Validation' (Protocol in workflow.md) ffbfb49
