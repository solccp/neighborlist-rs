# Track Spec: Holistic Codebase Optimization and Large-Scale Refactor

## Overview
Optimize the `neighborlist-rs` codebase for maximum performance and minimum memory footprint, with a specific focus on high-efficiency scaling for large atomistic systems (10k+ atoms). This track will refactor the core search engine and data structures to prioritize cache locality and parallel throughput.

## Functional Requirements
- **Spatial Sorting:** Implement atom reordering (e.g., via Z-order/Morton curve or Hilbert curve) during the `CellList::build` phase. This ensures that atoms close in space are close in memory, maximizing cache hits during the distance check.
- **Buffer Reuse:** Refactor the Python/Rust boundary to allow for reusing output buffers where possible, or further minimizing intermediate allocations during the two-pass search.
- **Advanced Parallelization:** Refine the Rayon strategy to use more granular work-partitioning, ensuring all 20+ CPU cores are saturated even when bin counts are relatively low compared to thread counts.

## Non-Functional Requirements
- **Performance Target:** Achieve >20% speedup on systems with 10k+ atoms compared to the current optimized baseline.
- **Memory Efficiency:** Reduce peak memory usage during construction by replacing temporary `Vec` collections with more compact structures or in-place transformations.
- **Correctness:** Maintain 100% agreement with the `vesin` reference and existing `proptest` invariants.

## Acceptance Criteria
1. `tests/comprehensive_benchmark.py` shows improved scaling and raw speed for 10k+ atom cases.
2. Memory profiling shows a reduction in peak heap allocation.
3. All existing Rust and Python tests pass.
