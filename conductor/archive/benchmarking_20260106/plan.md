# Plan: Comprehensive Neighbor List Benchmarking (External Execution)

## Phase 1: Temporary Environment and Verification
- [x] Task: Set up a temporary environment on DGX Spark with all target libraries (`neighborlist-rs`, `ase`, `matscipy`, `freud`, `vesin`, `torch_nl`, `torch_cluster`).
- [x] Task: Create a temporary verification script to ensure all libraries return identical neighbor lists for a set of test systems.
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Benchmark Execution (Ephemeral)
- [x] Task: Develop/Run ephemeral benchmarking logic to compare all libraries across:
    - System sizes (Small, Medium).
    - Boundary conditions (PBC, non-PBC).
    - Batch sizes (1, 8, 32, 128, 512).
- [x] Task: Collect raw timing data and throughput metrics for both CPU and GPU (for `torch_nl`/`torch_cluster`).
- [x] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Documentation of Results
- [x] Task: Format the collected data into a clean table and summary.
- [x] Task: Update the project `README.md` with the "Performance" section containing these results.
- [x] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

