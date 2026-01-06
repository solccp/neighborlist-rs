# Specification: Comprehensive Neighbor List Benchmarking

## Overview
This track aims to implement a rigorous, multi-faceted benchmark suite to evaluate `neighborlist-rs` against established and emerging neighbor list libraries. The goal is to quantify performance across various system sizes, batching strategies, and boundary conditions, providing a clear comparison for the atomistic machine learning community.

## Functional Requirements
- **Library Integration:** Implement benchmark wrappers for:
    - `neighborlist-rs` (Target)
    - `ase.neighborlist` (Standard baseline)
    - `matscipy.neighbours` (CPU performance baseline)
    - `freud.locality.NeighborList` (High-performance baseline)
    - `vesin` (High-performance baseline)
    - `torch_cluster.radius_graph` (ML-focused baseline, CPU & GPU)
    - `torch_nl` (ML-focused baseline, CPU & GPU)
- **Benchmarking Dimensions:**
    - **System Size:** Small (<100 atoms) to evaluate overhead; Medium (1k-10k atoms) to evaluate production scaling.
    - **Boundary Conditions:** Full support for both Periodic Boundary Conditions (PBC) and non-PBC (isolated) systems.
    - **Density Regimes:** Test across different atomic densities (e.g., dilute gas vs. dense solid).
    - **Batched Processing:** Evaluate scaling for batch sizes: 1, 8, 32, 128, 512.
- **Hardware Evaluation:** 
    - CPU benchmarking for all libraries.
    - GPU benchmarking for `torch_cluster` and `torch_nl` to serve as reference points for hardware acceleration.
    - Target hardware for final results: DGX Spark.
- **Metrics:** Report Wall-clock time, throughput (systems per second), and scaling behavior.

## Non-Functional Requirements
- **Correctness:** Before benchmarking, verify that all libraries return equivalent neighbor counts for a given system to ensure a fair "apples-to-apples" comparison.
- **Reproducibility:** The benchmark script should be self-contained and document environment/hardware details.
- **Performance:** Use `target-cpu=native` for `neighborlist-rs` and ensure release builds are used for all compiled libraries.

## Acceptance Criteria
- Verified timing results for all libraries on DGX Spark.
- `README.md` updated with "Performance" section containing DGX Spark results (table and/or summary).
- Benchmark code and external dependencies for benchmarking are NOT committed to the main repository.

## Out of Scope
- Committing benchmarking scripts, wrappers, or environment configurations to the repository.
- Distributed (multi-node) benchmarking.
- Benchmarking of non-radius-based neighbor lists (e.g., K-Nearest Neighbors).
