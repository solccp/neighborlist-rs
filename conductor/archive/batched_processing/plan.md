# Track: Batched Neighbor List Processing

**Status:** Planned
**Owner:** Sol
**Created:** 2026-01-04

## Context
`jmolnet` and other ML potentials typically process molecules in batches (e.g., 32-128 structures per batch). Currently, `neighborlist-rs` accepts a single structure. Calling it inside a Python loop for a batch of small molecules is inefficient due to:
1.  **GIL Overhead:** Repeated Python-Rust boundary crossings.
2.  **Under-utilization:** Small systems (N < 20) are processed serially to avoid overhead. A 64-core CPU stays mostly idle when processing one small molecule at a time.

## Objective
Implement a `build_neighborlists_batch` function in Rust that processes a batch of structures in parallel, fully utilizing the CPU even for small systems.

## Requirements

### 1. API Design
The API should align with PyTorch Geometric (PyG) conventions where possible for easy integration.

**Input:**
*   `positions`: $(N_{total}, 3)$ numpy array (concatenated positions of all atoms).
*   `batch`: $(N_{total},)$ integer numpy array (0-based index of the system each atom belongs to).
*   `cells`: $(B, 3, 3)$ numpy array (optional). If provided, `cells[i]` corresponds to system `i`. If `None`, all systems are treated as non-PBC (or auto-inferred box).
*   `cutoff`: `float`.

**Output:**
A dictionary containing:
*   `edge_i`: $(M,)$ Global indices of source atoms.
*   `edge_j`: $(M,)$ Global indices of target atoms.
*   `shift`: $(M, 3)$ Periodic shifts.
*   `batch_edge`: $(M,)$ System index for each edge (optional, but good for debugging/verification).

### 2. Implementation Strategy
*   **Data Splitting:**
    *   Parse the `batch` array to determine the slice ranges for `positions` for each system.
    *   Construct `Vector<Slice>` or similar lightweight descriptors.
*   **Parallel Execution:**
    *   Use `rayon::par_iter` to iterate over the *systems* in the batch.
    *   For each system, call the existing `CellList` or `brute_force` logic (which will likely default to serial for small molecules).
    *   This ensures we process $B$ molecules in parallel, achieving near-linear scaling with thread count.
*   **Result Assembly:**
    *   Collect results (edges) from each parallel task.
    *   Apply the necessary `offset` (start index of atoms in the global array) to the local `edge_i` and `edge_j` indices.
    *   Concatenate all results into the final output arrays.

## Phases

### Phase 1: API & Data Structures
*   [x] Define the `build_neighborlists_batch` function signature in `lib.rs`. [0ac2be2]
*   [x] Implement helper logic to parse `batch` array and split `positions` into per-system slices. [0ac2be2]

## Phase 2: Parallel Core
*   [x] Implement the `par_iter` loop over systems. [0ac2be2]
*   [x] Integrate with existing `search` module logic. [0ac2be2]
*   [x] Handle `cells` (PBC vs non-PBC) correctly for each system. [0ac2be2]

## Phase 3: Result Aggregation
*   [x] Efficiently merge vectors from threads. [0ac2be2]
*   [x] Correctly offset global indices. [0ac2be2]

## Phase 4: Benchmarking & Verification
*   [x] Create a benchmark comparing: [0ac2be2]
    *   Loop in Python (current baseline).
    *   `build_neighborlists_batch` (new implementation).
*   [x] Verify correctness against serial execution. [0ac2be2]

## Phase 5: Multi-Cutoff Batch Support
*   [x] Define the `build_neighborlists_batch_multi` function in `lib.rs`. [125e0b0]
*   [x] Implement fused parallel search over systems for multiple cutoffs. [125e0b0]
*   [x] Implement multi-cutoff aggregation (mapping cutoffs to concatenated results). [125e0b0]
*   [x] Add integration tests for batched multi-cutoff. [125e0b0]
*   [x] Verify correctness against `build_neighborlists_multi`. [125e0b0]

## Risks & Mitigations
*   **Memory Usage:** Creating large intermediate vectors for every system might spike memory.
    *   *Mitigation:* Pre-calculate edge counts if possible, or use standard `Vec` and collect. Since we target small/medium systems, memory shouldn't be the bottleneck compared to the graph size itself.
*   **Load Balancing:** Some systems might be much larger than others.
    *   *Mitigation:* Rayon's work-stealing handles this naturally.

## Success Metrics
*   **Speedup:** >10x speedup for batches of small molecules (e.g., 128 water molecules) compared to serial Python loop on a high-core count machine.
