# Track: Codebase Robustness and Ergonomics

## Overview
This track aims to improve the codebase across three main dimensions:
1. **Correctness & Safety**: Eliminating Undefined Behavior (UB) in parallel kernels.
2. **Performance**: Implementing zero-copy position handling, reducing allocations, and exploring SIMD optimizations.
3. **Robustness**: Adding rigorous input validation and expanding test coverage.
4. **Ergonomics**: Improving the Python API and supporting broader use-cases (Mixed PBC).

## Phase 1: Correctness & Safety
- [x] **Fix Undefined Behavior in Parallel Kernels**: Refactor `src/search.rs` to avoid overlapping mutable slices. Use raw pointers or `Sync` safe wrappers.
- [x] **Audit Safety Blocks**: Ensure all `unsafe` blocks are documented with `SAFETY` comments and follow Rust's safety guidelines.

## Phase 2: Performance Optimizations [checkpoint: 09bc62d]
- [x] **Zero-Copy Positions**: Use `bytemuck` to cast NumPy position arrays directly to `&[Vector3<f64>]` or `&[[f64; 3]]` to eliminate the $O(N)$ copy in `src/lib.rs`. [5d2ad5b]
- [x] **Remove Hot-Path Allocations**: Precompute offset tuples in `CellList` or use stack-bounded `ArrayVec` in `src/search.rs` to eliminate repeated `Vec` allocations during neighbor search. [5c5205d]
- [x] **Inner-Loop SIMD**: Implement an explicit SIMD kernel for the distance-squared calculation using the `wide` crate. [Pre-existing]
- [x] **Benchmarking Verification**: Verify speedups using `benchmarks/scaling.py`. [9ff42f6]

## Phase 3: Validation & Robustness [checkpoint: dec07b0]
- [~] **Input Validation**:
    - [x] Enforce `(N, 3)` shape for positions in batch bindings (`src/lib.rs`). [5d2ad5b]
    - [x] Validate `cutoff` is finite and positive. [3d34f1b]
    - [x] Validate batch IDs are sorted/monotonic or handle unsorted input. [1ec9a79]
- [x] **Edge Cases & Testing**:
    - [x] Add unit tests for invalid cutoffs and unsorted batch IDs. [3d34f1b, 1ec9a79]
    - [x] **ASE PBC Validation**: Enable and fix ASE PBC path tests to ensure `build_from_ase` handles transposes correctly. [43426 - implicit in passing existing logic with new tests]
    - [~] **Dtype Alignment**: Align `edge_index` documentation with implementation (decide on `int64` vs `uint64`).

## Phase 4: Ergonomics & Features
- [x] **Python Repr**: Add `__repr__` to `PyCell` for informative Python debugging. [d3ecf55]
- [~] **ASE Integration**:
    - [x] `PyCell.from_ase(atoms)`: Implement factory method. [00d5d12]
    - [x] Support Mixed PBC in ASE conversion. [59590]
- [x] Mixed PBC Support: Extend `Cell` and `CellList` to handle periodic and non-periodic dimensions independently. [59590]
- [~] **Python Type Stubs**: Generate a `neighborlist_rs.pyi` file for IDE autocompletion and static type checking.
