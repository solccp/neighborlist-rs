# Track: Codebase Robustness and Ergonomics

## Overview
This track aims to improve the codebase across three main dimensions:
1. **Correctness & Safety**: Eliminating Undefined Behavior (UB) in parallel kernels.
2. **Performance**: Implementing zero-copy position handling and exploring SIMD optimizations.
3. **Ergonomics**: Improving the Python API with better debugging tools and type safety.

## Phase 1: Correctness & Safety
- [x] **Fix Undefined Behavior in Parallel Kernels**: Refactor `src/search.rs` to avoid overlapping mutable slices. Use raw pointers or `Sync` safe wrappers.
- [x] **Audit Safety Blocks**: Ensure all `unsafe` blocks are documented with `SAFETY` comments and follow Rust's safety guidelines.

## Phase 2: Performance Optimizations
- [ ] **Zero-Copy Positions**: Use `bytemuck` to cast NumPy position arrays directly to `&[Vector3<f64>]` or `&[[f64; 3]]` to eliminate the $O(N)$ copy in `src/lib.rs`.
- [ ] **Inner-Loop SIMD**: Implement an explicit SIMD kernel for the distance-squared calculation using the `wide` crate.
- [ ] **Benchmarking Verification**: Verify speedups using `benchmarks/scaling.py`.

## Phase 3: Ergonomics and Type Safety
- [ ] **Python Repr**: Add `__repr__` to `PyCell` for informative Python debugging.
- [ ] **ASE Factory Method**: Implement `PyCell.from_ase(atoms)` to streamline cell creation from ASE objects.
- [ ] **Python Type Stubs**: Generate a `neighborlist_rs.pyi` file for IDE autocompletion and static type checking.
