# Specification: Codebase Robustness and Ergonomics

## 1. Objectives
This track aims to mature the `neighborlist-rs` codebase by addressing technical debt, improving safety, optimizing performance, and enhancing the Python API ergonomics. The goal is to make the library production-ready for diverse atomistic simulation workflows.

## 2. Scope

### 2.1 Correctness & Safety
- **Parallel Kernels:** Eliminate Undefined Behavior (UB) caused by overlapping mutable slices in `src/search.rs` when using Rayon.
- **Safety Audits:** Systematically document all `unsafe` blocks with `SAFETY` comments to justify their usage.

### 2.2 Performance
- **Zero-Copy Positions:** Avoid the $O(N)$ allocation and copy when passing position arrays from Python to Rust. Directly access NumPy memory.
- **Allocation Removal:** Eliminate repeated vector allocations inside the neighbor search hot loops (count and fill passes).
- **SIMD Acceleration:** Implement explicit SIMD kernels (using `wide` or intrinsics) for the core distance-squared calculations to maximize throughput on modern CPUs.

### 2.3 Robustness & Validation
- **Input Validation:**
    - Enforce `(N, 3)` shape for position arrays.
    - Validate that `cutoff` values are finite and positive.
    - Ensure batch IDs are sorted or handle unsorted inputs gracefully.
- **Testing:**
    - Add unit tests for invalid inputs.
    - Enable and fix ASE integration tests, specifically for Periodic Boundary Conditions (PBC) and cell transposes.

### 2.4 Ergonomics & Features
- **Python API:**
    - Add `__repr__` to `PyCell` for better debugging.
    - Create a factory method `PyCell.from_ase(atoms)` to simplify integration with the ASE library.
    - Generate `neighborlist_rs.pyi` type stubs for better IDE support.
- **Mixed PBC:** Support systems with mixed periodic and non-periodic boundary conditions (e.g., slabs).

## 3. Success Criteria
- No Undefined Behavior reported by Miri (where applicable) or safety audits.
- Significant reduction in memory allocations during neighbor search.
- Performance benchmarks (`benchmarks/scaling.py`) show speedups or neutral impact.
- All new validation tests pass; existing tests pass.
- ASE integration works correctly for PBC systems.
- Python type stubs are generated and accurate.
