# Product Guidelines

## Documentation Style: Technical & Precise
This project prioritizes mathematical correctness and performance. Documentation should reflect this by:
- **Formal Definitions:** Clearly define all physical and mathematical conventions (e.g., cell matrix layout, shift vector signs, coordinate systems).
- **Performance Benchmarking:** Include complexity analysis (O(N)) and provide benchmarks for common system sizes (1k, 10k, 100k atoms).
- **API Reference:** Every public function and struct must have clear docstrings with usage examples, specifically showing how to reconstruct displacement vectors.
- **Visual Aids:** Use diagrams or clear descriptions to explain Periodic Boundary Condition (PBC) wrapping and triclinic cell logic.

## Design Principles
1.  **Safety Over Speed:** Rust's memory safety must be leveraged. Use `Result` and `Option` instead of panics for expected edge cases.
2.  **Deterministic Results:** Output must be reproducible across different runs unless non-determinism is explicitly requested for performance.
3.  **Modular Multi-Cutoff:** The internal architecture must allow adding new interaction types (cutoffs) without re-scanning neighbors unnecessarily.
4.  **Zero-Copy Efficiency:** Python bindings should minimize data duplication to reduce overhead during large-scale GNN training.

## Code Quality Standards
- **Test Coverage:** Every mathematical transformation (Fractional <-> Cartesian, Shift logic) must have 100% unit test coverage.
- **Invariants:** Use property-based testing (e.g., `proptest`) to verify that the neighborlist logic holds for arbitrary random systems.
- **Idiomatic Rust:** Follow `clippy` and standard Rust formatting to maintain a professional and readable codebase.
