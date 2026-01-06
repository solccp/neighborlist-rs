# Technology Stack

## Core Language & Runtime
- **Rust (Stable):** Primary language for the core neighborlist logic, chosen for performance, memory safety, and excellent concurrency primitives.

## Parallelism & Concurrency
- **Rayon:** Used for data-parallel neighborlist construction. It provides an efficient, work-stealing scheduler to distribute the cell-list traversal across multiple CPU cores.

## Python Integration
- **PyO3:** High-level Rust bindings for Python. Enables seamless conversion between Rust types and Python objects.
- **Maturin:** Build system for building and publishing the Rust-based Python package.
- **NumPy (via rust-numpy):** Essential for zero-copy data exchange. Neighborlist arrays (indices and shifts) will be exposed as NumPy arrays to the Python layer.

## Mathematical Libraries
- **nalgebra:** For efficient 3x3 matrix operations (cell matrices) and 3D vector math (Cartesian/fractional conversions).
- **bytemuck:** For safe zero-copy casting between numeric types and SIMD vectors.
- **wide:** For explicit SIMD kernels to accelerate the inner loops of the neighbor search.

## Testing Frameworks
- **Built-in Rust Test Runner:** For unit and integration testing.
- **proptest:** For property-based testing to verify geometric invariants.
- **pytest:** For testing the Python bindings and ensuring integration with the Python scientific stack works as expected.

## Development Tools
-   **clippy:** For linting and ensuring idiomatic Rust code.
- **rustfmt:** For consistent code formatting.
- **tracing:** For instrumentation and performance monitoring.
- **dhat/perf:** For memory and CPU profiling during optimization cycles.
