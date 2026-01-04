# neighborlist-rs

High-performance neighborlist construction for atomistic systems in Rust with Python bindings.

## Project Overview

**Purpose:** This library generates neighbor lists for molecular dynamics (MD) and geometric deep learning (GNNs). It is optimized for speed and correctness, handling Periodic Boundary Conditions (PBC) for both orthogonal and triclinic cells.
**Primary Consumers:** Python/PyTorch-based MLIP (Machine Learning Interatomic Potentials) training and inference pipelines (e.g., `jmolnet`).

## Tech Stack

*   **Core Logic:** Rust (stable)
*   **Python Bindings:** [PyO3](https://github.com/PyO3/pyo3)
*   **Build System:** [Maturin](https://github.com/PyO3/maturin)
*   **Parallelism:** [Rayon](https://github.com/rayon-rs/rayon)
*   **Linear Algebra:** [Nalgebra](https://github.com/dimforge/nalgebra)
*   **Testing:** `cargo test` (Rust), `pytest` (Python), `proptest` (Property-based testing)

## Directory Structure

*   `src/`: Rust source code.
    *   `lib.rs`: PyO3 module definition and Python API entry points.
    *   `batch.rs`: Logic for processing batches of systems.
    *   `single.rs`: Logic for processing single systems and auto-box inference.
    *   `search.rs`: Core cell-list algorithm, parallel kernels, and brute-force fallbacks.
    *   `cell.rs`: Unit cell handling, MIC logic, and Cartesian/Fractional transformations.
*   `tests/`: Python integration tests.
    *   `test_basic.py`: Standard functional tests including multi-cutoff verification.
    *   `test_batch.py`: Rigorous tests for batched processing and mixed PBC/non-PBC batches.
*   `benchmarks/`: Performance benchmarking suite.
*   `conductor/`: Project management, track plans, and technical specifications.
*   `.cargo/config.toml`: Local compilation flags (e.g., `target-cpu=native`).

## Test Coverage

*   **Core Logic:** >95% coverage for `src/search.rs` and `src/cell.rs`.
*   **Batch/Single Logic:** >85% coverage for `src/batch.rs` and `src/single.rs`.
*   **Total Project:** ~70% total line coverage (including untested PyO3 boilerplate in `src/lib.rs`).
*   **Validation:** All features are additionally validated via comprehensive Python integration tests (`pytest tests/`).

## Development Workflow

### Prerequisites
*   Rust toolchain (cargo, rustc)
*   Python 3.8+
*   `maturin` build tool (`pip install maturin`)

### Building and Testing

**1. Rust Core:**
```bash
# Build
cargo build

# Run Unit & Property Tests
cargo test
```

**2. Python Bindings:**
To develop with the Python bindings active in your current environment:
```bash
# Fast development build (installs directly to current venv)
maturin develop --release
```
*Note: Using `--release` is mandatory for performance, especially for large systems or benchmarks.*

**3. Python Tests:**
```bash
# Run standard tests
pytest tests/
```

## Key Concepts

*   **Cell Lists:** The algorithm partitions the simulation box into bins (cells) larger than the cutoff radius to achieve O(N) complexity.
*   **Multi-Cutoff:** Generate multiple neighbor lists (e.g., for short-range, dispersion, and long-range interactions) in a **single pass** over the structure.
*   **Batched Processing:** Process a batch of many systems (e.g., 128 molecules) simultaneously in parallel using Rayon, reducing Python/GIL overhead.
*   **Auto-Box Inference:** Passing `None` for the cell automatically infers a safe bounding box for isolated (non-PBC) systems.
*   **Shifts:** For PBC, the output includes integer vectors (`shift`) representing the number of cell vectors added to the neighbor's position to bring it within the cutoff distance relative to the central atom.
    *   `r_j_image = r_j + shift * cell_matrix`

## Performance Optimizations

*   **Hardware Acceleration:** Compiled with `target-cpu=native` for optimal instruction set usage.
*   **Adaptive Strategy:** 
    *   **Brute Force Fallback:** Automatically uses O(N²) search for systems with $N < 500$ atoms (where it outperforms cell-lists).
    *   **Serial Fallback:** Automatically disables parallel overhead for tiny systems ($N < 20$ atoms).
*   **Z-Order Sorting:** Particles are sorted by Z-order (Morton curve) to maximize cache hits during cell traversal.

## Conventions

*   **Zero-Copy:** Returns NumPy arrays that take ownership of Rust-allocated memory or write directly into them to minimize copying.
*   **Matrix Layout:** Cell matrices are column-major vectors (a, b, c).
*   **Error Handling:** Invalid inputs (e.g., non-3x3 matrices, non-invertible cells) return `PyValueError`.