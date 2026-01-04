# neighborlist-rs

High-performance neighborlist construction for atomistic systems in Rust with Python bindings.

## Project Overview

**Purpose:** This library generates neighbor lists for molecular dynamics (MD) and geometric deep learning (GNNs). It is optimized for speed and correctness, handling Periodic Boundary Conditions (PBC) for both orthogonal and triclinic cells.
**Primary Consumers:** Python/PyTorch-based MLIP (Machine Learning Interatomic Potentials) training and inference pipelines.

## Tech Stack

*   **Core Logic:** Rust (stable)
*   **Python Bindings:** [PyO3](https://github.com/PyO3/pyo3)
*   **Build System:** [Maturin](https://github.com/PyO3/maturin)
*   **Parallelism:** [Rayon](https://github.com/rayon-rs/rayon)
*   **Linear Algebra:** [Nalgebra](https://github.com/dimforge/nalgebra)
*   **Testing:** `cargo test` (Rust), `pytest` (Python), `proptest` (Property-based testing)

## Directory Structure

*   `src/`: Rust source code.
    *   `lib.rs`: PyO3 module definition and Python interface logic.
    *   `search.rs`: Core cell-list algorithm and neighbor search implementation.
    *   `cell.rs`: Unit cell handling and wrapping logic.
*   `tests/`: Python integration tests.
    *   `test_basic.py`: Standard functional tests.
    *   `comprehensive_benchmark.py`: Performance benchmarking script.
*   `conductor/`: Project management and design documents.
    *   `spec.md`: Technical specification and requirements.
    *   `product.md`: High-level product goals.
*   `proptest-regressions/`: Saved failing cases for property-based tests.

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
*Note: Using `--release` is highly recommended for neighbor list performance, as debug builds are significantly slower.*

**3. Python Tests:**
```bash
# Run standard tests
pytest tests/

# Run benchmarks
python tests/comprehensive_benchmark.py
```

## Key Concepts

*   **Cell Lists:** The algorithm partitions the simulation box into bins (cells) larger than the cutoff radius to achieve O(N) complexity.
*   **PBC (Periodic Boundary Conditions):** Atoms interact across unit cell boundaries. The library computes the "minimum image" convention.
*   **Shifts:** For PBC, the output includes integer vectors (`shift`) representing the number of cell vectors added to the neighbor's position to bring it within the cutoff distance relative to the central atom.
    *   `r_j_image = r_j + shift * cell_matrix`

## Conventions

*   **Zero-Copy:** The Python bindings are designed to minimize data copying. `build_neighborlists` returns NumPy arrays that take ownership of Rust-allocated memory or write directly into them where possible.
*   **Matrix Layout:** Cell matrices are column-major vectors (a, b, c).
*   **Error Handling:** Invalid inputs (e.g., non-3x3 matrices) return `PyValueError`.
