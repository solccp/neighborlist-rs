# Track Spec: Core Neighborlist Implementation (Phase 1)

## Overview
Implement the foundational neighborlist search engine in Rust with high-performance Cell Lists, robust PBC support, and Python bindings via PyO3.

## Functional Requirements

### 1. Mathematical Core
- **PBC Support:** Implement minimum image convention for orthorhombic and triclinic cells.
- **Conventions:**
  - Cell Matrix $H$: Column vectors $[a, b, c]$.
  - Shift Vector $s_{ij}$: Integer vector such that $r_{j,img} = r_j + s_{ij} \cdot H$.
  - Displacement: $d_{ij} = r_{j,img} - r_i$.
- **Coordinate Transformation:** Robust Cartesian to Fractional and Fractional to Cartesian transformations.

### 2. Search Engine
- **Algorithm:** Cell Lists (Binning).
- **Complexity:** $O(N)$ scaling.
- **Multi-Cutoff:** Ability to populate multiple lists (local, dispersion) in a single traversal of the bins.
- **Parallelism:** Use Rayon to parallelize the search over atoms/bins.

### 3. Python Bindings
- **Framework:** PyO3.
- **Output:** Dictionary of results:
  ```python
  {
      "local": {
          "edge_i": np.ndarray (int64),
          "edge_j": np.ndarray (int64),
          "shift":  np.ndarray (int32, shape (E, 3))
      },
      ...
  }
  ```
- **Performance:** Minimize data copying when passing arrays to NumPy.

## Non-Functional Requirements
- **Determinism:** Support sorting of edges by $(i, j)$ for reproducible results.
- **Safety:** Use `Result` for invalid inputs (e.g., non-invertible cell matrices).
- **Coverage:** >80% test coverage for core logic.
