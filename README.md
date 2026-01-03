# neighborlist-rs

High-performance neighborlist construction for atomistic systems in Rust with Python bindings.

## Features
- **High-performance Cell Lists:** O(N) scaling for large systems.
- **Robust PBC support:** Minimum image convention for orthorhombic and triclinic cells.
- **Parallel Search:** Multi-core neighbor search using Rayon.
- **Python bindings:** Seamless integration with Python via PyO3.
- **Zero-copy NumPy integration:** Efficient data exchange with the scientific Python stack.

## Installation

### From Source
```bash
pip install .
```

## Usage (Python)

```python
import neighborlist_rs
import numpy as np

# 1. Define the cell matrix (column vectors a, b, c)
h = [
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0]
]
cell = neighborlist_rs.PyCell(h)

# 2. Define atom positions (N, 3)
positions = np.array([
    [1.0, 1.0, 1.0],
    [1.5, 1.0, 1.0],
    [9.5, 1.0, 1.0]
], dtype=np.float64)

# 3. Build neighbor lists
cutoff = 2.0
result = neighborlist_rs.build_neighborlists(cell, positions, cutoff, parallel=True)

# 4. Access indices and shifts
local = result["local"]
edge_i = local["edge_i"] # [0, 0]
edge_j = local["edge_j"] # [1, 2]
shifts = local["shift"]  # [[0, 0, 0], [-1, 0, 0]]
```

## Verification
Run Rust tests:
```bash
cargo test
```

Run Python tests:
```bash
pytest tests/test_basic.py
```