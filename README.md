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

## Performance

`neighborlist-rs` is optimized for high-performance scaling on multi-core systems.

### Key Optimizations
- **Spatial Sorting:** Uses Z-order (Morton) indexing to improve cache locality.
- **Adaptive Parallelization:** Dynamically adjusts work chunk sizes based on system size and CPU count.
- **Two-Pass Search:** Minimizes heap allocations by pre-calculating neighbor counts.

### Scaling Benchmarks
(System: 20,000 atoms, Ethanol PBC, Cutoff 6.0 Å, 20-core CPU)

| Threads | Time (ms) | Speedup |
|---------|-----------|---------|
| 1       | 57.71     | 1.0x    |
| 8       | 14.84     | 3.9x    |
| 20      | 11.23*    | 5.1x    |

*\*Measured via `benchmarks/scaling.py` on 50k atom systems.*

## Verification
Run Rust tests:
```bash
cargo test
```

Run Python tests:
```bash
pytest tests/test_basic.py
```