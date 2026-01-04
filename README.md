# neighborlist-rs

High-performance neighborlist construction for atomistic systems in Rust with Python bindings.

## Features
- **High-performance Cell Lists:** O(N) scaling for large systems.
- **Batched Processing:** Parallelized neighbor list construction across multiple systems.
- **Multi-Cutoff pass:** Generate multiple neighbor lists (e.g., 6Å, 14Å, 20Å) in a single optimized pass.
- **Auto-Box Inference:** Automatic bounding box calculation for isolated molecules.
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

Refer to [PYTHON_API.md](./PYTHON_API.md) for full documentation and advanced usage examples.

### Quick Start
```python
import neighborlist_rs
import numpy as np

# Single system search
result = neighborlist_rs.build_neighborlists(None, positions, cutoff=5.0)
edges = result["local"]["edge_i"]
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