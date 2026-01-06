# neighborlist-rs

High-performance neighborlist construction for atomistic systems in Rust with Python bindings.

[![CI](https://github.com/solccp/neighborlist-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/solccp/neighborlist-rs/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/neighborlist-rs.svg)](https://pypi.org/project/neighborlist-rs/)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

## Features
- **High-performance Cell Lists:** O(N) scaling for large systems.
- **Batched Processing:** Parallelized neighbor list construction across multiple systems (ideal for GNN training).
- **Multi-Cutoff pass:** Generate multiple neighbor lists (e.g., short-range GNN, long-range dispersion) in a single optimized pass.
- **SIMD Acceleration:** Explicit `wide` SIMD kernels for small systems and brute-force fallbacks.
- **Auto-Box Inference:** Automatic bounding box calculation for isolated molecules.
- **Robust PBC support:** Minimum image convention for orthorhombic and triclinic cells.
- **ASE Integration:** Direct support for `ase.Atoms` objects.
- **Zero-copy NumPy integration:** Efficient data exchange via PyO3 and `rust-numpy`.

## Installation

### From PyPI
```bash
pip install neighborlist-rs
```

### From Source
```bash
pip install .
```

## Usage (Python)

Refer to [PYTHON_API.md](./PYTHON_API.md) for full documentation and advanced usage examples.

### Quick Start (with ASE)
```python
import neighborlist_rs
from ase.build import molecule

atoms = molecule("C2H5OH")
# Single pass search directly from ASE Atoms
result = neighborlist_rs.build_from_ase(atoms, cutoff=5.0)

edge_index = result["edge_index"]  # (2, M) array of atom pairs
shifts = result["shift"]           # (M, 3) periodic shift vectors
```

### Manual Search
```python
import neighborlist_rs
import numpy as np

positions = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
# None for cell means isolated system (non-PBC)
result = neighborlist_rs.build_neighborlists(None, positions, cutoff=5.0)
```

## Performance

`neighborlist-rs` is optimized for both massive systems and many tiny molecules.

### Key Optimizations
- **Spatial Sorting:** Uses Z-order (Morton) indexing to improve cache locality.
- **Dynamic Configuration:** Runtime-tunable thresholds for switching between SIMD Brute-Force and Cell Lists.
- **Two-Pass Search:** Minimizes heap allocations by pre-calculating neighbor counts.
- **Cross-Platform SIMD:** Optimized for both x86_64 (AVX2/FMA) and aarch64 (NEON).

### Micro-Benchmarks (Isolated Systems)
(Measured on arm64 Apple Silicon / Linux)

| N Atoms | Vesin (ms) | neighborlist-rs (ms) | Speedup |
|---------|------------|----------------------|---------|
| 50      | 0.035      | 0.007                | 5.0x    |
| 100     | 0.033      | 0.010                | 3.3x    |
| 1000    | 3.707      | 1.574                | 2.3x    |

## Verification
Run Rust tests:
```bash
cargo test
```

Run Python tests:
```bash
pytest tests/
```

## License
Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.
