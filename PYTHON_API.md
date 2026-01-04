# Python API Documentation: neighborlist-rs

`neighborlist_rs` is a high-performance Rust library for neighbor list construction, designed for molecular dynamics and machine learning interatomic potentials (MLIPs).

## Table of Contents
1. [Core Conventions](#core-conventions)
2. [Single-System API](#single-system-api)
   - [Basic Search](#basic-neighbor-search)
   - [Isolated Systems (Non-PBC)](#isolated-systems-non-pbc)
   - [Multi-Cutoff Search](#multi-cutoff-search)
3. [Batched API](#batched-api)
   - [Standard Batch](#standard-batched-search)
   - [Multi-Cutoff Batch](#multi-cutoff-batched-search)
4. [Utility Classes & Methods](#utility-classes--methods)
5. [Advanced Tuning](#performance-tuning)

---

## Core Conventions

### Cell Matrix (H-matrix)
The cell matrix $H$ is defined such that the columns are the lattice vectors $\mathbf{a}, \mathbf{b}, \mathbf{c}$:
$$H = [\mathbf{a} | \mathbf{b} | \mathbf{c}] = \begin{bmatrix} a_x & b_x & c_x \\ a_y & b_y & c_y \\ a_z & b_z & c_z \end{bmatrix}$$
*Note: This is the transpose of the convention used by ASE.*

### Result Schema
All neighbor list results return a dictionary with a `"local"` key containing:
- `edge_i`: `(M,)` uint64 array of source atom indices.
- `edge_j`: `(M,)` uint64 array of target atom indices.
- `shift`: `(M, 3)` int32 array of periodic shifts.

### Reconstructing Vectors
To compute the displacement vector $\mathbf{r}_{ij}$ between atoms $i$ and $j$ considering PBC:
$$\mathbf{r}_{ij} = \mathbf{pos}_j + (S_{ij} \cdot H) - \mathbf{pos}_i$$
Where $S_{ij}$ is the shift vector for that edge.

---

## Single-System API

### Basic Neighbor Search
Computes pairs within a single cutoff for a periodic system.

```python
import neighborlist_rs
import numpy as np

# Define 10x10x10 cubic cell
cell = neighborlist_rs.PyCell([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
positions = np.random.rand(100, 3) * 10.0

result = neighborlist_rs.build_neighborlists(cell, positions, cutoff=5.0)
edges = result["local"]
print(f"Found {len(edges['edge_i'])} pairs")
```

### Isolated Systems (Non-PBC)
If you pass `None` as the cell, the library automatically infers a safe bounding box for the system.

```python
# Pass None for isolated systems
result = neighborlist_rs.build_neighborlists(None, positions, cutoff=5.0)
```

### ASE Integration
Directly compute neighbor lists from an ASE `Atoms` object. This handles extracting positions, cell, and PBC flags automatically.

#### Single Cutoff
```python
from ase.build import bulk
import neighborlist_rs

atoms = bulk("Cu", "fcc", a=3.6) * (3, 3, 3)

# Automatically uses the cell if pbc=True, or infers box if pbc=False
result = neighborlist_rs.build_from_ase(atoms, cutoff=5.0)
edges = result["local"]
```

#### Multiple Cutoffs
```python
cutoffs = [5.0, 10.0]
# Highly efficient single-pass search for multiple cutoffs
results = neighborlist_rs.build_multi_from_ase(atoms, cutoffs)

for i, rc in enumerate(cutoffs):
    edge_index = results[i]["edge_index"]
    print(f"Cutoff {rc}: {edge_index.shape[1]} pairs")
```

With labels:
```python
results = neighborlist_rs.build_multi_from_ase(
    atoms, cutoffs, labels=["short", "long"]
)
short_edges = results["short"]["edge_index"]
```

### Multi-Cutoff Search
Compute multiple neighbor lists (e.g., Short-range vs Electrostatics) in a **single pass**. This is much faster than calling the library multiple times.

```python
cutoffs = [6.0, 14.0, 20.0]
results = neighborlist_rs.build_neighborlists_multi(cell, positions, cutoffs)

# Access results by index (order of cutoffs)
for i, rc in enumerate(cutoffs):
    edge_index = results[i]["edge_index"]  # Shape (2, N)
    print(f"Cutoff {rc}A: {edge_index.shape[1]} edges")
```

#### Disjoint Shells
By default, neighbor lists are cumulative (e.g., $r < 14.0$ includes $r < 6.0$).
To get exclusive shells (e.g., $6.0 \le r < 14.0$), pass `disjoint=True`:

```python
results = neighborlist_rs.build_neighborlists_multi(
    cell, positions, cutoffs, disjoint=True
)
# results[0]: r < 6.0
# results[1]: 6.0 <= r < 14.0
# results[2]: 14.0 <= r < 20.0
```

You can also pass `labels` to get a dictionary keyed by string labels instead of indices:

```python
labels = ["short", "medium", "long"]
results = neighborlist_rs.build_neighborlists_multi(cell, positions, cutoffs, labels=labels)

short_edges = results["short"]["edge_index"]
```

---

## Batched API

The Batched API allows processing a large number of systems (e.g., a training batch) in parallel across CPU cores. This minimizes the overhead of Python loops and the GIL.

### Standard Batched Search
Input positions should be a single flattened array, with a `batch` array indicating system ownership.

```python
# positions: (N_total, 3), batch: (N_total,)
# cells: (B, 3, 3) where B is number of systems
batch_res = neighborlist_rs.build_neighborlists_batch(
    positions, batch, cells=cells, cutoff=5.0
)

all_edges = batch_res["local"]
```

### Multi-Cutoff Batched Search
The most efficient way to generate data for MLIPs. Processes an entire batch and multiple cutoffs in one call.

```python
cutoffs = [6.0, 14.0]
results = neighborlist_rs.build_neighborlists_batch_multi(
    positions, batch, cells=cells, cutoffs=cutoffs
)

# Returns: { 0: {"edge_index": (2, N), "shift": (N, 3)}, ... }
mlp_edges = results[0]["edge_index"]
dispersion_edges = results[1]["edge_index"]
```

You can also use `disjoint=True` here to generate exclusive shells for efficient memory usage.

With labels:
```python
labels = ["mlp", "dispersion"]
results = neighborlist_rs.build_neighborlists_batch_multi(
    positions, batch, cells=cells, cutoffs=cutoffs, labels=labels
)

# Returns: { "mlp": {"edge_index": ...}, "dispersion": {"edge_index": ...} }
mlp_edges = results["mlp"]["edge_index"]
```

*Note: In batched mode, if `cells[i]` is a zero-matrix, system `i` is treated as non-PBC with auto-box inference.*

---

## Utility Classes & Methods

### `PyCell`
Represents the simulation box.
- `PyCell(h_matrix)`: Create from 3x3 list/array.
- `wrap(pos)`: Wrap a 3D point into the primary cell.

### Global Configuration
```python
# Set number of Rayon threads (default is total logical cores)
neighborlist_rs.set_num_threads(8)

# Get current thread count
n = neighborlist_rs.get_num_threads()

# Initialize Rust-side logging (info, debug, trace)
neighborlist_rs.init_logging("info")
```

---

## Performance Tuning

### 1. Brute Force Threshold
For systems with $N < 500$ atoms, the library automatically uses a highly optimized brute-force kernel which avoids the overhead of cell-list construction.

### 2. Parallelism Threshold
For tiny systems ($N < 20$), internal parallelism is disabled to avoid thread synchronization overhead. In batched mode, parallelism happens *across* systems, so even 1-atom systems are processed efficiently in parallel.

### 3. Compilation
For maximum performance, ensure the library is compiled with native hardware support:
```bash
# Handled automatically if using the provided .cargo/config.toml
maturin develop --release
```

