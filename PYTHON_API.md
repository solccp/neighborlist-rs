# Python API Documentation: neighborlist-rs

`neighborlist_rs` is a high-performance Rust library for neighbor list construction, designed for molecular dynamics and machine learning interatomic potentials (MLIPs).

## Table of Contents
1. [Core Conventions](#core-conventions)
2. [ASE Integration](#ase-integration)
3. [Single-System API](#single-system-api)
   - [Basic Search](#basic-neighbor-search)
   - [Isolated Systems (Non-PBC)](#isolated-systems-non-pbc)
   - [Multi-Cutoff Search](#multi-cutoff-search)
4. [Batched API](#batched-api)
   - [Standard Batch](#standard-batched-search)
   - [Multi-Cutoff Batch](#multi-cutoff-batched-search)
5. [Utility Classes & Methods](#utility-classes--methods)
6. [Performance Tuning](#performance-tuning)

---

## Core Conventions

### Cell Matrix (H-matrix)
The cell matrix $H$ is defined such that the columns are the lattice vectors $\mathbf{a}, \mathbf{b}, \mathbf{c}$:
$$H = [\mathbf{a} | \mathbf{b} | \mathbf{c}] = \begin{bmatrix} a_x & b_x & c_x \\ a_y & b_y & c_y \\ a_z & b_z & c_z \end{bmatrix}$$
*Note: This is the transpose of the convention used by ASE.*

### Result Schema
All neighbor list results return a dictionary containing:
- `edge_index`: `(2, M)` int64 array of atom pairs.
- `shift`: `(M, 3)` int32 array of periodic shifts.

### Reconstructing Vectors
To compute the displacement vector $\mathbf{r}_{ij}$ between atoms $i$ and $j$ considering PBC:
$$\mathbf{r}_{ij} = \mathbf{pos}_j + (S_{ij} \cdot H) - \mathbf{pos}_i$$
Where $S_{ij}$ is the shift vector for that edge.

---

## ASE Integration

The simplest way to use `neighborlist-rs` if you already use the Atomic Simulation Environment (ASE).

```python
import neighborlist_rs
from ase.build import bulk

atoms = bulk("Si", "diamond", a=5.43)
# Single cutoff
res = neighborlist_rs.build_from_ase(atoms, cutoff=3.0)

# Multi cutoff
cutoffs = [3.0, 5.0]
res_multi = neighborlist_rs.build_multi_from_ase(atoms, cutoffs, labels=["short", "long"])
```

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
edge_index = result["edge_index"]
print(f"Found {edge_index.shape[1]} pairs")
```

### Isolated Systems (Non-PBC)
If `cell` is `None`, the library automatically infers a safe bounding box and uses non-periodic boundary conditions.

```python
# positions: (N, 3)
result = neighborlist_rs.build_neighborlists(None, positions, cutoff=5.0)
```

### Multi-Cutoff Search
Generates multiple neighbor lists in a single optimized pass over the structure.

```python
cutoffs = [6.0, 14.0, 20.0]
results = neighborlist_rs.build_neighborlists_multi(cell, positions, cutoffs)

# Access results by index (order of cutoffs)
for i, rc in enumerate(cutoffs):
    edge_index = results[i]["edge_index"]
```

---

## Batched API

The Batched API allows processing a large number of systems (e.g., a training batch) in parallel across CPU cores.

### Standard Batched Search
```python
# positions: (N_total, 3), batch: (N_total,)
# cells: (B, 3, 3) numpy array
batch_res = neighborlist_rs.build_neighborlists_batch(
    positions, batch, cells=cells, cutoff=5.0
)
```

---

## Utility Classes & Methods

### `PyCell`
- `PyCell(h_matrix, pbc=[True, True, true])`: Create from 3x3 list/array and PBC flags.
- `wrap(pos)`: Wrap a 3D point into the primary cell.
- `PyCell.from_ase(atoms)`: Create directly from an ASE Atoms object.

### Global Configuration
```python
# Set number of Rayon threads
neighborlist_rs.set_num_threads(8)

# Initialize Rust-side logging (info, debug, trace)
neighborlist_rs.init_logging("info")
```

---

## Performance Tuning

You can dynamically tune the internal thresholds to optimize performance for your specific hardware or system sizes.

```python
# Atoms below this use SIMD Brute-Force instead of Cell-Lists
neighborlist_rs.set_brute_force_threshold(1000)

# Atoms below this run serially (no thread overhead)
neighborlist_rs.set_parallel_threshold(300)

# Atoms below this use stack-allocated memory (max 1024)
neighborlist_rs.set_stack_threshold(1000)

print(f"Current BF threshold: {neighborlist_rs.get_brute_force_threshold()}")
```