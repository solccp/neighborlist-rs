# Rust API Documentation: neighborlist-rs

`neighborlist-rs` provides a high-performance Rust API for neighbor list construction.

## Basic Usage

To use `neighborlist-rs` in your Rust project, add it as a dependency in your `Cargo.toml`:

```toml
[dependencies]
neighborlist-rs = { git = "https://github.com/your-repo/neighborlist-rs.git" }
```

### Cell and Shift Conventions
The cell `h: [[f64; 3]; 3]` passed as `Some((&h, pbc))` holds the lattice vectors $\mathbf{a}, \mathbf{b}, \mathbf{c}$ as its **rows**, the same layout as ASE's `atoms.cell` (and pymatgen, torch-sim, nvalchemiops):

$$h = \begin{bmatrix} \mathbf{a} \\ \mathbf{b} \\ \mathbf{c} \end{bmatrix} = \begin{bmatrix} a_x & a_y & a_z \\ b_x & b_y & b_z \\ c_x & c_y & c_z \end{bmatrix}$$

Each edge `(i, j)` with integer shift `S` (a row vector) stands for the displacement

$$\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i + S\,h$$

i.e. `x_j - x_i + S @ atoms.cell`. The same applies to every cell in `build_neighborlists_batch`.

*Changed in 0.2.0:* earlier versions took the lattice vectors as the **columns** of `h`. Callers that passed a transposed cell (`atoms.cell.T`) must drop the transpose. Orthogonal and symmetric cells look the same either way, so the difference only shows up on oblique cells, such as fcc(111) slabs.

### Single System Search

```rust
use neighborlist_rs::{build_neighborlists, NeighborList};

fn main() -> Result<(), String> {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let cutoff = 1.5;
    
    // Non-periodic (Isolated)
    let result = build_neighborlists(&positions, cutoff, None, true)?;
    
    println!("Found {} edges", result.edge_index.len() / 2);
    
    // Periodic Boundary Conditions
    let cell_matrix = [
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ];
    let pbc = [true, true, true];
    let result_pbc = build_neighborlists(
        &positions, 
        cutoff, 
        Some((&cell_matrix, pbc)), 
        true
    )?;
    
    Ok(())
}
```

### Batched Search

```rust
use neighborlist_rs::build_neighborlists_batch;

fn main() -> Result<(), String> {
    let positions = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], // System 0
        [1.0, 1.0, 1.0], [9.0, 1.0, 1.0]  // System 1
    ];
    let batch = [0, 0, 1, 1];
    let cutoff = 2.5;
    
    let cell_1 = [
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ];
    let pbc_1 = [true, true, true];
    
    let cells = [
        None,             // System 0 is isolated
        Some((cell_1, pbc_1)) // System 1 is periodic
    ];
    
    let result = build_neighborlists_batch(
        &positions,
        &batch,
        cutoff,
        Some(&cells),
        true
    )?;
    
    Ok(())
}
```

### Half List with Compact Output (`build_half_list`)

For large cutoffs and repeated calls (e.g. DFT-D3 at 60 Bohr), `build_half_list` returns the same pairs as `build_neighborlists` as separate `u32`/`i32` arrays:

```rust
use neighborlist_rs::build_half_list;

fn main() -> Result<(), String> {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]];
    let cell = [[3.0, 0.0, 0.0], [1.5, 2.6, 0.0], [0.0, 0.0, 10.0]]; // rows
    let list = build_half_list(&positions, 31.75, Some((&cell, [true, true, false])))?;
    for e in 0..list.len() {
        let (i, j, s) = (list.i[e], list.j[e], list.shifts[e]);
        // r_ij = x_j - x_i + s[0] * cell[0] + s[1] * cell[1] + s[2] * cell[2]
    }
    Ok(())
}
```

- Pairs have `i < j`, or `i == j` with the first non-zero component of the shift positive. The cell convention is the same as for `build_neighborlists`; `None` means an isolated system.
- Each candidate pair is evaluated once by sorted index, a branchless count pass sizes the output, and the fill pass writes it once, in parallel, with no concatenation.
- It runs on rayon's *current* pool, so you can call it inside your own `ThreadPool::install`. It never changes the global pool. The output order is deterministic and does not depend on the thread count.

## Data Schema

### `NeighborList` Struct
- `edge_index: Vec<i64>`: A flat vector of length `2 * E`. It contains source indices followed by target indices: `[src_0, src_1, ..., src_E, dst_0, dst_1, ..., dst_E]`. This layout is designed for compatibility with GNN frameworks like PyTorch Geometric.
- `shifts: Vec<i32>`: A flat vector of length `3 * E`. It contains the periodic shift vectors for each edge: `[sx_0, sy_0, sz_0, sx_1, sy_1, sz_1, ...]`.

### Note on Directionality
The library currently enforces `i < j` for neighbor pairs in isolated systems and unique pairs in periodic systems. This means it returns **half-neighbors** (one direction only). If your application requires both directions (undirected graph for message passing), you should double the edges and negate the shifts for the reverse direction.

When the cutoff exceeds a cell length, an atom sees its own periodic images. Each self-image pair `(i, i, S)` is emitted once, with the sign chosen so that the first non-zero component of `S` is positive; `(i, i, -S)` is its implied reverse. Doubling the list as above therefore reproduces a full list (e.g. ASE's `neighbor_list`) exactly. Along a non-periodic axis no images are generated, regardless of the cutoff or vacuum width.
