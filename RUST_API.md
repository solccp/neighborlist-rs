# Rust API Documentation: neighborlist-rs

`neighborlist-rs` provides a high-performance Rust API for neighbor list construction.

## Basic Usage

To use `neighborlist-rs` in your Rust project, add it as a dependency in your `Cargo.toml`:

```toml
[dependencies]
neighborlist-rs = { git = "https://github.com/your-repo/neighborlist-rs.git" }
```

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

## Data Schema

### `NeighborList` Struct
- `edge_index: Vec<i64>`: A flat vector of length `2 * E`. It contains source indices followed by target indices: `[src_0, src_1, ..., src_E, dst_0, dst_1, ..., dst_E]`. This layout is designed for compatibility with GNN frameworks like PyTorch Geometric.
- `shifts: Vec<i32>`: A flat vector of length `3 * E`. It contains the periodic shift vectors for each edge: `[sx_0, sy_0, sz_0, sx_1, sy_1, sz_1, ...]`.

### Note on Directionality
The library currently enforces `i < j` for neighbor pairs in isolated systems and unique pairs in periodic systems. This means it returns **half-neighbors** (one direction only). If your application requires both directions (undirected graph for message passing), you should double the edges and negate the shifts for the reverse direction.
