# Specification: Pure Rust API Surface

## Overview
This track aims to expose a clean, idiomatic, and high-performance Rust API for `neighborlist-rs`. Currently, the library is primarily consumed via Python bindings, making its core functionality inaccessible or difficult to use directly from other Rust applications (e.g., high-performance inference servers).

## Goals
- Expose core neighbor list construction logic as a public Rust API.
- Support both single-system and batched processing.
- Minimize external dependencies for consumers by using standard Rust types in the public API where possible.
- Provide a data layout that is "GNN-ready" (compatible with PyTorch Geometric and similar frameworks).

## Functional Requirements

### 1. Public API Module
- Create a new module `src/api.rs` to house the public Rust API.
- Re-export the contents of `src/api.rs` in `src/lib.rs` to make them available at the crate root.

### 2. Data Structures
- **`NeighborList` Struct:**
    - `edge_index: Vec<i64>`: Length `2 * N_edges`, containing source indices followed by target indices (GNN-style).
    - `shifts: Vec<i32>`: Length `3 * N_edges`, containing flat `[x, y, z]` shift vectors.
- **`Cell` and `PBC` Support:**
    - Expose necessary types to define simulation boxes and periodic boundary conditions in a type-safe manner.

### 3. Public Functions
- **`build_neighborlists`:**
    - Inputs:
        - `positions: &[[f64; 3]]`
        - `cutoff: f64`
        - `cell: Option<(&[[f64; 3]; 3], [bool; 3])>` (Cell matrix and PBC flags)
        - `parallel: bool`
    - Output: `Result<NeighborList, String>`
- **`build_neighborlists_batch`:**
    - Inputs:
        - `positions: &[[f64; 3]]` (All atoms in batch)
        - `batch: &[i32]` (Batch indices for each atom)
        - `cutoff: f64`
        - `cells: Option<&[([f64; 9], [bool; 3])]>` (Optional cell for each system in batch)
        - `parallel: bool`
    - Output: `Result<NeighborList, String>`

## Non-Functional Requirements
- **Performance:** The Rust API should have zero or minimal overhead compared to the internal search logic.
- **Idiomatic Rust:** Use standard types and `Result` for error handling.

## Acceptance Criteria
- [ ] `neighborlist-rs` can be added as a git or crates.io dependency and used in a pure Rust project.
- [ ] `build_neighborlists` and `build_neighborlists_batch` are accessible from the crate root.
- [ ] Documentation and/or examples are provided for the new Rust API.
- [ ] Existing Python bindings continue to function without regression.

## Out of Scope
- Exposing internal grid or brute-force structures directly.
- Multi-cutoff support in the initial Rust API (can be added later if needed).
