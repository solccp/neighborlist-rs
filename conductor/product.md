# Initial Concept

Rust Neighborlist Library (neighborlist-rs): A high-performance neighborlist construction library for atomistic systems (molecules and crystals/PBC), primarily for Python/PyTorch MLIP training and inference.

# Product Guide

## Project Name
**neighborlist-rs**

## Vision & Goals
To build a high-performance, safe, and deterministic neighborlist construction library in Rust that serves as a foundational component for machine learning interatomic potentials (MLIPs) and molecular dynamics (MD) engines.
- **Primary Goal:** Enable equivariant GNNs (like e3nn) to efficiently query atomistic neighbors with correct PBC displacement vectors.
- **Secondary Goal:** Support dynamic MD simulations with skin/rebuild logic.

## Target Audience
1.  **MLIP Researchers/Developers:** Using Python/PyTorch for training equivariant GNNs.
2.  **MD Engine Developers:** Integrating custom rust-based or python-based MD loops.

## Core Capabilities (Phase 1)
-   **System Support:**
    -   Molecules (Non-PBC).
    -   Crystals (PBC) including Triclinic cells.
    -   Mixed PBC systems (e.g., Slabs, Wires).
-   **Multi-List Generation:**
    -   Simultaneously generate lists for Local (GNN), Dispersion (DFTD3), and Coulomb interactions in a single pass.
-   **Batched Processing:**
    -   Efficiently process batches of structures in parallel, maximizing CPU utilization for datasets with many small systems.
-   **Output:**
    -   Edge indices (`i`, `j`) and Shift vectors (`sx`, `sy`, `sz`).
    -   Configurable directed/undirected edges.
-   **Integration:**
    -   Zero-copy (where possible) Python bindings via PyO3/maturin.
    -   Direct integration with ASE (Atomic Simulation Environment) for seamless workflow.

## Advanced Features (Phase 2)
-   **Dynamic Neighbor Management:**
    -   Skin buffer implementation.
    -   Automatic rebuild detection based on displacement.

## Non-Functional Requirements
-   **Performance:** $O(N)$ scaling for large systems with optimizations for cache locality (Z-order sorting) and adaptive parallelization.
    -   **Dynamic Configuration:** Runtime-tunable thresholds for selecting optimal strategies (SIMD Brute-Force vs Cell-List) based on target hardware capability.
-   **Parallelism:** Multi-core CPU support (e.g., via Rayon) to maximize construction speed for large atomistic systems.
-   **Determinism:** Configurable sorting policies for reproducible results.
-   **Safety:** Robust handling of edge cases (sparse/dense systems) without panics.
