# ASE Integration Plan

**Goal:** Add a Python API function `build_from_ase` that accepts an ASE `Atoms` object and computes neighbor lists, handling `positions`, `cell`, and `pbc` automatically.

## Context
ASE (Atomic Simulation Environment) is the standard interface for atomic structures in Python. Users currently have to manually extract positions and cell, and handle the logic for isolated/periodic systems before calling `neighborlist-rs`.

## Strategy

1.  **Signature:** `build_from_ase(atoms, cutoff, parallel=True)`
2.  **Logic:**
    *   Extract `atoms.get_positions()` -> `(N, 3)` array.
    *   Extract `atoms.get_cell()` -> `(3, 3)` array.
    *   Extract `atoms.get_pbc()` -> `(3,)` bool array.
    *   **PBC Handling:**
        *   **All True:** Use the provided cell.
        *   **All False:** Use `None` for cell (triggers auto-box inference).
        *   **Mixed:** Construct a modified cell where non-periodic dimensions are expanded significantly (e.g., by extending the lattice vector to cover the atoms + buffer).
            *   *Alternative:* For v1, we might just restrict to all-or-nothing PBC, or just implement the "Large Box" trick for mixed cases.
            *   *Decision:* Implement "Large Box" logic in Rust. If a dimension is non-periodic, replace its lattice vector with a very large vector covering the extent of atoms + margins, ensuring no wrapping occurs within the cutoff.
            *   *Actually:* `neighborlist-rs` `Cell` requires a 3x3 matrix. If we have a slab (periodic X/Y, free Z), we need the X/Y vectors to be exact. The Z vector should be huge.
            *   We will modify the input cell matrix before creating the internal `Cell`.
3.  **Implementation:**
    *   Add `build_from_ase` to `src/lib.rs`.
    *   Use `PyAny` to duck-type the `atoms` object (look for `.positions` or `.get_positions()`, `.cell`, `.pbc`).
    *   Calls internal `build_neighborlists` logic.

## Steps
1.  [x] Modify `src/lib.rs` to add `build_from_ase`.
2.  [x] Implement extraction of ASE attributes via PyO3.
3.  [x] Implement mixed PBC handling logic (mixed PBC currently raises ValueError).
4.  [x] Update `PYTHON_API.md`.
5.  [x] Add `tests/test_ase.py` with `pytest` and `ase` dependency (mocked or real).
