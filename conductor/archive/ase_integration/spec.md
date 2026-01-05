# Specification: ASE Integration

## 1. Objectives
Provide seamless integration with the ASE (Atomic Simulation Environment) library, allowing users to generate neighbor lists directly from ASE `Atoms` objects.

## 2. Scope
-   Implement `build_from_ase(atoms, cutoff)` in the Python API.
-   Handle `positions`, `cell`, and `pbc` extraction automatically.
-   Support Periodic Boundary Conditions (PBC):
    -   All-periodic: Standard cell.
    -   Non-periodic: Auto-box inference.
    -   Mixed PBC: Correctly handle systems with some periodic and some non-periodic dimensions.

## 3. Success Criteria
-   `build_from_ase` is available in the `neighborlist_rs` module.
-   It accepts an ASE `Atoms` object and returns the correct neighbor list dictionary.
-   Mixed PBC systems are handled correctly without errors.
