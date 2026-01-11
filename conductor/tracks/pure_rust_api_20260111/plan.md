# Plan: Pure Rust API Surface Implementation

## Phase 1: Foundation & Types [checkpoint: e99fb08]
- [x] Task: Define the public `NeighborList` struct and associated types in a new module.
- [x] Task: Implement conversions from standard Rust types (`[f64; 3]`, etc.) to internal `nalgebra` types.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Foundation & Types' (Protocol in workflow.md)

## Phase 2: Single System API [checkpoint: e6541a8]
- [x] Task: Implement the public-facing `build_neighborlists` function in `src/api.rs`.
- [x] Task: Add unit tests in Rust to verify `build_neighborlists` for both PBC and non-PBC cases.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Single System API' (Protocol in workflow.md)

## Phase 3: Batched API
- [x] Task: Implement the public-facing `build_neighborlists_batch` function in `src/api.rs`.
- [x] Task: Add unit tests in Rust to verify `build_neighborlists_batch` with mixed PBC/non-PBC systems.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Batched API' (Protocol in workflow.md)

## Phase 4: Integration & Documentation
- [ ] Task: Re-export the public API at the crate root in `src/lib.rs`.
- [ ] Task: Update `PYTHON_API.md` or add a `RUST_API.md` (if appropriate) to document usage.
- [ ] Task: Final verification of Python bindings to ensure no regressions.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Integration & Documentation' (Protocol in workflow.md)
