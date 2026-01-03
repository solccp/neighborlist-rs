# Track Plan: Core Neighborlist Implementation (Phase 1)

## Phase 1: Foundation & Math [checkpoint: 77d41f6]
- [x] Task: Project Scaffolding (Cargo.toml, PyO3 setup, Directory structure) 63f851e
- [x] Task: Implement `Cell` struct and Coordinate Transformations (Cartesian <-> Fractional) 243f922
- [x] Task: Implement PBC Wrapping and Shift Calculation logic ec6eba2
- [x] Task: Conductor - User Manual Verification 'Phase 1: Foundation & Math' (Protocol in workflow.md) 77d41f6

## Phase 2: Core Search Engine
- [x] Task: Implement `CellList` binning logic 102c9a8
- [x] Task: Implement Sequential Neighbor Search (Brute-force reference) 6e26185
- [x] Task: Implement Cell-List based Neighbor Search 4a7de22
- [ ] Task: Parallelize Search using Rayon
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Core Search Engine' (Protocol in workflow.md)

## Phase 3: Python Bindings
- [ ] Task: Setup PyO3 Module and NumPy integration
- [ ] Task: Implement `build_neighborlists` Python wrapper
- [ ] Task: Implement sorting policy logic (ByIThenJ)
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Python Bindings' (Protocol in workflow.md)

## Phase 4: Verification & Finalization
- [ ] Task: Implement comprehensive integration tests (Rust & Python)
- [ ] Task: Benchmark performance against large systems
- [ ] Task: Documentation and Examples (README.md)
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Verification & Finalization' (Protocol in workflow.md)
