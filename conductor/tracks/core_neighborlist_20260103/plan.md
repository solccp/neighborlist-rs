# Track Plan: Core Neighborlist Implementation (Phase 1)

## Phase 1: Foundation & Math
- [ ] Task: Project Scaffolding (Cargo.toml, PyO3 setup, Directory structure)
- [ ] Task: Implement `Cell` struct and Coordinate Transformations (Cartesian <-> Fractional)
- [ ] Task: Implement PBC Wrapping and Shift Calculation logic
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Foundation & Math' (Protocol in workflow.md)

## Phase 2: Core Search Engine
- [ ] Task: Implement `CellList` binning logic
- [ ] Task: Implement Sequential Neighbor Search (Brute-force reference)
- [ ] Task: Implement Cell-List based Neighbor Search
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
