# Plan: GitHub Actions CI Pipeline

## Phase 1: Linting & Static Analysis [checkpoint: 3cf462c]
- [x] Task: Initialize GitHub Actions directory and basic workflow structure in `.github/workflows/ci.yml`. [0a0cdcc]
- [x] Task: Implement Rust linting jobs (`rustfmt`, `clippy`). [55a7ac2]
- [x] Task: Implement Python linting jobs (`ruff check`, `ruff format`). [b8473bc]
- [x] Task: Verify linting jobs fail on intentional errors and pass on clean code. [55a7ac2, b8473bc]
- [ ] Task: Conductor - User Manual Verification 'Linting & Static Analysis' (Protocol in workflow.md)

## Phase 2: Linux Matrix Build & Test [checkpoint: c4f2c9a]
- [x] Task: Configure `maturin-action` for Linux (amd64). [8999686]
- [x] Task: Implement Python version matrix (3.11, 3.12, 3.13) for Linux. [8999686]
- [x] Task: Integrate `cargo test` and `pytest tests/` into the Linux job. [8999686]
- [x] Task: Verify tests run and report results correctly in the GitHub UI. [8999686]
- [ ] Task: Conductor - User Manual Verification 'Linux Matrix Build & Test' (Protocol in workflow.md)

## Phase 3: Cross-Platform Expansion [checkpoint: da87ffd]
- [x] Task: Add macOS (arm64) support to the matrix. [4c7a3df]
- [x] Task: Add Windows (amd64) support to the matrix. [4c7a3df]
- [x] Task: Add Ubuntu (arm64) support (using `maturin-action`'s cross-compilation or specialized runners). [4c7a3df]
- [x] Task: Explicitly add native Ubuntu arm64 runners to the matrix. [d406f30]
- [x] Task: Verify successful builds across the entire OS/Python matrix. [4c7a3df]
- [ ] Task: Conductor - User Manual Verification 'Cross-Platform Expansion' (Protocol in workflow.md)

## Phase 4: Benchmarks & Caching [checkpoint: e550380]
- [x] Task: Implement caching for `~/.cargo/registry` and `~/.cache/pip`. [c645049]
- [x] Task: Add a dedicated job to run `benchmarks/small_systems.py`. [c645049]
- [x] Task: Verify that CI execution time is optimized and benchmarks are recorded. [c645049]
- [ ] Task: Conductor - User Manual Verification 'Benchmarks & Caching' (Protocol in workflow.md)
