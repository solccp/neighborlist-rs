# Repository Guidelines

## Project Structure & Modules
- `src/`: Rust core with `cell` (box geometry), `single`/`batch` (search kernels), and `search` (shared logic); `lib.rs` exposes PyO3 bindings.
- `tests/`: Python pytest suite validating the bindings and ASE integration; fixtures live in `tests/data/`.
- `benchmarks/`: Python scripts for scaling and regression timing runs.
- `docs/`, `spec.md`, `GEMINI.md`: design and review notes; `conductor/` holds product and style guardrails; `proptest-regressions/` stores property-test seeds.

## Build, Test, and Development Commands
- `pip install .` (or `maturin develop`): build and install the Python package against the local Rust library.
- `cargo build`: compile the Rust crate.
- `cargo fmt` / `cargo clippy --all-targets --all-features`: format and lint; keep both clean before sending patches.
- `cargo test`: run Rust unit/property tests.
- `pytest tests`: run Python binding tests; add `-k <name>` to target specific cases.
- `python benchmarks/scaling.py`: optional performance smoke; note hardware in results.

## Coding Style & Naming
- Rust: prefer safe code paths; follow `rustfmt` defaults and fix `clippy` warnings; structs/enums `PascalCase`, modules/functions `snake_case`, constants `SCREAMING_SNAKE_CASE`. Align with `conductor/product-guidelines.md` (determinism, zero-copy).
- Python: Google style summary applies (`conductor/code_styleguides/python.md`): 4-space indent, 80 cols, docstrings with Args/Returns/Raises, `snake_case` functions, `PascalCase` classes. Keep NumPy/ASE interop zero-copy when possible.
- Comments explain *why*; prefer tracing via `tracing_subscriber` over ad hoc prints; gate heap profiling behind the `dhat-heap` feature only.

## Testing Guidelines
- Add Rust tests alongside implementations or under `tests/` using `proptest` for geometry invariants; record new failing seeds in `proptest-regressions/`.
- Python: mirror new APIs with pytest cases; include small deterministic fixtures and PBC edge cases. Use `pytest --maxfail=1 -q` for quick iterations.
- For behavior that depends on cell topology or multi-cutoff passes, cover both periodic and isolated systems.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit style: `type(scope): action` (e.g., `perf(simd): optimize inner loop`). Use imperative verbs.
- PRs should state intent, list test commands run (cargo/pytest/benchmarks if performance-related), and link issues or specs. Include screenshots only when user-facing behavior changes (e.g., docs diagrams).
- Note any benchmarking context (CPU count, dataset) when claiming performance wins; add reproduction commands in the PR description.
