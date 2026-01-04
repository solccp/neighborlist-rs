# Code Review — 2026-01-05

Scope: Rust core neighbor search, PyO3 bindings, and Python tests/docs.

## Findings (detailed)
- **Shape validation missing in batch bindings (`src/lib.rs:275`, `src/lib.rs:376`)**  
  Risk: `build_neighborlists_batch` and `build_neighborlists_batch_multi` assume `positions` is `(N,3)` and index `row[2]` directly; a bad caller shape (e.g., `(N,2)` from a slicing mistake) will panic in Rust and take down the interpreter.  
  Suggestion: mirror the single-system guard—reject any non-`(N,3)` shape with a `ValueError` before building `pos_vec`. Add a pytest covering a wrong shape to ensure it stays checked.

- **Batch grouping assumes sorted batch ids (`src/lib.rs:307`, `src/lib.rs:405`)**  
  Risk: System boundaries are inferred from value changes in `batch`, so unsorted ids silently merge/split systems (typical with shuffled PyTorch/ASE batches). That yields wrong neighbor lists without an explicit error signal.  
  Suggestion: either validate monotonicity (error if `batch[i] < batch[i-1]`) or construct segment offsets from a stable grouping of unique ids. At minimum, document the sorted requirement in `PYTHON_API.md` and add a test that fails on unsorted input to avoid silent corruption.

- **Cutoff validation absent in all entrypoints (`src/single.rs:9`, `src/single.rs:89`, bindings)**  
  Risk: `cutoff`/`cutoffs` can be negative or NaN. Auto-boxing then produces negative box edges; bin counts can become zero/NaN; comparisons short-circuit to empty outputs. This is a silent logic hazard that makes debugging hard.  
  Suggestion: early-check every cutoff for `is_finite() && > 0.0` in Rust and raise `ValueError` from PyO3. Add unit tests for invalid cutoffs to pin the behavior.

- **Hot-path allocations in neighbor search (`src/search.rs:383`, `src/search.rs:492`)**  
  Risk: Each atom visit allocates three `Vec`s for offset tables in both count and fill passes. On large systems this adds allocator churn and pressure to the GC bridge, reducing throughput and increasing variance.  
  Suggestion: precompute the offset tuples once per `CellList` (store on the struct) or reuse stack-bounded `ArrayVec`/small arrays sized by `n_search`. Share them between count/fill passes to make the inner loops allocation-free. A micro-benchmark in `benchmarks/scaling.py` would quantify gains.

- **ASE PBC path effectively untested (`tests/test_ase.py:12-83`)**  
  Risk: The critical column-vs-row cell convention is only described in comments and ends with `pass`, so regressions in `build_from_ase` transpose logic will slip in unnoticed.  
  Suggestion: replace the comment block with an assertion that `build_from_ase` matches a manual `PyCell` call on a small periodic cell (and that mixed-PBC errors still raise). Mark it as a non-skip test to guard the conversion contract.

- **Documentation type drift (`PYTHON_API.md:26-29` vs `src/lib.rs:203`, `src/lib.rs:355`)**  
  Risk: Docs promise `edge_index` as `uint64`, but the bindings emit `int64`. Downstream users relying on unsigned or expecting no negatives may get surprises.  
  Suggestion: align one way or the other—either change docs to `int64` or switch the numpy arrays to `u64` on export. Add a small doc-test snippet to `PYTHON_API.md` to lock the dtype expectation.

## Additional minor observations
- Auto-box span calculation in single/batch paths uses `max_bound - min_bound` without padding for degenerate systems of a single atom in one dimension; currently the `margin` keeps it positive, but a brief comment or `max(span, 0.0)` guard would clarify intent.
- Logging: `init_logging` uses `FmtSpan::CLOSE`, which can be noisy for hot paths; consider gating span logging behind an env flag (e.g., `NLRS_SPAN_LOG=1`) to avoid surprising users who enable debug logs.

## Suggested next steps
1) Add validation + tests for `(N,3)` shapes and positive/finite cutoffs.  
2) Decide on batch id contract (sorted vs arbitrary) and enforce it in code/tests/docs.  
3) Remove per-atom allocations in `search.rs` and benchmark the delta.  
4) Reinstate an ASE orientation test and fix any discovered transpose issues.  
5) Reconcile `edge_index` dtype between code and docs.
