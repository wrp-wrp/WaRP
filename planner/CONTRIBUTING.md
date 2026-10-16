# Contributing

Run the checks below and include tests for behavior changes.

## Setup

Run from `planner/`:

```bash
bash scripts/quickstart.sh              # cargo test --release + --features probe
bash scripts/quickstart.sh --with-python # also maturin develop + pytest
```

Use the pinned Rust 1.94.0 toolchain and, for the Python extension,
Python 3.9+ with maturin.

The directory is `planner/`. The Rust crate is `warp-planner`
(`warp_planner`); the Python package is `vecio-planner` (`vecio_planner`).

## Workflow

1. **Fork + branch** from `main`.
2. **Run the checks** before pushing:
   ```bash
   make fmt      # cargo fmt --all
   make lint     # cargo clippy --workspace --all-targets -- -D warnings
   make test-all # cargo test --release --features probe
   make release-check # reject private metadata in tracked files
   ```
3. **Add tests** alongside your change:
   - Trait-level invariants → `rust/tests/cross_strategy.rs`.
   - Strategy-specific behaviour → `#[cfg(test)] mod tests` in the
     strategy file.
   - Golden updates after an intentional algorithm change →
     `make bless`, then review the diff in
     `rust/tests/data/*/expected_plans.json` before committing.
4. **Keep the API stable.** Public additions are welcome; public
   renames or removals need a changelog entry and a deprecation path.
5. **Open a PR** describing what changed, why, and the performance
   impact (if any). Benchmark numbers from `bash scripts/bench.sh`
   are appreciated for planner changes.

The `HISTORY=1` release check applies to the original standalone release
branch. Use the normal `make release-check` for this bundled source.

## What goes where

- Paper algorithms live in `rust/src/strategy/*.rs` (one file per
  strategy, no cross-strategy imports).
- Engineering optimisations that preserve algorithmic output live in
  `rust/src/internal/`.
- Calibration / probing live in `rust/src/calibrate.rs` and
  `rust/src/probe.rs` (feature `probe`).
- The Python extension re-exports a small subset — don't expose
  anything async there.

The module-level docstring in `rust/src/lib.rs` lists the public
API; `cargo doc --open` renders it.

## Style

- `cargo fmt` for Rust, Ruff/Black defaults for Python.
- Prefer short module-level docstrings and one-line item docstrings
  over prose essays in the source.
- Keep `rust/src/strategy/*.rs` files self-contained (no
  cross-strategy imports).
- New dependencies on the core crate should be justified — the core
  has intentionally tiny dep closure.

## Reporting bugs

File an issue with:

- what you ran (`cargo` / `pip` command, minimal code sample),
- expected vs actual output,
- `rustc --version` and `uname -a` (or OS + arch).
