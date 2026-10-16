# WaRP Paper Reproduction Guide

Run the paper experiments from [../evaluation](../evaluation/). This fork
contains the fixed engine implementation and benchmark binaries.

## What is fixed and what the reader supplies

Scientific parameters are fixed by the experiment registry and runner code:
dataset geometry, query counts, warm-up counts, concurrency, `nprobe`, refine
factor, request budgets, planner parallelism, greedy grids, random seeds, and
rendering inputs. A reader chooses only infrastructure inputs such as the Lance
checkout, public dataset cache, object-store location, region, and credentials.

The companion checkout is selected with `VECIO_LANCE_ROOT` (or the equivalent
experiment CLI flag). AWS credentials are read from the standard SDK credential
chain. Do not add credentials to command files or repository configuration.

## Implementation map

```text
vector query
    |
    v
Lance IVF partition selection
    |  rust/lance/src/io/exec/knn.rs
    v
batched IVF payload ranges
    |  rust/lance-index/src/vector/storage.rs
    v
WaRP request planning
    |  rust/lance-io/src/scheduler/{cost_model,balanced_ub}.rs
    v
Lance file scheduler and object-store Range GETs
       rust/lance-io/src/scheduler/scheduler_impl.rs
```

Centroid reordering and its IVF build hook live in:

- `rust/lance-index/src/vector/centroid_reorder.rs`
- `rust/lance/src/index/vector/ivf.rs`
- `rust/lance/src/index/vector/ivf/v2.rs`

## Validation order

1. Run `cargo fmt --all -- --check`.
2. Run `cargo test -p lance-io`.
3. Build the six release examples listed in [ARTIFACT.md](./ARTIFACT.md).
4. Run the toolchain preflight from `../evaluation` against this source.
5. Reproduce one selected experiment with the experiment registry's `--only`
   interface, then run the full campaign if the required public datasets and
   object-store resources are available.

The experiment repository records the exact companion commit and binary hashes
with fresh results. Frozen paper outputs remain available for rendering without
cloud credentials.

## Standalone examples

Use these local examples to check engine behavior. Run paper measurements
through the evaluation registry:

```bash
cargo run -p lance-io --example merge_comparison
cargo run -p lance-io --example realistic_ivf_benchmark
cargo bench -p lance-io --bench merge_strategies
```

Planner selection for low-level examples is controlled by environment variables
including `LANCE_IO_MERGE_STRATEGY`, `LANCE_IO_MERGE_MAX_REQUESTS`, and
`LANCE_IO_MERGE_CONCURRENT_PARALLELISM`. For paper runs, use the experiment
registry instead of manually reconstructing these settings.
