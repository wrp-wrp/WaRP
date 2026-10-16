# WaRP Paper Reproduction Guide

This fork supplies the Lance-side implementation used by the WaRP experiment
artifact. Readers should launch paper campaigns from
[`vecio-experiments`](https://anonymous.4open.science/r/vecio-experiments-anon-E7F8); this
repository supplies the pinned implementation and compiled binaries.

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
4. Run the `warp-experiments` toolchain preflight against this checkout.
5. Reproduce one selected experiment with the experiment registry's `--only`
   interface, then run the full campaign if the required public datasets and
   object-store resources are available.

The experiment repository records the exact companion commit and binary hashes
with fresh results. Frozen paper outputs remain available for rendering without
cloud credentials.

## Standalone examples

The following are developer checks, not substitutes for the registered paper
campaigns:

```bash
cargo run -p lance-io --example merge_comparison
cargo run -p lance-io --example realistic_ivf_benchmark
cargo bench -p lance-io --bench merge_strategies
```

Planner selection for low-level examples is controlled by environment variables
including `LANCE_IO_MERGE_STRATEGY`, `LANCE_IO_MERGE_MAX_REQUESTS`, and
`LANCE_IO_MERGE_CONCURRENT_PARALLELISM`. For paper runs, use the experiment
registry instead of manually reconstructing these settings.
