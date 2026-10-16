# WaRP Lance Companion

This repository is the Lance implementation companion for the WaRP paper. It
contains the scheduler integration, IVF layout changes, and executable entry
points used by the public experiment artifact.

## Repository split

```text
WaRP open-source release
|
+-- warp-planner
|   `-- standalone planner library, examples, and microbenchmarks
|
+-- warp-experiments
|   +-- public dataset preparation
|   +-- resolved paper protocols and parameters
|   +-- figure/table runners and renderers
|   `-- frozen paper results
|
`-- lance-s3-io-optimization          (this repository)
    +-- rust/lance-io/src/scheduler/  planner and scheduler integration
    +-- rust/lance-index/src/vector/  vector storage and centroid reordering
    +-- rust/lance/src/index/vector/  IVF execution integration
    +-- rust/lance-io/examples/       strict-budget and overhead binaries
    `-- rust/lance/examples/          full-pipeline and recall binaries
```

The manuscript sources are not required to build the code or reproduce the
paper results.

## Paper-facing binaries

| Binary | Purpose |
|---|---|
| `lance_scheduler_qps_s3` | Figure 4 real-S3 tiny-request fanout sweep |
| `lance_scheduler_qps_s3_per_iop` | Figure 4 serial per-I/O baseline |
| `strict_budget_real_s3` | Strict-S3 planner comparisons and Figure 11 policy transfer |
| `planner_overhead_real` | Planner compute-cost measurements |
| `full_pipeline_real_s3` | Figure 7/8 end-to-end throughput, latency, I/O, and range export |
| `full_pipeline_recall_s3` | Figure 7 full-query-set Recall@10 validation |

Build exactly the binaries consumed by `warp-experiments`:

```bash
cargo build --release -p lance \
  --example full_pipeline_real_s3 \
  --example full_pipeline_recall_s3
cargo build --release -p lance-io \
  --example strict_budget_real_s3 \
  --example planner_overhead_real \
  --example lance_scheduler_qps_s3 \
  --example lance_scheduler_qps_s3_per_iop
```

The executables are written to `target/release/examples/`.

Verify the two Figure 11 companion capabilities without running cloud work:

```bash
target/release/examples/strict_budget_real_s3 --help | grep -- --only-method
target/release/examples/full_pipeline_real_s3 --help | grep -- --out-query-ranges-json
```

## Reproducing paper results

Use [`vecio-experiments`](https://anonymous.4open.science/r/vecio-experiments-anon-E7F8)
as the reproduction entry point.
Its registry owns dataset preparation, fixed scientific parameters, execution,
validation, and rendering. Point it at this checkout with
`VECIO_LANCE_ROOT=/path/to/lance-s3-io-optimization` or its documented CLI
option. The experiment artifact validates the exact companion Git commit and the
required binary capabilities before starting a campaign; it does not clone or
modify this repository automatically.

For anonymous review, download this repository with the mirror's **Download
Repository** button. The exported tree has no `.git`; `warp-experiments`
instead verifies its complete Cargo, Rust, proto, and Python source fingerprint
against the pinned companion release before building. This is an identity
check, not a bypass of the source contract.

Object-store experiments require credentials supplied through the standard AWS
credential chain and a reader-owned region/bucket or public object URI. No
credentials, account identifiers, or machine-local paths are embedded here.

For standalone smoke checks that require no cloud access:

```bash
cargo run -p lance-io --example merge_comparison
cargo run -p lance-io --example realistic_ivf_benchmark
cargo test -p lance-io
```

See [PAPER_REPRODUCTION_GUIDE.md](./PAPER_REPRODUCTION_GUIDE.md) for the
implementation map and experiment workflow.
