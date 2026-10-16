# WaRP Lance integration

This fork contains request planning, IVF layout changes, and the benchmark
binaries used by [the evaluation project](../evaluation/).

## Repository layout

```text
../planner/                       standalone Rust/Python library
../evaluation/                    experiments, datasets, and frozen results
./rust/lance-io/src/scheduler/     request planner and I/O scheduler
./rust/lance-index/src/vector/     vector storage and centroid reordering
./rust/lance/src/index/vector/     IVF execution integration
./rust/lance-io/examples/          I/O and overhead benchmarks
./rust/lance/examples/             end-to-end and recall benchmarks
```

## Benchmark binaries

| Binary | Purpose |
| --- | --- |
| `lance_scheduler_qps_s3` | Figure 4 tiny-request fanout sweep |
| `lance_scheduler_qps_s3_per_iop` | Figure 4 serial per-I/O baseline |
| `strict_budget_real_s3` | Planner comparisons and Figure 11 policy transfer |
| `planner_overhead_real` | Planner computation overhead |
| `full_pipeline_real_s3` | Figure 7/8 throughput, latency, I/O, and range export |
| `full_pipeline_recall_s3` | Figure 7 Recall@10 validation |

Build from `lance/`:

```bash
cargo build --release -p lance   --example full_pipeline_real_s3 --example full_pipeline_recall_s3
cargo build --release -p lance-io   --example strict_budget_real_s3 --example planner_overhead_real   --example lance_scheduler_qps_s3 --example lance_scheduler_qps_s3_per_iop
```

The binaries are written to `target/release/examples/`. Figure 11 requires
these two interfaces:

```bash
target/release/examples/strict_budget_real_s3 --help | grep -- --only-method
target/release/examples/full_pipeline_real_s3 --help | grep -- --out-query-ranges-json
```

## Reproduce paper results

Follow [evaluation/README.md](../evaluation/README.md). Its scripts prepare
datasets, apply the paper parameters, run measurements, and render figures.
They discover this directory as `../lance` and verify the pinned source
fingerprint before building. Set `VECIO_LANCE_ROOT` to use an explicit path.

Live experiments use the standard AWS credential chain and a reader-owned
bucket and region. Frozen results can be rendered locally.

For local engine checks:

```bash
cargo run -p lance-io --example merge_comparison
cargo run -p lance-io --example realistic_ivf_benchmark
cargo test -p lance-io
```

See [PAPER_REPRODUCTION_GUIDE.md](PAPER_REPRODUCTION_GUIDE.md) for the query
pipeline and validation order.
