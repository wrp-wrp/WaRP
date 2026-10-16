# Lance integration for WaRP

This fork contains the Lance implementation used in the WaRP paper:
request planning in the I/O scheduler, IVF execution changes, and cluster
reordering. The planner entry point is `lance_io::scheduler::VecioMergePlanner`.

The experiments and frozen results are in [../evaluation](../evaluation/).
For the standalone Rust/Python planner, use [../planner](../planner/).

## WaRP Research Fork

- [ARTIFACT.md](ARTIFACT.md): benchmark binaries and build commands.
- [PAPER_REPRODUCTION_GUIDE.md](PAPER_REPRODUCTION_GUIDE.md): implementation map and reproduction steps.
- [rust/lance-io/src/scheduler/](rust/lance-io/src/scheduler/): request planning and scheduler integration.

## Quick Start

Run a local example from `lance/`:

```bash
cargo run -p lance-io --example merge_comparison
cargo run -p lance-io --example realistic_ivf_benchmark
cargo bench -p lance-io --bench merge_strategies
```

For paper measurements, follow [evaluation/README.md](../evaluation/README.md).
The evaluation scripts build this fixed source and check its fingerprint
before running S3 experiments.

## Directory structure

| Path | Contents |
| --- | --- |
| `rust/lance-io/src/scheduler/` | Request planner and I/O scheduler |
| `rust/lance-index/src/vector/` | Vector storage and centroid reordering |
| `rust/lance/src/index/vector/` | IVF execution integration |
| `rust/lance-io/examples/` | I/O and planner-overhead benchmarks |
| `rust/lance/examples/` | End-to-end and recall benchmarks |
| `python/`, `java/` | Language bindings |

## Upstream Lance

Lance is a storage format for multimodal data with vector search and random
access. See the [Lance documentation](https://lance.org/) and
[upstream repository](https://github.com/lance-format/lance) for general use.
This fork retains the upstream [LICENSE](LICENSE) and [NOTICE](NOTICE).
