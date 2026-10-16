# WaRP Planner

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
![crates.io](https://img.shields.io/badge/crates.io-pending%20publish-lightgrey.svg)
![PyPI](https://img.shields.io/badge/PyPI-pending%20publish-lightgrey.svg)
![docs.rs](https://img.shields.io/badge/docs.rs-pending%20publish-lightgrey.svg)

**I/O merge planning for vector search over cloud object storage.**

Vector indexes often read many small ranges per query. WaRP Planner merges them
into fewer requests while balancing request latency, transferred bytes, and
available I/O concurrency.

The repository and Rust crate are named `warp-planner`; Rust imports
`warp_planner`. The Python distribution and import remain `vecio-planner` and
`vecio_planner`.

> **Pre-release:** version `0.1.0` is not yet on crates.io, PyPI, or docs.rs.
> Install from source for now.

The library provides six planning strategies, linear and piecewise latency
models, optional object-store calibration, and checked Rust and Python
interfaces. It plans reads; it does not build or reorder an index.

Related repositories: [vecio-experiments](https://anonymous.4open.science/r/vecio-experiments-anon-E7F8)
contains the paper artifact, while
[lance-s3-io-optimization](https://anonymous.4open.science/r/lance-s3-io-optimization-anon-95B5)
integrates the planner into Lance.

## Quick start

Open the [vecio-planner anonymous repository](https://anonymous.4open.science/r/vecio-planner-anon-2523),
click **Download Repository**, extract the ZIP, and run the commands below from
the extracted repository root.

### Rust

Use the pre-release crate as a path dependency:

```toml
[dependencies]
warp-planner = { path = "../warp-planner/rust" }
```

```rust
use warp_planner::{
    plan, CostModel, IoRequest, MainSolution, PlanConstraints,
};

let requests = vec![
    IoRequest { cluster_id: 0, offset: 0, length: 8_192 },
    IoRequest { cluster_id: 1, offset: 8_192, length: 8_192 },
    IoRequest { cluster_id: 2, offset: 5_000_000, length: 8_192 },
];
let cost = CostModel {
    latency_per_request: 27.0,
    latency_per_byte: 1.0e-5,
    concurrent_parallelism: 16.0,
    ..CostModel::default()
};

let output = plan(
    &MainSolution,
    &requests,
    &cost,
    &PlanConstraints::default(),
)
.expect("valid planning problem");
println!("{} merged requests", output.merged_requests.len());
```

Run the complete example with:

```bash
cargo run --release --example quickstart
```

### Python

Build the Python extension from the same checkout:

```bash
python3 -m venv python/.venv
source python/.venv/bin/activate
pip install maturin pytest
(cd python && maturin develop --release)
```

```python
import vecio_planner as vp

requests = [(0, 0, 8192), (1, 8192, 8192), (2, 5_000_000, 8192)]
cost = dict(
    latency_per_request=27.0,
    latency_per_byte=1.0e-5,
    concurrent_parallelism=16.0,
)
output = vp.plan(requests, "main_solution", **cost)
print(output["merged_requests"])
```

## Cost-model calibration

Feed measured `(bytes, latency_ms)` samples to `Calibrator`; it selects a
linear or piecewise model:

```rust
use warp_planner::Calibrator;

let mut calibrator = Calibrator::new();
for (bytes, latency_ms) in samples {
    calibrator.record(bytes, latency_ms);
}
let cost = calibrator.to_cost_model();
```

Enable the `probe` feature to collect samples directly from an `object_store`
backend. See
[`probe_and_plan.rs`](rust/examples/probe_and_plan.rs) for S3, GCS, Azure,
local disk, and in-memory usage.

## Results

The committed benchmark uses SIFT1M IVF ranges and the paper's S3 latency and
GET-cost model. Every strategy receives the same ranges.

| Setting | Value |
|---|---|
| Data | SIFT1M: 1,000,000 vectors, 128 dimensions |
| Index | FAISS `IndexIVFFlat`, L2, `nlist=4096`, `nprobe=128` |
| Training | First 200,000 vectors |
| Queries | First 32 queries; 128 inverted lists per query |
| Layout | One 512 MB object, 512 bytes/vector; lists ordered by centroid PC1 |
| Greedy | 4 MiB maximum gap, 10 MiB maximum merged span |
| MainSolution | No request cap; modeled concurrent reads from 1 to 64 |
| Storage model | Piecewise S3 calibration; `$0.0004` per 1,000 GETs |

| Concurrent reads | Separate | Greedy | MainSolution | vs Separate | Savings / 1M queries |
|---:|---:|---:|---:|---:|---:|
| 1 | 3,750.4 ms | 1,454.0 ms | **1,276.5 ms** | 66% lower latency | $39.48 |
| 16 | 262.8 ms | 194.2 ms | **125.2 ms** | 52% lower latency | $35.09 |
| 64 | 88.4 ms | 131.2 ms | **60.2 ms** | 32% lower latency | $32.11 |

These are model predictions, not end-to-end service latency. The committed
[`fixtures_manifest.json`](rust/tests/data/sift_paper_pca_4096x128/fixtures_manifest.json)
records the source hashes, toolchain, and index setup.

## Verification and benchmarks

Run the complete Rust and Python verification:

```bash
source python/.venv/bin/activate
bash scripts/quickstart.sh --with-python
make release-check
```

Run the short benchmark smoke or the full Criterion sweep:

```bash
make bench-smoke
make bench            # about 15–25 minutes
```

The full sweep writes CSV files under `target/bench-artifacts/` and HTML under
`target/criterion/`. See [docs/benchmarks.md](docs/benchmarks.md) for the
methodology and output schema.

## Algorithm guarantees

`MainSolution` is globally exact on small candidate sets. Larger sets use a
bounded adaptive search with a 5% stopping target. The inner partition solver
is exact for every candidate considered.

The larger search path does not claim a global `(1+ε)` guarantee. Brute-force
tests cover small inputs, and frozen workload tests detect quality regressions.

## License

Apache-2.0. See [LICENSE](LICENSE).
