# WaRP planner

Plan object-store reads from a list of byte ranges. The planner balances
request overhead, extra bytes, and I/O concurrency, then returns merged
ranges and assignments to the original requests.

The Rust crate is `warp-planner` (`warp_planner`); the Python package is
`vecio-planner` (`vecio_planner`). Version 0.1.0 is available from source.
The library includes six strategies, linear and piecewise latency models,
and optional object-store calibration.

Paper experiments are in [../evaluation](../evaluation/). The paper's Lance
implementation is in [../lance](../lance/); it is separate from this library.

## Quick start

Run these commands from `planner/`. Rust 1.82+ is required.

### Rust

```bash
cargo run --locked --release --example quickstart
```

In another project, add the crate as a path dependency:

```toml
[dependencies]
warp-planner = { path = "/path/to/WaRP/planner/rust" }
```

```rust
use warp_planner::{plan, CostModel, IoRequest, MainSolution, PlanConstraints};

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
let output = plan(&MainSolution, &requests, &cost, &PlanConstraints::default())
    .expect("valid planning inputs");
println!("{:?}", output.merged_requests);
```

### Python

Use Python 3.9+ and an active virtual environment:

```bash
python3 -m venv python/.venv
source python/.venv/bin/activate
CARGO_PROFILE_RELEASE_STRIP=none python -m pip install ./python
python python/examples/quickstart.py
```

Keeping native symbols avoids a Rust/macOS import issue on recent macOS versions.

```python
import vecio_planner as vp

requests = [(0, 0, 8192), (1, 8192, 8192), (2, 5_000_000, 8192)]
cost = dict(latency_per_request=27.0, latency_per_byte=1.0e-5,
            concurrent_parallelism=16.0)
output = vp.plan(requests, "main_solution", **cost)
print(output["merged_requests"])
```

See the [Python API](python/README.md) and [Rust examples](rust/examples/README.md).

## Cost-model calibration

Fit a model from measured `(bytes, latency_ms)` samples:

```rust
use warp_planner::Calibrator;

let mut calibrator = Calibrator::new();
for (bytes, latency_ms) in samples {
    calibrator.record(bytes, latency_ms);
}
let cost = calibrator.to_cost_model();
```

Enable the `probe` feature to collect samples from an `object_store` backend.
[probe_and_plan.rs](rust/examples/probe_and_plan.rs) covers S3, GCS, Azure,
local disk, and in-memory storage.

## Results

The included SIFT1M fixture uses 1,000,000 128-dimensional vectors, an
L2 IVF-Flat index with `nlist=4096` and `nprobe=128`, and the first 32 queries.
Training uses the first 200,000 vectors. Cluster lists are ordered by centroid
PC1 in a 512 MB object. Greedy uses a 4 MiB gap and 10 MiB span; MainSolution
has no request cap. The model uses piecewise S3 latency and $0.0004 per 1,000 GETs.

| Concurrent reads | Separate | Greedy | MainSolution | Latency reduction vs Separate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3,750.4 ms | 1,454.0 ms | 1,276.5 ms | 66% |
| 16 | 262.8 ms | 194.2 ms | 125.2 ms | 52% |
| 64 | 88.4 ms | 131.2 ms | 60.2 ms | 32% |

These are model predictions. For measured engine performance, use the paper's
Lance experiments. The [fixture manifest](rust/tests/data/sift_paper_pca_4096x128/fixtures_manifest.json)
records source hashes, toolchain, and index settings.

## Verification and benchmarks

```bash
cargo test --locked --release -p warp-planner
python -m pip install pytest
python -m pytest python/tests -q
make release-check

make bench-smoke
make bench
```

The full Criterion sweep takes about 15–25 minutes and writes CSV files to
`target/bench-artifacts/` and HTML to `target/criterion/`.
See [docs/benchmarks.md](docs/benchmarks.md) for the method and output schema.

## Algorithm guarantees

`MainSolution` is exact on small candidate sets. Larger sets use a bounded
adaptive search with a 5% stopping target; that path has no global `(1+ε)`
guarantee. The inner partition solver is exact for each candidate it evaluates.
Brute-force tests cover small inputs, and frozen workloads check for regressions.

## License

Apache-2.0. See [LICENSE](LICENSE).
