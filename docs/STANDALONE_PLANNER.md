# Standalone Rust and Python planner

The source at `third_party/warp-planner` can be installed and called without
Lance, AWS credentials, an object store, or the experiment dependencies.
It accepts requested byte ranges, a cost model, and constraints, and returns
merged ranges with assignments back to the original requests. The caller
executes the reads. The Rust crate is `warp-planner` (import `warp_planner`);
the Python distribution is `vecio-planner` (import `vecio_planner`).

## Rust

From this repository's root, run the included minimal example:

```bash
cargo run --locked --release \
  --manifest-path third_party/warp-planner/Cargo.toml --example quickstart
```

In another Rust project, use a path dependency pointing to the bundled core:

```toml
[dependencies]
warp-planner = { path = "/path/to/warp-artifact/third_party/warp-planner/rust" }
```

```rust
use warp_planner::{plan, CostModel, IoRequest, MainSolution, PlanConstraints};

fn main() {
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
}
```

The default core depends only on `serde`. Object-store calibration is an
optional `probe` feature; it is unnecessary for this example.

## Python

Use Python 3.9+ and Rust 1.82+. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
CARGO_PROFILE_RELEASE_STRIP=none python -m pip install ./third_party/warp-planner/python
python third_party/warp-planner/python/examples/quickstart.py
```

This source install builds the PyO3 extension using maturin. The extension
calls the same checked core as the Rust interface:

```python
import vecio_planner as vp

requests = [(0, 0, 8192), (1, 8192, 8192), (2, 5_000_000, 8192)]
cost = dict(latency_per_request=27.0, latency_per_byte=1.0e-5,
            concurrent_parallelism=16.0)
plan = vp.plan(requests, "main_solution", **cost)
print(plan["merged_requests"])
print(vp.evaluate_plan(requests, plan, **cost))
```

The install command keeps native symbols. This avoids a Rust/macOS linker
stripping incompatibility that can produce a `mis-aligned LINKEDIT string
pool` import error on recent macOS versions; it does not change planner code.
Upstream report: https://github.com/rust-lang/rust/issues/157750.

For the existing regression suites:

```bash
cargo test --locked --release \
  --manifest-path third_party/warp-planner/Cargo.toml -p warp-planner
python -m pip install pytest
python -m pytest third_party/warp-planner/python/tests -q
```

Version 0.1.0 is a source release. These instructions do not assume a
published crates.io or PyPI package.

## Implementation boundaries

The paper's Lance experiments use `VecioMergePlanner` in
`third_party/lance/rust/lance-io/src/scheduler/`. The standalone library has
a separately maintained search implementation. Lance has no dependency on
the standalone crate in this release.

The standalone library documents exact search on small candidate sets and a
bounded adaptive search on larger inputs, without claiming a global
`(1 + epsilon)` guarantee for that larger search path. See its
[algorithm guarantees](../third_party/warp-planner/README.md#algorithm-guarantees).
The paper experiments must be reproduced with the pinned Lance implementation;
standalone model predictions and library tests establish a different scope.

This consolidated release preserves both implementations. It adds a single
delivery location and usage instructions without migrating the paper's
engine to a different algorithm implementation.
