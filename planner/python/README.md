# Python binding

`vecio_planner` calls the checked Rust planner through PyO3. Version 0.1.0 is
available from source and requires Python 3.9+. The repository pins the
validated Rust 1.94.0 toolchain; install Rust through rustup.

## Install

From `planner/`, create a virtual environment and install the binding:

```bash
python3 -m venv python/.venv
source python/.venv/bin/activate
CARGO_PROFILE_RELEASE_STRIP=none python -m pip install ./python
python python/examples/quickstart.py
```

For development, install maturin and pytest in that environment, then run
`CARGO_PROFILE_RELEASE_STRIP=none maturin develop --release` from `planner/python/`.
The symbol setting avoids a Rust/macOS linker-stripping import issue.

## API

```python
import vecio_planner as vp

requests = [(0, 0, 8192), (1, 8192, 8192), (2, 5_000_000, 8192)]
cost = dict(latency_per_request=27.0, latency_per_byte=1.0e-5,
            concurrent_parallelism=16.0)
plan = vp.plan(requests, "main_solution", max_requests=16, **cost)
metrics = vp.evaluate_plan(requests, plan, **cost)
```

Each request is `(cluster_id, offset, length)`. A plan contains
`merged_requests`, a list of `(offset, length)` pairs, and `assignments`,
which maps each input request to a merged request.

Cost-model arguments mirror Rust's `CostModel`:

| Argument | Meaning |
| --- | --- |
| `latency_model` | `"linear"` or `"piecewise_linear"` |
| `latency_per_request`, `latency_per_byte` | Linear latency parameters |
| `latency_pw_breaks_bytes`, `latency_pw_overheads_ms`, `latency_pw_rates_ms_per_byte` | Piecewise latency parameters |
| `concurrent_parallelism` | Expected concurrent reads for `MainSolution` |
| `cost_per_request`, `cost_per_byte`, `cost_per_wasted_byte` | Monetary-cost parameters used when scoring a plan |

Planning arguments depend on the strategy:

| Argument | Strategies |
| --- | --- |
| `max_requests`, `exact_requests` | `serial`, `bottleneck`, `main_solution` |
| `greedy_min_gap_bytes`, `greedy_max_span_bytes` | `greedy` |
| `pre_merge_max_gap_bytes`, `pre_merge_max_merged_size_bytes` | `pre_merge` |

Accepted strategies are `separate`, `greedy`, `pre_merge`, `serial`,
`bottleneck`, and `main_solution`. Unknown arguments, or arguments for another
strategy, raise `ValueError`. `max_merge_span` is not a `CostModel` field.

PreMerge defaults to `(0, 0)`, which leaves ranges separate. The opt-in preset
`pre_merge_max_gap_bytes=32*1024` and
`pre_merge_max_merged_size_bytes=1024*1024` uses a 32 KiB gap and 1 MiB span.
It can precede `serial` or `main_solution` for inputs with more than 128 ranges.

## Cost-model calibration

Fit a model from your own measurements:

```python
cal = vp.Calibrator()
for bytes_, ms in my_samples:
    cal.record(bytes_, ms)
cost = cal.to_cost_model(concurrent_parallelism=16.0)
plan = vp.plan(requests, "main_solution", **cost)
```

`fit_linear()` returns `(overhead_ms, ms_per_byte)`. `fit_piecewise()` returns
`None` or a dictionary with `break_bytes`, `overheads_ms`, and `rates_ms_per_byte`.
Network probing is available through the Rust `probe` feature.

## End-to-end demo

The [FAISS example](examples/README.md) builds a local IVF index, plans the
cluster reads, and compares plans using the latency and cost model.
