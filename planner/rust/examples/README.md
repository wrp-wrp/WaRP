# Rust examples

| Example | Purpose |
| --- | --- |
| [quickstart.rs](quickstart.rs) | Create requests, plan reads, and evaluate the result |
| [calibrate.rs](calibrate.rs) | Fit a cost model from measured samples |
| [probe_and_plan.rs](probe_and_plan.rs) | Probe an in-memory object store, fit a model, and plan reads; requires `probe` |
| [savings_demo.rs](savings_demo.rs) | Compare request counts, extra bytes, modeled latency, and cost across strategies |
| [paper_table4.rs](paper_table4.rs) | Check synthetic planner overhead for the table's `(nprobe, K)` shapes on the current host |

From `planner/`:

```bash
cargo run --release --example quickstart
cargo run --release --example calibrate
cargo run --release --example probe_and_plan --features probe
cargo run --release --example savings_demo
cargo run --release --example paper_table4
```

`paper_table4` provides a local spot check. Its referenced published values
are context; this command does not reproduce the paper's measurement setup.
Use `make bench-smoke` for a short Criterion run or `make bench` for the full sweep.

## Update golden fixtures

[bless_golden.rs](bless_golden.rs) runs each strategy on the committed FAISS-IVF
fixtures and writes `tests/data/<dist>/expected_plans.json`.
After an intentional algorithm change, run the tool and review the diff:

```bash
make bless
# Or:
cargo run --release --example bless_golden
```

The regression checks are in `tests/golden.rs`.
