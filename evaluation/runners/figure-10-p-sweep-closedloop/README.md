# Closed-loop QPS/latency measurement runner

Collects measurements for
`experiments/figure-10-p-sweep-closedloop/`. The experiment measures how
planner parallelism `P` behaves under serving-style closed-loop load.

Use `python3 -m artifact show figure-10-p-sweep-closedloop --parameters`
for the paper dataset and concurrency settings. The runner compares Separate,
Lance, Greedy, WaRP, and WaRP + Reorder under request budgets.
`--preset sanity` selects a smaller validation run.

Use the repository interface:

```bash
python3 -m artifact run figure-10-p-sweep-closedloop --dry-run
python3 -m artifact run figure-10-p-sweep-closedloop
python3 -m artifact render figure-10-p-sweep-closedloop
```

`run_data.py` exports query ranges, executes the local or remote benchmark,
aggregates repeats, and writes only to
`experiments/figure-10-p-sweep-closedloop/data/latest/`. The render command reads these new measurements. To render the frozen
`data/paper/` inputs instead, add `--profile light`.
