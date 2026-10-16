# Closed-loop QPS/latency measurement runner

This directory owns the measurement phase for
`experiments/figure-10-p-sweep-closedloop/`. The experiment measures how
planner parallelism `P` behaves under serving-style closed-loop load.

The default protocol covers SIFT1M and GloVe, `nlist=4096`, `nprobe=128`,
multiple query/I/O concurrency settings, strict request budgets, and the
Separate, Lance, Greedy, WaRP, and WaRP + reorder methods. `--preset sanity`
selects a smaller local validation matrix.

Use the repository interface:

```bash
python3 -m artifact run figure-10-p-sweep-closedloop --dry-run
python3 -m artifact run figure-10-p-sweep-closedloop
python3 -m artifact render figure-10-p-sweep-closedloop
```

`run_data.py` exports query ranges, executes the local or remote benchmark,
aggregates repeats, and writes only to
`experiments/figure-10-p-sweep-closedloop/data/latest/`. Rendering is a
separate phase owned by the Paper Experiment and reads the frozen snapshot
under `data/paper/`.
