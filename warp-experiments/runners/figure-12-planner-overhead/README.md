# Real-data planner-overhead runner

This directory owns the measurement phase for the planner-overhead Paper
Experiment. `run_data.py` exports real query probe lists and cluster layouts,
then invokes the Lance `planner_overhead_real` benchmark. A full run fails if
the benchmark binary is unavailable or if it does not emit the complete
Plain/D&C/Adaptive/WQS schema. Runtime
JSON and CSV outputs live under:

```text
experiments/figure-12-planner-overhead/data/latest/
```

Full runs keep the detailed companion output as `planner_benchmark_raw.csv`
and publish the 18-row canonical `merged_summary.csv` consumed by both the
four-way figure and paper table.

```bash
python3 -m artifact run figure-12-planner-overhead --dry-run
python3 -m artifact run figure-12-planner-overhead
python3 -m artifact render figure-12-planner-overhead
```

Only reviewed inputs are promoted from `data/latest/` to `data/paper/`.
