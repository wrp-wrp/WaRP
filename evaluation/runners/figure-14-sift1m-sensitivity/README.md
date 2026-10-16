# SIFT1M real-S3 sensitivity runner

Collects measurements for
`experiments/figure-14-sift1m-sensitivity/`. Run it through the repository
interface:

```bash
python3 -m artifact run figure-14-sift1m-sensitivity --dry-run
python3 -m artifact run figure-14-sift1m-sensitivity
```

`run_data.py` is the orchestration entrypoint. It executes the registered
`nprobe` and cluster-granularity settings, computes recall, and writes the
results under:

```text
experiments/figure-14-sift1m-sensitivity/data/latest/
```

The implementation files in this directory are internal to that entrypoint:

- `stress_runner.py` runs one real-S3 stress setting;
- `compute_ivf_recall.py` computes the held-out recall result;
- `generate_table.py` and `generate_matrix.py` summarize a completed rerun.

Use `python3 -m artifact render figure-14-sift1m-sensitivity` to render the
new measurements, or add `--profile light` for the frozen paper inputs.
See the reproduction guide before replacing any frozen measurements.
