# SIFT1M real-S3 sensitivity runner

This runner owns the measurement phase for the Paper Experiment at
`experiments/figure-14-sift1m-sensitivity/`. Run it through the repository
interface:

```bash
python3 -m artifact run figure-14-sift1m-sensitivity --dry-run
python3 -m artifact run figure-14-sift1m-sensitivity
```

`run_data.py` is the orchestration entrypoint. It executes the registered
`nprobe` and cluster-granularity settings, computes recall, and writes the
canonical runner output under:

```text
experiments/figure-14-sift1m-sensitivity/data/latest/
```

The implementation files in this directory are internal to that entrypoint:

- `stress_runner.py` runs one real-S3 stress setting;
- `compute_ivf_recall.py` computes the held-out recall result;
- `generate_table.py` and `generate_matrix.py` summarize a completed rerun.

The paper renderer does not read runner-local scratch. It reads the frozen,
checksummed snapshot under the Paper Experiment's `data/paper/` directory.
Promote new evidence deliberately with `python3 -m artifact freeze` only after
reviewing a completed campaign.
