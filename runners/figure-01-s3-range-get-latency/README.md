# Figure 1

S3 Range-GET latency decomposition used by the introduction figure.

Files:

- `run_data.py`: materializes the frozen latency breakdown CSV used by the paper.
- `generate.py`: regenerates the paper figure from the frozen CSV.
- `run.py`: runs both steps.
- `data.csv`: the frozen breakdown currently used by the paper.

Run:

```bash
python3 runners/figure-01-s3-range-get-latency/run.py
```
