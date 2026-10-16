# All-dataset QPS-versus-cost runner

This directory owns the measurement phase for
`experiments/figure-09-qps-cost/`. It prepares the additional
1M-scale datasets and runs the closed-loop `Q=4`, `IO=64` campaign.

```bash
python3 -m artifact run figure-09-qps-cost --dry-run
python3 -m artifact run figure-09-qps-cost
python3 -m artifact render figure-09-qps-cost
```

The default campaign is the complete six-dataset paper grid: SIFT1M, GloVe,
DEEP1M, OpenImage, MSong, and Random-1M. Missing datasets are fatal by
default; `--skip-unavailable` is only for exploratory partial runs.

`prepare_datasets.py` writes dataset readiness metadata and `run_data.py`
writes measurement output under:

```text
experiments/figure-09-qps-cost/data/latest/
```

No figure is stored or rendered under `runners/`. The Paper Experiment owns
the renderer, frozen `data/paper/` snapshot, and authoritative PDF.
