# Reorder Cost Cluster Sweep

This directory captures the frozen reorder-timing sweep used by the paper's reorder-cost and figure-13-reordering-methods figures.

Files:

- `run_data.py`: recomputes the surrogate cluster-centroid sweep from the local dataset code.
- `generate.py`: regenerates the paper figure from the frozen summary CSV.
- `run.py`: lightweight local figure entrypoint.
- `figure_reorder_cost_cluster_sweep.csv`: the frozen summary currently used by the paper.

Run the local figure rebuild with:

```bash
python3 runners/reorder-cost/run.py
```
