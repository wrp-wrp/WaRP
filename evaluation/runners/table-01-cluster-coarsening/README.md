# Table 1: cluster granularity at matched recall

Compares fine and coarse IVF clustering on SIFT1M at matched recall.
The table reports requests per query and candidate counts alongside recall.
Larger clusters can reduce requests while increasing candidate processing.

## Setup

The comparison uses `nlist=4096` and `nlist=1024`. The tuning step selects a
recall-matched `nprobe` for each setting.

- `run_data.py tune` finds the recall-matched pair.
- `run_data.py plan` writes the two-row manifest and optional pipeline configs.
- `run_data.py summarize` combines the ANN and runtime metrics.
- `generate.py` renders the CSV and TeX table.

Run the commands below from `evaluation/`.

### Workflow

1. Find a recall-matched pair.

```bash
python3 runners/table-01-cluster-coarsening/run_data.py \
  tune \
  --dataset sift1m \
  --fine-nlist 4096 \
  --coarse-nlist 1024 \
  --fine-nprobes 128,160,192,224,256 \
  --coarse-nprobes 32,48,64,80,96,112,128 \
  --queries 300 \
  --query-seed 12345 \
  --train-size 100000 \
  --recall-k 10
```

2. Freeze the two-row design and optionally emit configs.

```bash
python3 runners/table-01-cluster-coarsening/run_data.py \
  plan \
  --dataset sift1m \
  --fine-nlist 4096 \
  --coarse-nlist 1024 \
  --emit-configs
```

3. Run the two configs and fill the `run_dir` fields in:

- `experiments/table-01-cluster-coarsening/data/latest/manifest.json`

4. Summarize the final table inputs.

```bash
python3 runners/table-01-cluster-coarsening/run_data.py \
  summarize \
  --manifest experiments/table-01-cluster-coarsening/data/latest/manifest.json
```

Omitting `--manifest` is an offline reproduction path: it summarizes the
frozen paper manifest into `data/latest/` without changing the paper snapshot.

5. Render the intro table.

```bash
python3 runners/table-01-cluster-coarsening/generate.py
```

### Output files

After `summarize`:

- `experiments/table-01-cluster-coarsening/data/latest/summary.json`
- `experiments/table-01-cluster-coarsening/data/latest/intro_cluster_coarsening_sanity_check.csv`

After `generate.py`:

- `experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.csv`
- `experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.tex`
