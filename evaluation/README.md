# WaRP evaluation

Experiments and frozen measurements for *WaRP: Workload-Aware Request Planning
for Vector Retrieval on Object Storage*.

Use [../planner](../planner/) for the standalone library and [../lance](../lance/)
for the paper's engine implementation. Both are included in this repository.
Run the commands below from `evaluation/`.

## Project structure

```text
artifact/       experiment registry, CLI, and checks
experiments/    frozen measurements, renderers, and paper figures/tables
runners/        measurement programs
src/            shared experiment code
scripts/        setup and experiment entrypoints
docs/           reproduction guide and figure/table map
```

Each experiment stores the paper measurements in `data/paper/` and new
measurements in `data/latest/`. Offline renders go to `_regen/`; renders of
new measurements go to `_latest/`. The committed figures and data stay intact.

## Reproducing the paper

Start with the offline check. Use a single experiment to check the live
pipeline before a full run.

### Offline figures and tables

Install Poppler (`poppler-utils` on Ubuntu, `poppler` on Homebrew), then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-light.txt

python -m artifact check
python -m artifact compare
```

On macOS, set `export TMPDIR=/private/tmp` before running the original tests.
The checks verify file hashes, schemas, runner interfaces, and the experiment
catalog, then regenerate the paper figures and tables. PDF comparisons use a
10% pixel threshold; generated TeX tables must match byte for byte. See the
[repository README](../README.md#known-reproduction-issues) for the Figure 10
layout limitation.

To render selected experiments from frozen measurements:

```bash
bash scripts/run_all.sh --profile light --only figure-07-real-lance-metrics
```

### One live experiment

On the target Linux/EC2 host:

```bash
bash scripts/setup_ec2.sh
cp scripts/run_all.env.example scripts/run_all.env
# Set VECIO_S3_BUCKET and VECIO_S3_REGION in the local env file.
# Supply AWS credentials through the standard provider chain.

bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity --plan
bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity
```

The scripts find the bundled Lance source at `../lance`, check its fingerprint,
build the needed examples, and prepare the selected experiment's data.
Live runs create S3 objects and incur cloud charges.

A CPU-only measurement is also available:

```bash
bash scripts/run_all.sh --only reorder-cost
```

### Full measurements

Inspect the plan before starting:

```bash
bash scripts/run_all.sh --plan
bash scripts/run_all.sh
```

The full run can take several days. It downloads public datasets, builds
matched Lance indexes, runs the registered measurements, and renders the
results from `data/latest/`. Figure 4 needs four network-topology definitions;
see [its runner guide](runners/figure-04-object-store-io/README.md).

The paper campaigns use the fixed Lance source at commit
`c768d004506231f4f5d81f61b9cbc55ccfc904f3`. The checks support both a Git
checkout and a source archive. Keep the bundled source unchanged for a paper
reproduction. The default frozen-data bucket label, `s3://paper-artifact/...`,
is a redacted identifier; use your own bucket for live measurements.

## Find an experiment

```bash
python -m artifact list
python -m artifact show figure-07-real-lance-metrics --parameters
python -m artifact prepare figure-07-real-lance-metrics --dry-run
```

`show --parameters` lists fixed paper parameters separately from infrastructure
inputs such as bucket names and source paths. `--only` accepts comma-separated
experiment IDs and prepares their dependencies.

The [figure/table map](docs/PAPER_ARTIFACTS_MAP.md) links paper numbers to
experiment directories. The [reproduction guide](docs/REPRODUCTION_GUIDE.md)
covers data preparation, index identity checks, and the full measurement
protocols. Figures 2, 3, and 6 are explanatory diagrams; Tables 2 and 3 describe
methods and datasets.

## License and citation

See [LICENSE](LICENSE), [NOTICE](NOTICE), and the repository's
[CITATION.cff](../CITATION.cff). Final publication metadata is pending.
