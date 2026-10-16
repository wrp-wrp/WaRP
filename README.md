# WaRP: Workload-Aware Request Planning for Vector Retrieval on Object Storage

Ruipeng Wan, Xinyu Zeng, Yiming Qiao, and Huanchen Zhang.

**Paper status:** Under shepherding for PVLDB Volume 20 (VLDB 2027).

WaRP groups byte-range reads into object-store requests to reduce request
overhead during vector retrieval. This repository contains the standalone
planner, the Lance implementation used in the paper, and the experiments.

## Contents

| Directory | Purpose |
| --- | --- |
| [planner/](planner/) | Standalone Rust library and Python binding |
| [evaluation/](evaluation/) | Experiments, recorded measurements, and figure/table scripts |
| [lance/](lance/) | Lance fork used by the paper |

## Reproduce the paper

Start each workflow below from the repository root.

### Recreate figures and tables locally

This uses the included measurements and requires no AWS account. Use Python
3.12 and install Poppler with `sudo apt-get install poppler-utils` on Ubuntu
or `brew install poppler` on macOS. On macOS, set `export TMPDIR=/private/tmp`
before running the checks.

```bash
cd evaluation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-light.txt
bash scripts/run_all.sh --profile light
python -m artifact compare
```

Generated files go to each experiment's `_regen/` directory. The comparison
command checks them against the paper figures and tables.

### Rerun experiments on S3

Use a Linux host with AWS credentials and a writable S3 bucket. Start with
the small request-fanout experiment below, setting your bucket and region
in the environment file before running it.

```bash
cd evaluation
bash scripts/setup_ec2.sh
cp scripts/run_all.env.example scripts/run_all.env
# Edit scripts/run_all.env: VECIO_S3_BUCKET and VECIO_S3_REGION.
# Configure AWS credentials through the standard provider chain.

bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity --plan
bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity
```

The scripts build the bundled Lance and prepare the experiment. Measurements
go to `data/latest/`; experiments with figures also write `_latest/`.
The `--plan` option prints the commands without executing them.

To run all experiments, configure [Figure 4's network topologies](evaluation/runners/figure-04-object-store-io/README.md),
then run from `evaluation/`:

```bash
bash scripts/run_all.sh --plan
bash scripts/run_all.sh
```

Full runs download large datasets, incur cloud charges, and can take several
days. Figure 1 uses recorded latency values; its script does not make new
S3 measurements.

We have tested local redraws and the standalone planner on macOS (arm64).
The Linux/S3 workflow has not yet been validated end to end. See
[VALIDATION.md](VALIDATION.md) for the environment, commands, and results.

The [experiment guide](evaluation/README.md) covers data preparation and
individual runs. Use the [figure/table map](evaluation/docs/PAPER_ARTIFACTS_MAP.md)
to find the experiment for a paper result.

## Use the standalone planner

Run the Rust example from the repository root:

```bash
cargo +1.94.0 run --locked --release --manifest-path planner/Cargo.toml --example quickstart
```

For Python, activate a virtual environment, then run:

```bash
CARGO_PROFILE_RELEASE_STRIP=none python -m pip install ./planner/python
python planner/python/examples/quickstart.py
```

The library takes byte ranges, a cost model, and constraints, then returns a
merge plan. Rust and Python share one core. The paper experiments use a
separate implementation in `lance/`. See the [planner guide](planner/README.md)
for the API and algorithm guarantees.

Original component versions, file hashes, and reproduction changes are
recorded in the [source manifest](SOURCE_MANIFEST.json).

## License and citation

See [CITATION.cff](CITATION.cff) to cite this work. The code is licensed under
Apache-2.0; each component includes its LICENSE and NOTICE files. External
datasets and dependencies retain their own licenses.
