# WaRP: Workload-Aware Request Planning for Vector Retrieval on Object Storage

Ruipeng Wan, Xinyu Zeng, Yiming Qiao, and Huanchen Zhang.

Paper status: **Accept with Shepherding** for PVLDB Volume 20 (VLDB 2027),
with the artifact availability check pending.

Code and experiments for WaRP. The planner groups byte-range reads into
object-store requests. The paper implementation integrates request planning
and cluster reordering into Lance.

## Contents

| Directory | Purpose |
| --- | --- |
| [planner/](planner/) | Standalone Rust library and Python binding |
| [evaluation/](evaluation/) | Experiments, frozen measurements, and figure/table scripts |
| [lance/](lance/) | Lance fork used by the paper |

The submitted code and measurements are included here. Reproduction fixes,
documentation updates, and original file hashes are recorded in
`SOURCE_MANIFEST.json`. Frozen measurements are unchanged.

## Reproduce the paper

Each workflow below starts at the repository root.

### Local redraw: frozen measurements, no AWS required

Regenerate the paper figures and tables from the bundled measurements.
Use Python 3.12 and install Poppler: `sudo apt-get install poppler-utils` on Ubuntu or
`brew install poppler` on macOS. On macOS, also set `export TMPDIR=/private/tmp`.

```bash
cd evaluation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-light.txt
bash scripts/run_all.sh --profile light
python -m artifact compare
```

The script checks frozen inputs and writes regenerated figures and tables to
each experiment's `_regen/` directory. This workflow has passed local validation.

### Run measurements: Linux and a writable S3 bucket

Start with one S3 experiment. Configure the bucket, region, and AWS credentials
before running the measurement command.

```bash
cd evaluation
bash scripts/setup_ec2.sh
cp scripts/run_all.env.example scripts/run_all.env
# Edit scripts/run_all.env: VECIO_S3_BUCKET and VECIO_S3_REGION.
# Configure AWS credentials through the standard provider chain.

bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity --plan
bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity
```

The scripts build the bundled Lance and prepare the selected experiment.
Measurements go to its `data/latest/`; figure runs also write `_latest/`.
Linux/S3 execution still awaits validation.

For all experiments, configure [Figure 4's network topologies](evaluation/runners/figure-04-object-store-io/README.md),
then run these commands from `evaluation/`:

```bash
bash scripts/run_all.sh --plan
bash scripts/run_all.sh
```

Full runs incur cloud charges, download large datasets, and can take several days.
See the [experiment guide](evaluation/README.md) for setup details and the
[figure/table map](evaluation/docs/PAPER_ARTIFACTS_MAP.md) to select an experiment.

## Call the planner

Run the Rust example from the repository root:

```bash
cargo +1.94.0 run --locked --release --manifest-path planner/Cargo.toml --example quickstart
```

For Python, activate a virtual environment, then run:

```bash
CARGO_PROFILE_RELEASE_STRIP=none python -m pip install ./planner/python
python planner/python/examples/quickstart.py
```

The standalone library accepts byte ranges, a cost model, and constraints,
then returns a merge plan. Rust and Python share one core. The paper's Lance
experiments use a separate implementation preserved in `lance/`.
See [planner/README.md](planner/README.md) for the API and guarantee scope.

## Versions and validation

| Component | Submitted commit |
| --- | --- |
| Evaluation | `23b54eea35c60605e690ae3d2f63142545d60400` |
| Planner | `6e8c3fe04ff8e9b4ce4eb477e965d6ed315e41c3` |
| Lance | `c768d004506231f4f5d81f61b9cbc55ccfc904f3` |

Local validation covered 17 offline checks, 93 experiment-framework tests,
121 Rust tests/doctests, and 25 Python tests. The frozen inputs and the Lance
build-source fingerprint were checked. Live S3 measurements and the full
Lance benchmark build remain pending. See [VALIDATION.md](VALIDATION.md)
for the commands, environment, and validation limits.

## Reproduction notes

The Figure 10 renderer now uses the paper's layout. PDF comparison rejects
aspect-ratio differences above 2% before resizing images. Figure 1 exports
recorded measurement values; its run command does not collect new S3 data.

## License and citation

See `CITATION.cff` for the paper citation. Final PVLDB metadata is pending.
Each component retains its LICENSE and NOTICE files. External datasets and
dependencies retain their own licenses.
