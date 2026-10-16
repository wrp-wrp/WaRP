# [VLDB2027] WaRP: Workload-Aware Request Planning for Vector Retrieval on Object Storage

Ruipeng Wan, Xinyu Zeng, Yiming Qiao, and Huanchen Zhang.

Code and experiments for WaRP. The planner groups byte-range reads into
object-store requests. The paper implementation integrates request planning
and cluster reordering into Lance.

## Contents

| Directory | Purpose |
| --- | --- |
| [planner/](planner/) | Standalone Rust library and Python binding |
| [evaluation/](evaluation/) | Experiments, frozen measurements, and figure/table scripts |
| [lance/](lance/) | Lance fork used by the paper |

Code and frozen data retain the submitted versions. Documentation changes
and the original file hashes are recorded in `SOURCE_MANIFEST.json`.

## Reproduce figures and tables

```bash
cd evaluation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-light.txt
python -m artifact check
python -m artifact compare
```

Install Poppler first: `sudo apt-get install poppler-utils` on Ubuntu or
`brew install poppler` on macOS. On macOS, set `export TMPDIR=/private/tmp`
before the checks to account for the original path tests.

These commands verify the frozen measurements and regenerate figures and
tables locally. See [evaluation/README.md](evaluation/README.md) for live
measurements and [the artifact map](evaluation/docs/PAPER_ARTIFACTS_MAP.md)
to locate a paper figure or table.

## Call the planner

Run the Rust example from the repository root:

```bash
cargo run --locked --release --manifest-path planner/Cargo.toml --example quickstart
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

## Run live experiments

Use the instructions in [evaluation/README.md](evaluation/README.md).
The scripts discover `../lance` and check its fixed source fingerprint.
The reported environment is EC2 `c7gn.8xlarge` with same-region S3 in
`ap-northeast-1`. Full runs download large datasets, write S3 objects, and
can take several days.

## Versions and validation

| Component | Submitted commit |
| --- | --- |
| Evaluation | `23b54eea35c60605e690ae3d2f63142545d60400` |
| Planner | `6e8c3fe04ff8e9b4ce4eb477e965d6ed315e41c3` |
| Lance | `c768d004506231f4f5d81f61b9cbc55ccfc904f3` |

Local validation covered 17 offline checks, 88 experiment-framework tests,
121 Rust tests/doctests, and 25 Python tests. The frozen inputs and the Lance
build-source fingerprint were checked. Live S3 measurements and the full
Lance benchmark build remain pending.

## Known reproduction issues

The submitted Figure 10 renderer produces a different aspect ratio from the
saved PDF. The original image comparison can accept this difference because
it resizes images. The earlier
[packaged-layout branch](https://github.com/wrp-wrp/WaRP/tree/archive/packaged-layout-2026-10-16)
contains the layout repair and a stricter comparison. The renderer on `main`
retains the submitted version.

## License and citation

See `CITATION.cff` for the paper citation. Final PVLDB metadata is pending.
Each component retains its LICENSE and NOTICE files. External datasets and
dependencies retain their own licenses.
