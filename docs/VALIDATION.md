# Local packaging validation

Recorded on 2026-10-16 on macOS 27.0 arm64, Python 3.12.14, Rust 1.91.1.
This record covers local source delivery, offline rendering, and the
standalone library. It does not report new object-store measurements.

## Checks and results

| Check | Result |
| --- | --- |
| Original source snapshots versus fixed GitHub commit trees | All 386 experiment, 1,419 Lance, and 196 planner files matched their Git blob hashes before declared adaptations |
| Bundled algorithm source snapshots | Lance and standalone planner files retain their original bytes |
| Frozen paper inputs | All 191 files matched the recorded sizes and SHA-256 hashes |
| Experiment repository unit tests | 88 passed |
| Offline experiment pipeline | All 18 smoke checks passed, including full paper-artifact rendering |
| Paper artifact comparison | 12 PDFs passed the existing 10% pixel threshold and the new 2% aspect-ratio check; all 3 TeX tables matched byte for byte |
| Largest PDF pixel difference | 4.168%; the unchanged frozen input checks also passed |
| Figure 10 after layout repair | 0.000% pixel difference; both PDF pages are 500.76 by 279.90625 points |
| Standalone Rust regression suites and doctests | 121 passed, 0 failed, 0 ignored |
| Independent Rust consumer | A separate Cargo project imported the bundled crate by path and produced a valid merge plan without Lance or S3 |
| Python source installation | Native wheel built and installed into an isolated virtual environment |
| Python regression suite | 25 passed |
| Python minimal example | Passed without Lance or S3; assignments were `[0, 0, 1]` for the included three-range example |
| Bundled Lance identity gate | Passed: paper commit and complete build-source fingerprint matched |

## Explicit adaptations

`SOURCE_MANIFEST.json` records the original component revisions, file hashes,
file modes, and the first-party files adapted for this package. Changes are
limited to unified documentation/attribution, including bundled sources in
Git, source verification, release-check scope, two macOS test-path fixes,
and the following Figure 10 rendering repair.

The original Figure 10 renderer generated a shorter figure than the
committed paper PDF. The previous pixel comparison resized it, allowing
the layout drift to pass. The comparison now rejects PDF aspect-ratio
changes greater than 2% before resizing. It rejected the original distorted
regeneration (1044 by 420 pixels versus 1044 by 584). Restoring subplot
height and spacing reproduced the saved figure at 0.000% pixel difference.
No measurement, numerical curve computation, or optimizer code was changed.

The two first-party path tests now resolve their temporary directories
before comparing them with production paths, accounting for macOS directory
aliases while retaining the original assertions.

The vendored source snapshots remain byte-identical. Their upstream public
contacts, example credentials, and test object UUIDs are validated by source
integrity rather than first-party anonymity/workflow naming heuristics.
Unexpected or modified bundled files fail the source-integrity check.

## Python/macOS build compatibility

The first native-wheel build compiled but failed to import with
`mis-aligned LINKEDIT string pool`. Rebuilding with
`CARGO_PROFILE_RELEASE_STRIP=none` produced a working extension, as verified
by the minimal example and all 25 Python tests. The installation guide
includes that environment setting. Planner source was not changed.
See the [upstream Rust report](https://github.com/rust-lang/rust/issues/157750).

## Reproduction environment

The local rendering environment used numpy 2.3.5, pandas 2.2.3, scipy 1.18.1,
scikit-learn 1.9.1, matplotlib 3.10.9, PyYAML 6.0.3, and Pillow 12.3.0.
The Python library build used maturin 1.15.0; its tests used pytest 9.1.1.
These versions are a record of this validation, not a claim that other
versions allowed by the requirements files were tested.

## Remaining live validation

The paper Lance benchmark binaries have not been compiled or executed as
part of this local packaging validation. Live S3 smoke measurements,
representative end-to-end performance/recall comparison, public input-data
downloads, and complete heavyweight reruns remain to be verified on the
target environment. The source and original fixed protocols are included.

The permanent source repository is [wrp-wrp/WaRP-VLDB2027](https://github.com/wrp-wrp/WaRP-VLDB2027).
Final publication metadata will be added when assigned. This consolidated
artifact has not yet been submitted to CMT for the new availability check.
