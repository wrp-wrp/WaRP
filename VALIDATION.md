# Local validation

Validated on macOS arm64 on 2026-10-16 with Python 3.12.14 and the pinned
Rust 1.94.0 toolchain.

| Check | Result |
| --- | --- |
| `cd evaluation && python -m artifact check` | 17 checks passed |
| `cd evaluation && python -m artifact compare` | 12 PDFs passed; 3 TeX tables matched exactly |
| `cd evaluation && python -m unittest discover -s tests -v` | 93 framework tests passed |
| `cd planner && cargo test --locked --release -p warp-planner` | 121 tests and doctests passed |
| `python -m pytest planner/python/tests -q` | 25 tests passed |
| `cd planner && make release-check` | 13 release-check tests and the metadata check passed |
| Rust and Python quickstarts | Passed after rebuilding the Python extension |
| Frozen-data integrity | All 191 inputs matched their recorded hashes |
| Lance source check | Matched the pinned build-source fingerprint |

On macOS, the original path tests require `TMPDIR=/private/tmp`.
The Python extension was built with `CARGO_PROFILE_RELEASE_STRIP=none`.

## Source versions

For reproduction, use a release or commit of this unified repository. The
table records the original source snapshots from which it was assembled.

| Component | Original commit |
| --- | --- |
| Evaluation | `23b54eea35c60605e690ae3d2f63142545d60400` |
| Planner | `6e8c3fe04ff8e9b4ce4eb477e965d6ed315e41c3` |
| Lance | `c768d004506231f4f5d81f61b9cbc55ccfc904f3` |

## Reproduction fixes

- Figure 10 now renders at the paper PDF's size, `500.76 × 279.906 pt`.
  Its pixel difference was 0.000% in this environment. The earlier renderer
  produced a shorter `500.76 × 201.282 pt` page.
- PDF comparison rejects aspect-ratio differences above 2% before resizing.
  A regression test rejected a vertically squeezed chart after the fix and
  demonstrated that the earlier comparator accepted it. Resolution changes
  with the same aspect ratio remain supported.
- The Linux bootstrap installs `protobuf-compiler`, `pkg-config`, and
  `libssl-dev`. Its missing-`protoc` check was exercised with that executable
  excluded from `PATH`; shell syntax and the help entrypoint also passed.
- Planner toolchain requirements, the tiny S3 stage's working directory,
  and Figure 1's fixed-data export are documented explicitly.

`SOURCE_MANIFEST.json` retains the original file hashes and records each
documentation update, source fix, and added file. Frozen measurements and
the Lance implementation were not changed.

## Repository links

Citation, Cargo, and Python package metadata point to the consolidated
repository and its component README files. The privacy checks allow this
artifact repository and continue to reject other author repositories,
including names that share the `WaRP` prefix.

## Remaining validation

The full Lance build on a clean Linux host, live S3 measurements, and full
paper campaigns have not been rerun for this release. Local checks verify
the frozen-data workflow and standalone planner; they do not establish
live end-to-end performance. Use the steps in [evaluation/README.md](evaluation/README.md)
when running those campaigns.
