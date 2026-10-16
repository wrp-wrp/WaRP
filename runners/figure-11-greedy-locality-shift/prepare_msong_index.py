#!/usr/bin/env python3
"""Build/validate Figure 11's MSong M=60 pair; supports pure ``--check-only``."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.shared.matched_lance_datasets import main


if __name__ == "__main__":
    manifest_root = (
        ROOT
        / "experiments"
        / "figure-11-greedy-locality-shift"
        / "data"
        / "latest"
        / "dataset-builds"
    )
    raise SystemExit(
        main(
            [
                "--datasets",
                "msong-figure11",
                "--manifest-root",
                str(manifest_root),
                *sys.argv[1:],
            ]
        )
    )
