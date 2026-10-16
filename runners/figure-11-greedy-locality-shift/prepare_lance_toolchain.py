#!/usr/bin/env python3
"""Build and verify Figure 11 Lance examples; supports pure ``--check-only``."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact.lance_toolchain import main


if __name__ == "__main__":
    forwarded = list(sys.argv[1:])
    if (
        os.environ.get("VECIO_SKIP_LANCE_BUILD") == "1"
        and "--check-only" not in forwarded
    ):
        forwarded.append("--verify-existing")
    raise SystemExit(main(["--figure11", *forwarded]))
