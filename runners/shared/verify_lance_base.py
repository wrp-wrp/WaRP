#!/usr/bin/env python3
"""Verify that the configured companion checkout is at the public paper release."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact.campaigns import LANCE_BASE_COMMIT
from artifact.environment import discover_lance_root
from artifact.lance_toolchain import verify_lance_history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-root", default="")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Accepted for orchestrator compatibility; verification is always read-only.",
    )
    args = parser.parse_args()
    lance_root = discover_lance_root(args.lance_root)
    if lance_root is None:
        raise SystemExit("companion Lance checkout not found; set VECIO_LANCE_ROOT")
    identity = verify_lance_history(lance_root)
    print(
        "[ok] companion source is public paper release "
        f"{LANCE_BASE_COMMIT} ({identity['source_mode']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
