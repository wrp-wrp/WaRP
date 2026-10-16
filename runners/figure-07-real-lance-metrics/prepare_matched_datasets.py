#!/usr/bin/env python3
"""Build/validate matched datasets; supports ``--check-only`` and ``--download-public-data``."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.shared.matched_lance_datasets import main


if __name__ == "__main__":
    raise SystemExit(main())
