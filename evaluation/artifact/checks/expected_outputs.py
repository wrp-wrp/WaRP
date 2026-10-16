#!/usr/bin/env python3
"""After rendering every paper artifact, verify expected_outputs all exist.

Walks `ExperimentSpec.expected_outputs` and asserts each file is
present under `experiments/<id>/`. Size > 0 is treated as a sanity check.

Intended as the final step of `rebuild_paper_artifacts.py --profile
light` or the `--dry-run-full` smoke.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..registry import EXPERIMENTS, ROOT


def main() -> int:
    errors: list[str] = []
    total = 0
    for spec in EXPERIMENTS:
        for rel in spec.expected_outputs:
            total += 1
            abs_path = ROOT / rel
            if not abs_path.exists():
                errors.append(f"{spec.id}: expected output missing: {rel}")
                continue
            size = abs_path.stat().st_size
            if size == 0:
                errors.append(f"{spec.id}: expected output is empty (0 bytes): {rel}")
                continue
            # For PDFs, sanity-check the file header.
            if rel.endswith(".pdf"):
                with abs_path.open("rb") as f:
                    header = f.read(4)
                if header != b"%PDF":
                    errors.append(f"{spec.id}: expected PDF has bad header: {rel}")
    if errors:
        print(f"[fail] {len(errors)} expected-output issues:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"[ok] all {total} expected_outputs present and non-empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
