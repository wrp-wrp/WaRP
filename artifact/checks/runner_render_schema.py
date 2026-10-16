#!/usr/bin/env python3
"""Static check: every runner output ends up at a path that a render
consumes.

For each spec with a runner:
  1. translate its `expected_runner_outputs` entries from the full-mode
     `data/latest/...` paths into the corresponding `data/paper/...`
     paths that the render scripts read when no re-run has happened yet.
  2. verify that the translated paths actually exist under the spec's
     declared artifact directory (they came from the paper snapshot).
  3. if any render script hard-codes a path under `data/paper/` that is
     never promised by a runner's `expected_runner_outputs`, flag it so
     we either extend the runner manifest or the render has no way of
     being re-generated end-to-end.

This tool never invokes the runners themselves, so it works on a laptop
without Lance or S3.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from ..registry import EXPERIMENTS, ROOT


def main() -> int:
    errors: list[str] = []
    promised_paper_paths: set[str] = set()

    for spec in EXPERIMENTS:
        if not spec.expected_runner_outputs:
            continue
        for latest_path in spec.expected_runner_outputs:
            if "/data/latest/" not in latest_path:
                errors.append(
                    f"{spec.id}: expected_runner_outputs entry does not live under data/latest/: {latest_path}"
                )
                continue
            paper_path = latest_path.replace("/data/latest/", "/data/paper/")
            promised_paper_paths.add(paper_path)
            abs_paper = ROOT / paper_path
            if not abs_paper.exists():
                errors.append(
                    f"{spec.id}: paper snapshot missing for promised runner output: {paper_path}"
                )

    # Scan every spec-owned render script for hardcoded data/paper/<file> paths.
    # Anything a render reads that is NOT in promised_paper_paths is a
    # render-only-from-snapshot dependency (= fine if the paper imports
    # data we do not regenerate, but we want to flag it so it is
    # intentional).
    path_re = re.compile(
        r'"((?:experiments|tables)/[a-z0-9_./-]+/data/paper/[a-z0-9_./-]+)"'
    )
    render_scripts = {
        ROOT / part
        for spec in EXPERIMENTS
        for command in spec.render.commands
        for part in command.argv
        if part.endswith(".py")
    }
    unpromised: dict[str, list[str]] = {}
    for render in sorted(render_scripts):
        text = render.read_text()
        for m in path_re.finditer(text):
            rel = m.group(1)
            if rel in promised_paper_paths:
                continue
            unpromised.setdefault(rel, []).append(render.relative_to(ROOT).as_posix())

    if unpromised:
        print("[info] render-side paper-snapshot paths not covered by any runner's expected_runner_outputs:")
        for rel in sorted(unpromised):
            readers = ", ".join(sorted(unpromised[rel]))
            print(f"  - {rel}  (read by: {readers})")
        print("      ^ these are paper-snapshot-only inputs. Fine if intentional (e.g.")
        print("        derived figures like greedy / figure-13-reordering-methods that reuse another")
        print("        runner's output); otherwise add the spec's expected_runner_outputs.")

    if errors:
        print("\n[fail] schema check:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"\n[ok] schema check: {len(promised_paper_paths)} promised paths all present in data/paper/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
