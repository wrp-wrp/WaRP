#!/usr/bin/env python3
"""Extract + verify schemas for every paper-snapshot data file.

Motivation
----------
`python3 -m artifact.checks.runner_render_schema` proves that runner output paths
line up with render input paths *by name*. This tool goes one level
deeper: it fingerprints the **shape** of each paper-snapshot file (the
set of CSV columns or top-level JSON keys) and persists those
fingerprints in `docs/SCHEMA_FINGERPRINTS.json`. On re-run it re-reads
the paper snapshot and flags any drift.

Why this matters on EC2
-----------------------
When a reviewer triggers full mode on EC2, the runner writes to
`data/latest/`. If the runner is intact, its output should match the
fingerprint. If the fingerprint doesn't match (e.g., a column was
renamed upstream), the render will crash with a confusing KeyError.

Usage
-----
  --fingerprint   rebuild `docs/SCHEMA_FINGERPRINTS.json` from the
                  current paper snapshot. Commit the result.
  --verify        (default) re-fingerprint paper snapshot and compare
                  against the committed file; fail on drift.
  --check-latest  like --verify but fingerprints `data/latest/` instead
                  of `data/paper/`. Only useful after a runner has
                  actually run. Helps EC2 post-run validation.

Scope: limited to files listed in each spec's `expected_runner_outputs`
plus a small set of extra paper-snapshot files that renders hard-code.
We don't fingerprint every byte under `data/paper/`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ..registry import EXPERIMENTS, ROOT


FINGERPRINT_PATH = ROOT / "docs" / "SCHEMA_FINGERPRINTS.json"

# Extra paper-snapshot files the renders read that are not covered by a
# runner's expected_runner_outputs (derived-figure inputs that come from
# other specs or live in subdirectories). Listed explicitly so drift in
# these is also caught.
EXTRA_PAPER_FILES = (
    # closed-loop / greedy reuse the same rich combined JSON
    "experiments/figure-10-p-sweep-closedloop/data/paper/data_gap_env3_v3/aggregated_results.json",
    # reorder-cost frozen CSV feeds figure-13-reordering-methods
    "experiments/figure-13-reordering-methods/stages/reorder-cost/data/paper/figure_reorder_cost_cluster_sweep.csv",
    # real-s3-tiny-request per-iop CSV
    "experiments/figure-04-object-store-io/stages/real-s3-tiny-request-fanout-sanity/data/paper/fanout_001_per_iop.csv",
    # Figure 7's current hash-validating renderer reads the frozen performance/cache
    # display table and both campaign summaries directly.
    "experiments/figure-07-real-lance-metrics/data/paper/figure7_source.csv",
    "experiments/figure-07-real-lance-metrics/data/paper/figure7_performance_summary.json",
    "experiments/figure-07-real-lance-metrics/data/paper/figure7_cache_summary.json",
    # Payload-accounting table uses the same Figure 7 campaign with frozen AWS prices.
    "experiments/figure-07-real-lance-metrics/data/paper/aws_pricing.json",
)


def _fingerprint(path: Path) -> dict[str, object]:
    """Return a small signature of `path`'s shape.

    For CSV: column list (order-preserving).
    For JSON: sorted list of top-level keys, plus keys of the first
    element if the top is a list of dicts.
    """
    suffix = path.suffix.lower()
    fp: dict[str, object] = {"type": suffix.lstrip(".")}
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
            row_count = sum(1 for _ in reader)
        fp["columns"] = header
        fp["row_count"] = row_count
        return fp
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            fp["top_keys"] = sorted(payload.keys())
            if "rows" in payload and isinstance(payload["rows"], list) and payload["rows"]:
                first = payload["rows"][0]
                if isinstance(first, dict):
                    fp["row_keys"] = sorted(first.keys())
                    fp["row_count"] = len(payload["rows"])
        elif isinstance(payload, list):
            fp["top_kind"] = "list"
            fp["list_len"] = len(payload)
            if payload and isinstance(payload[0], dict):
                fp["row_keys"] = sorted(payload[0].keys())
        return fp
    fp["unsupported"] = True
    return fp


def _tracked_files() -> list[str]:
    files: list[str] = []
    for spec in EXPERIMENTS:
        for out in spec.expected_runner_outputs:
            # Paper-snapshot counterpart of the runner output path.
            files.append(out.replace("/data/latest/", "/data/paper/"))
    files.extend(EXTRA_PAPER_FILES)
    # Preserve order, dedupe
    seen: set[str] = set()
    uniq: list[str] = []
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        uniq.append(f)
    return uniq


def _fingerprint_slot(slot: str) -> dict[str, object]:
    """Produce fingerprints from either 'paper' or 'latest' slot."""
    entries: dict[str, object] = {}
    missing: list[str] = []
    for rel in _tracked_files():
        if slot == "latest":
            rel = rel.replace("/data/paper/", "/data/latest/")
        abs_path = ROOT / rel
        if not abs_path.exists():
            missing.append(rel)
            continue
        entries[rel] = _fingerprint(abs_path)
    return {"entries": entries, "missing": missing}


def _write_fingerprints() -> int:
    payload = _fingerprint_slot("paper")
    if payload["missing"]:
        print(f"[warn] {len(payload['missing'])} paper-snapshot files are missing and cannot be fingerprinted:",
              file=sys.stderr)
        for rel in payload["missing"]:
            print(f"  - {rel}", file=sys.stderr)
    FINGERPRINT_PATH.write_text(
        json.dumps(
            {"version": 1, "entries": payload["entries"]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote {FINGERPRINT_PATH.relative_to(ROOT)} ({len(payload['entries'])} entries)")
    return 0


def _compare(expected: dict, actual: dict, slot: str) -> list[str]:
    errors: list[str] = []
    for rel, ref_fp in expected.items():
        got = actual.get(rel)
        if got is None:
            errors.append(f"{slot}: missing file: {rel}")
            continue
        for key in ("type", "columns", "top_keys", "row_keys", "top_kind"):
            if key in ref_fp and ref_fp[key] != got.get(key):
                errors.append(f"{slot}: schema drift on {rel}: {key} differs "
                              f"(expected {ref_fp[key]!r}, got {got.get(key)!r})")
    for rel in actual:
        if rel not in expected:
            errors.append(f"{slot}: unexpected new file not in fingerprints: {rel}")
    return errors


def _verify(slot: str) -> int:
    if not FINGERPRINT_PATH.exists():
        print(f"[fail] fingerprints missing: {FINGERPRINT_PATH.relative_to(ROOT)}", file=sys.stderr)
        return 1
    expected = json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8")).get("entries", {})
    actual_slot = _fingerprint_slot(slot)
    # Rewrite actual entries' keys to match the fingerprint keys (which are the
    # /data/paper/ path form).
    canonical_actual: dict[str, object] = {}
    for rel, fp in actual_slot["entries"].items():
        canonical = rel.replace("/data/latest/", "/data/paper/")
        canonical_actual[canonical] = fp

    # For --check-latest, compare only the files that have actually been
    # produced. data/latest/ is populated incrementally by whichever
    # runners have been executed; flagging every non-executed spec as
    # "missing" is noise, not a real error.
    if slot == "latest":
        # Drop "missing on disk" from the comparison and limit schema
        # drift checks to files that exist.
        filtered_expected = {k: v for k, v in expected.items() if k in canonical_actual}
        errors = _compare(filtered_expected, canonical_actual, slot)
        not_yet_run = [k for k in expected if k not in canonical_actual]
        if not_yet_run:
            print(f"[info] {len(not_yet_run)} paper files have no latest/ counterpart yet "
                  f"(skipped). Run the corresponding runner to populate them.")
        print(f"[ok] schema fingerprint check against latest slot "
              f"({len(filtered_expected)} present / {len(expected)} total).")
        return 0 if not errors else 1
    else:
        errors = _compare(expected, canonical_actual, slot)
        if actual_slot["missing"]:
            for rel in actual_slot["missing"]:
                canonical = rel.replace("/data/latest/", "/data/paper/")
                if canonical in expected:
                    errors.append(f"{slot}: missing on disk: {rel}")
    if errors:
        print(f"[fail] {len(errors)} schema issues ({slot}):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"[ok] schema fingerprint check against {slot} slot ({len(expected)} files).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--fingerprint", action="store_true",
                     help="rebuild docs/SCHEMA_FINGERPRINTS.json from data/paper/")
    grp.add_argument("--verify", action="store_true",
                     help="verify data/paper/ matches committed fingerprints (default)")
    grp.add_argument("--check-latest", action="store_true",
                     help="verify data/latest/ matches committed fingerprints (post-run)")
    args = ap.parse_args()
    if args.fingerprint:
        return _write_fingerprints()
    if args.check_latest:
        return _verify("latest")
    return _verify("paper")


if __name__ == "__main__":
    raise SystemExit(main())
