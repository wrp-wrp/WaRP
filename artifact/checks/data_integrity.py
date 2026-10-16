#!/usr/bin/env python3
"""Compute / verify the SHA-256 manifest of every frozen paper input.

Runs in two modes:

  --write     Walk every spec's declared artifact directory and write the authoritative
              manifest to `experiments/DATA_MANIFEST.json`. Commit the
              resulting file along with the data. Run after every data
              change.

  --verify    (default) Re-hash every committed data file and fail if
              any differs from the manifest, or if a file listed in the
              manifest is missing, or if a file exists on disk but is
              not in the manifest.

The manifest is the authoritative snapshot of "which bytes produced the
paper figures" - verifying it is how a reviewer proves they are looking
at the same frozen inputs we rendered the paper from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ..registry import EXPERIMENTS, ROOT


DATA_ROOT = ROOT / "experiments"
MANIFEST_PATH = DATA_ROOT / "DATA_MANIFEST.json"
CHUNK = 1 << 20


def _hash_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def _data_files() -> list[Path]:
    """Hash every declared spec's immutable `data/paper/` snapshot.

    data/latest/ is gitignored (produced by full-mode re-runs) and is not
    part of the authoritative artifact.
    """
    files: set[Path] = set()
    for spec in EXPERIMENTS:
        paper_dir = ROOT / spec.out_dir / "data" / "paper"
        if not paper_dir.is_dir():
            continue
        for path in sorted(paper_dir.rglob("*")):
            if path.is_file():
                files.add(path)
    return sorted(files)


def _build_manifest() -> dict:
    entries: dict[str, dict[str, object]] = {}
    for path in _data_files():
        digest, size = _hash_file(path)
        rel = str(path.relative_to(ROOT))
        entries[rel] = {"sha256": digest, "size": size}
    totals = {
        "file_count": len(entries),
        "total_size_bytes": sum(int(e["size"]) for e in entries.values()),
    }
    return {"manifest_version": 1, "totals": totals, "entries": entries}


def _write(manifest: dict) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    totals = manifest["totals"]
    print(
        f"[ok] wrote {MANIFEST_PATH.relative_to(ROOT)} ({totals['file_count']} files, "
        f"{totals['total_size_bytes'] / (1 << 20):.1f} MiB)"
    )


def _verify() -> int:
    if not MANIFEST_PATH.exists():
        print(f"[fail] manifest missing: {MANIFEST_PATH.relative_to(ROOT)}", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    recorded: dict[str, dict[str, object]] = manifest.get("entries", {})
    errors: list[str] = []

    seen: set[str] = set()
    for rel, entry in sorted(recorded.items()):
        path = ROOT / rel
        if not path.exists():
            errors.append(f"missing on disk: {rel}")
            continue
        digest, size = _hash_file(path)
        if size != int(entry["size"]):
            errors.append(f"size mismatch: {rel} (disk={size}, manifest={entry['size']})")
        elif digest != entry["sha256"]:
            errors.append(f"sha256 mismatch: {rel}")
        seen.add(rel)

    for path in _data_files():
        rel = str(path.relative_to(ROOT))
        if rel not in seen:
            errors.append(f"not tracked in manifest: {rel}")

    if errors:
        print(f"[fail] {len(errors)} discrepancies:", file=sys.stderr)
        for line in errors:
            print(f"  - {line}", file=sys.stderr)
        return 1
    totals = manifest.get("totals", {})
    print(
        f"[ok] verified {totals.get('file_count', len(recorded))} files "
        f"({totals.get('total_size_bytes', 0) / (1 << 20):.1f} MiB) against {MANIFEST_PATH.relative_to(ROOT)}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="rebuild the manifest from the current data")
    group.add_argument("--verify", action="store_true", help="verify the committed data against the manifest (default)")
    args = ap.parse_args()
    if args.write:
        _write(_build_manifest())
        return 0
    return _verify()


if __name__ == "__main__":
    raise SystemExit(main())
