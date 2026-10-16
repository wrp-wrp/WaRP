"""Verify the frozen sources bundled in the consolidated paper release."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIRS = {".git", "target", "__pycache__", ".venv", ".pytest_cache",
                  ".mypy_cache", ".ruff_cache"}


def verify(root: Path = ROOT) -> list[str]:
    manifest_path = root / "SOURCE_MANIFEST.json"
    if not manifest_path.is_file():
        return ["SOURCE_MANIFEST.json is missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return [f"cannot read source manifest: {exc}"]
    errors: list[str] = []
    for source in manifest["sources"]:
        relative_root = Path(source["path"])
        if relative_root.is_absolute() or ".." in relative_root.parts:
            errors.append(f"unsafe source root: {source['id']}")
            continue
        source_root = root / relative_root
        adapted = set(source.get("adapted_files", []))
        expected = source["files"]
        for relative, entry in expected.items():
            rel = Path(relative)
            if rel.is_absolute() or ".." in rel.parts:
                errors.append(f"unsafe source path: {source['id']}:{relative}")
                continue
            path = source_root / rel
            if not path.is_file() or path.is_symlink():
                errors.append(f"source file missing or replaced by link: {path.relative_to(root)}")
                continue
            if relative in adapted:
                continue
            payload = path.read_bytes()
            if (len(payload) != entry["size"] or
                    hashlib.sha256(payload).hexdigest() != entry["sha256"]):
                errors.append(f"source bytes differ: {path.relative_to(root)}")
        # Extra source/configuration files cannot silently enter fixed snapshots.
        if source["path"] != ".":
            for path in source_root.rglob("*"):
                rel = path.relative_to(source_root)
                if any(part in GENERATED_DIRS for part in rel.parts):
                    continue
                if path.is_file() and rel.as_posix() not in expected:
                    errors.append(f"unexpected bundled file: {path.relative_to(root)}")
    return errors


def main() -> int:
    errors = verify()
    if errors:
        for error in errors:
            print(f"[fail] {error}", file=sys.stderr)
        return 1
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for source in manifest["sources"]:
        print(f"[ok] {source['id']}: {len(source['files'])} original files, "
              f"{len(source.get('adapted_files', []))} declared packaging adaptations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
