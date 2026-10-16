#!/usr/bin/env python3
"""Write or verify a deterministic receipt for one paper Lance binary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact.lance_toolchain import validate_lance_source


RECEIPT_SUFFIX = ".warp-build-receipt.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt_path(binary: Path) -> Path:
    binary = binary.expanduser().resolve()
    return binary.with_name(binary.name + RECEIPT_SUFFIX)


def validate_checkout(root: Path) -> str:
    """Backward-compatible commit accessor for callers that only need the pin."""
    try:
        return validate_lance_source(root)["lance_commit"]
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc


def validate_source_identity(root: Path) -> dict[str, str]:
    try:
        return validate_lance_source(root)
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc


def expected_receipt(
    lance_root: Path,
    binary: Path,
    *,
    package: str,
    example: str,
) -> dict[str, object]:
    lance_root = lance_root.expanduser().resolve()
    binary = binary.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"paper Lance binary is missing or not executable: {binary}")
    cargo_lock = lance_root / "Cargo.lock"
    if not cargo_lock.is_file():
        raise ValueError(f"Cargo.lock is missing: {cargo_lock}")
    source_identity = validate_source_identity(lance_root)
    return {
        "schema_version": 2,
        **source_identity,
        "package": package,
        "example": example,
        "profile": "release",
        "binary_sha256": sha256(binary),
        "cargo_lock_sha256": sha256(cargo_lock),
    }


def write_receipt(
    lance_root: Path,
    binary: Path,
    *,
    package: str,
    example: str,
) -> Path:
    binary = binary.expanduser().resolve()
    output = receipt_path(binary)
    payload = expected_receipt(
        lance_root,
        binary,
        package=package,
        example=example,
    )
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def validate_receipt(
    lance_root: Path,
    binary: Path,
    *,
    package: str,
    example: str,
) -> dict[str, object]:
    binary = binary.expanduser().resolve()
    path = receipt_path(binary)
    if not path.is_file():
        raise ValueError(
            f"paper Lance build receipt is missing: {path}; rebuild the example "
            "and generate its receipt"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = expected_receipt(
        lance_root,
        binary,
        package=package,
        example=example,
    )
    if payload != expected:
        changed = sorted(
            key
            for key in set(payload) | set(expected)
            if payload.get(key) != expected.get(key)
        ) if isinstance(payload, dict) else ["document"]
        raise ValueError(f"paper Lance build receipt mismatch: {', '.join(changed)}")
    return {
        **expected,
        "receipt_sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--example", required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.verify_existing:
        identity = validate_receipt(
            args.lance_root,
            args.binary,
            package=args.package,
            example=args.example,
        )
        print(f"[ok] verified build receipt {identity['receipt_sha256']}")
    else:
        path = write_receipt(
            args.lance_root,
            args.binary,
            package=args.package,
            example=args.example,
        )
        print(f"[ok] wrote build receipt {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
