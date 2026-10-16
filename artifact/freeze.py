"""Refresh integrity and schema metadata after approving new frozen evidence."""

from __future__ import annotations

from .checks import data_integrity, output_schemas


def main() -> int:
    manifest = data_integrity._build_manifest()
    data_integrity._write(manifest)
    schema_status = output_schemas._write_fingerprints()
    if schema_status != 0:
        return int(schema_status)
    print("[ok] refreshed frozen-evidence integrity and schema metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
