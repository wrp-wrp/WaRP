#!/usr/bin/env python3
"""Report aggregate runtime requirements for a registry selection."""

from __future__ import annotations

import argparse

from .registry import ordered_for_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    selected = [item.strip() for item in args.only.split(",") if item.strip()] or None
    specs = ordered_for_run(selected)
    needs_lance = any(spec.needs_lance for spec in specs)
    needs_lance_python = any(spec.needs_lance_python for spec in specs)
    needs_s3 = any(spec.needs_s3 for spec in specs)
    has_prepare = any(spec.prepare.commands for spec in specs)
    needs_figure4_topologies = any(
        spec.id == "figure-04-object-store-io" for spec in specs
    )
    print(
        "\t".join(
            "1" if value else "0"
            for value in (
                needs_lance,
                needs_lance_python,
                needs_s3,
                has_prepare,
                needs_figure4_topologies,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
