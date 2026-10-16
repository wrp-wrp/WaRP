#!/usr/bin/env python3
"""Single command-line interface for the WaRP reproduction artifact."""

from __future__ import annotations

import importlib
import sys

from . import cli


MODULE_COMMANDS = {
    "rebuild": "artifact.rebuild",
    "compare": "artifact.compare",
    "check": "artifact.smoke",
    "freeze": "artifact.freeze",
    "catalog": "artifact.catalog",
    "orchestrate": "artifact.orchestrator",
    "prepare-dataset": "artifact.prepare_dataset",
    "requirements": "artifact.requirements",
}


def _usage() -> str:
    return """Usage: python3 -m artifact <command> [options]

Paper Experiment commands:
  list                         List registered experiment specs.
  show <id>                    Show one spec and its commands.
  show <id> --parameters       Print its resolved paper-parameter contract.
  prepare <id>                 Prepare its public datasets and companion tools.
  run <id>                     Execute one spec's measurement phase.
  render <id>                  Safely render one spec to _regen/ or _latest/.
  rebuild                      Re-render all paper artifacts safely.
  compare                      Compare regenerated and authoritative artifacts.
  check                        Run the offline reproducibility checks.
  freeze                       Refresh frozen-data hashes and schema metadata.

Maintainer commands:
  catalog <check|write-doc|check-doc>
  orchestrate [options]        Run the dependency-ordered full registry.
  prepare-dataset [options]    Prepare a public benchmark dataset.
  requirements [--only IDS]    Print aggregate registry runtime requirements.
"""


def _delegate(module_name: str, argv: list[str]) -> int:
    module = importlib.import_module(module_name)
    previous = sys.argv
    try:
        sys.argv = [f"python3 -m {module_name}", *argv]
        return int(module.main())
    finally:
        sys.argv = previous


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0
    command, *argv = sys.argv[1:]
    if command in {"list", "show", "prepare", "run", "render"}:
        return cli.main([command, *argv])
    module_name = MODULE_COMMANDS.get(command)
    if module_name is None:
        print(f"unknown artifact command: {command}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    return _delegate(module_name, argv)


if __name__ == "__main__":
    raise SystemExit(main())
