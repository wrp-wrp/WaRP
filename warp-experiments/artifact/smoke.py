#!/usr/bin/env python3
"""Representative reproducibility smoke test.

Verifies the artifact is structurally sound without running any real
benchmark: compile-checks every runner/renderer touched by the manifest,
probes that each runner honours `--check-only` (skipping those that do
not implement it), runs the light-profile plan, and re-renders all
paper artifacts from bundled frozen inputs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .environment import discover_lance_root, with_lance_env
from .registry import EXPERIMENTS, ROOT, run_phase


# Specs whose `run_data.py --check-only` works without any heavy deps
# (boto3 / scipy / sklearn not imported at module top-level). Keep this
# set tight so `artifact check` stays runnable on a bare-minimum Python
# environment; full-mode prerequisites are checked by scripts/run_all.sh.
RUNNERS_WITH_CHECK_ONLY = {
    "real-s3-tiny-request-fanout-sanity",
    "figure-13-reordering-methods",
    "figure-01-s3-range-get-latency",
}


def _py_compile_targets() -> list[str]:
    targets: list[str] = []
    for spec in EXPERIMENTS:
        for phase in (spec.run, spec.render):
            for cmd in phase.commands:
                for part in cmd.argv:
                    if part.endswith(".py") and part not in targets and not part.startswith("-"):
                        targets.append(part)
    for source_root in (ROOT / "runners", ROOT / "src", ROOT / "artifact"):
        for path in sorted(source_root.rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative not in targets:
                targets.append(relative)
    return targets


def _base_commands() -> list[dict]:
    return [
        {
            "id": "py_compile",
            "cmd": [sys.executable, "-m", "py_compile", *_py_compile_targets()],
        },
        {
            "id": "data_integrity",
            "cmd": [sys.executable, "-m", "artifact.checks.data_integrity", "--verify"],
        },
        {
            "id": "output_schemas",
            "cmd": [sys.executable, "-m", "artifact.checks.output_schemas", "--verify"],
        },
        {
            "id": "runner_cli_flags",
            "cmd": [sys.executable, "-m", "artifact.checks.runner_cli"],
        },
        {
            "id": "unified_prepare_plan",
            "cmd": [
                sys.executable,
                "-m",
                "artifact",
                "orchestrate",
                "--phase",
                "prepare",
                "--mode",
                "plan",
            ],
        },
        {
            "id": "unified_runner_plan",
            "cmd": [sys.executable, "-m", "artifact", "orchestrate", "--mode", "plan"],
        },
        {
            "id": "unit_tests",
            "cmd": [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        },
        {
            "id": "experiment_entrypoint_list",
            "cmd": [sys.executable, "-m", "artifact", "list"],
        },
        {
            "id": "artifact_catalog",
            "cmd": [sys.executable, "-m", "artifact", "catalog", "check"],
        },
        {
            "id": "public_repository_hygiene",
            "cmd": [sys.executable, "-m", "artifact.checks.public_hygiene"],
        },
        {
            "id": "run_all_syntax",
            "cmd": [
                "bash",
                "-n",
                "scripts/run_all.sh",
                "scripts/setup_ec2.sh",
                "scripts/setup_lance_python.sh",
            ],
        },
    ]


def _per_spec_check_commands() -> list[dict]:
    """Invoke run_full for specs that accept --check-only, stripping trailing path args."""
    out: list[dict] = []
    for spec in EXPERIMENTS:
        if spec.id not in RUNNERS_WITH_CHECK_ONLY:
            continue
        run = run_phase(spec, "full")
        if not run.commands:
            continue
        cmd = run.commands[0]
        argv = [part for part in cmd.resolve()]
        if "--check-only" not in argv:
            argv.append("--check-only")
        out.append({"id": f"{spec.id}_check", "cmd": argv})
    return out


def _final_commands() -> list[dict]:
    return [
        {
            "id": "paper_artifacts_smoke",
            "cmd": [sys.executable, "-m", "artifact", "rebuild", "--profile", "light"],
        },
        {
            "id": "expected_outputs",
            "cmd": [sys.executable, "-m", "artifact.checks.expected_outputs"],
        },
    ]


def _dry_run_full_commands() -> list[dict]:
    """Verify runner-output paths line up with render-input paths.

    This is a static consistency check: for every spec with a runner,
    confirm that `expected_runner_outputs` covers at least one file
    that render_* will actually read. Runs without invoking the runner.
    """
    return [
        {
            "id": "dry_run_full_schema",
            "cmd": [sys.executable, "-m", "artifact.checks.runner_render_schema"],
        }
    ]


def all_commands() -> list[dict]:
    return _base_commands() + _per_spec_check_commands() + _final_commands()


def all_commands_dry_run_full() -> list[dict]:
    return (
        _base_commands()
        + _per_spec_check_commands()
        + _dry_run_full_commands()
        + _final_commands()
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a representative local reproducibility smoke test.")
    ap.add_argument("--only", default="", help="Comma-separated smoke ids to run.")
    ap.add_argument(
        "--local-lance-root",
        default="",
        help="Optional companion Lance checkout. If omitted, discover one locally; cloning stays disabled by default.",
    )
    ap.set_defaults(dry_run_full=True)
    ap.add_argument(
        "--dry-run-full",
        action="store_true",
        help="Include runner/render schema consistency (default).",
    )
    ap.add_argument(
        "--no-dry-run-full",
        dest="dry_run_full",
        action="store_false",
        help="Skip runner/render schema consistency.",
    )
    return ap.parse_args()


def selected_commands(args: argparse.Namespace) -> list[dict]:
    commands = all_commands_dry_run_full() if getattr(args, "dry_run_full", False) else all_commands()
    if not args.only.strip():
        return commands
    requested = {tok.strip() for tok in args.only.split(",") if tok.strip()}
    missing = sorted(requested - {entry["id"] for entry in commands})
    if missing:
        raise SystemExit(f"unknown smoke ids: {', '.join(missing)}")
    return [entry for entry in commands if entry["id"] in requested]


def _check_lance_binary_interfaces(lance_root: Path) -> None:
    strict_bin = lance_root / "target" / "release" / "examples" / "strict_budget_real_s3"
    planner_bin = lance_root / "target" / "release" / "examples" / "planner_overhead_real"
    checks = [
        (strict_bin, "--greedy-gap-kb", "strict_budget_real_s3"),
        (planner_bin, "--input-json", "planner_overhead_real"),
    ]
    for binary, expected_flag, label in checks:
        if not binary.exists():
            continue
        proc = subprocess.run(
            [str(binary), "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        help_text = proc.stdout or ""
        if expected_flag not in help_text:
            raise SystemExit(
                f"{label} binary is stale or incompatible: missing {expected_flag} in --help output"
            )
        print(f"[smoke] {label}_help ok")


def main() -> int:
    args = parse_args()
    # Static checks do not require a companion checkout. If one is present,
    # validate its already-built interfaces without cloning or changing it.
    lance_root = discover_lance_root(str(args.local_lance_root))
    if lance_root is not None:
        env = with_lance_env(os.environ, lance_root)
        print(f"[env] lance_root={lance_root}")
        _check_lance_binary_interfaces(lance_root)
    else:
        print("[env] no companion Lance checkout; skipping optional binary checks.")
        env = dict(os.environ)
    commands = selected_commands(args)
    for entry in commands:
        print(f"[smoke] {entry['id']}")
        print("$ " + " ".join(entry["cmd"]))
        subprocess.run(entry["cmd"], cwd=ROOT, check=True, env=env)
    print(f"[ok] completed {len(commands)} smoke checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
