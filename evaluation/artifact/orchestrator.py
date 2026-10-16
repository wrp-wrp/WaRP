#!/usr/bin/env python3
"""Run or preflight the experiment registry in dependency order."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from .registry import ROOT, ordered_for_run, prepare_phase, run_phase


RUN_MODES = {"local-run", "server-run"}
CHECK_MODES = {"server-check"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="plan",
        choices=("plan", "local-run", "server-check", "server-run", "remote-check", "remote-run"),
        help="Local/server aliases execute registry commands on this host. Global remote-run is unsupported; use a runner's SSH flags.",
    )
    parser.add_argument("--only", default="", help="Comma-separated registry ids; dependencies are included.")
    parser.add_argument("--profile", default="full", choices=("full", "light"))
    parser.add_argument("--phase", default="run", choices=("prepare", "run"))
    parser.add_argument("--local-lance-root", default="")
    parser.add_argument("--s3-bucket", default="")
    parser.add_argument("--s3-region", default="")
    parser.add_argument("--force", action="store_true")
    # Kept only to turn historical invocations into a clear error instead of
    # silently pretending all heterogeneous runners share one remote contract.
    parser.add_argument("--ssh-target", default="")
    parser.add_argument("--ssh-key", default="")
    return parser.parse_args()


def _selected(raw: str) -> list[str] | None:
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    return selected or None


def _script_path(command: list[str]) -> Path | None:
    for part in command[1:]:
        if part.endswith(".py"):
            path = ROOT / part
            return path if path.is_file() else None
    return None


def _declares_flag(command: list[str], flag: str) -> bool:
    script = _script_path(command)
    return script is not None and flag in script.read_text(encoding="utf-8", errors="ignore")


def _with_control_flag(command: list[str], flag: str) -> list[str] | None:
    script = _script_path(command)
    if script is None:
        return None
    source = script.read_text(encoding="utf-8", errors="ignore")
    if flag not in source:
        return None
    # argparse global options must precede a subcommand. Registry commands use
    # ``python script.py <subcommand> ...``, so place control flags immediately
    # after the script for that parser shape; ordinary runners accept them last.
    if "add_subparsers" in source:
        script_index = next(index for index, part in enumerate(command) if part.endswith(".py"))
        return [*command[: script_index + 1], flag, *command[script_index + 1 :]]
    return [*command, flag]


def _environment(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    if args.local_lance_root:
        env["VECIO_LANCE_ROOT"] = str(Path(args.local_lance_root).expanduser().resolve())
    if args.s3_bucket:
        env["VECIO_S3_BUCKET"] = args.s3_bucket
    if args.s3_region:
        env["VECIO_S3_REGION"] = args.s3_region
        env.setdefault("AWS_DEFAULT_REGION", args.s3_region)
    return env


def main() -> int:
    args = parse_args()
    if args.mode in {"remote-check", "remote-run"} or args.ssh_target or args.ssh_key:
        raise SystemExit(
            "there is no safe repository-global SSH contract; invoke the selected runner "
            "with its documented --ssh-target/--ssh-key flags"
        )
    specs = ordered_for_run(_selected(args.only))
    env = _environment(args)
    executed = 0
    for spec in specs:
        phase = prepare_phase(spec, args.profile) if args.phase == "prepare" else run_phase(spec, args.profile)
        commands = [command.resolve() for command in phase.commands]
        if not commands:
            print(f"[skip] {spec.id}: no {args.phase} phase for profile={args.profile}")
            continue
        print(f"[experiment] {spec.id}", flush=True)
        for raw in commands:
            command = list(raw)
            if args.force:
                forced = _with_control_flag(command, "--force")
                if forced is not None:
                    command = forced
            if args.mode in CHECK_MODES:
                checked = _with_control_flag(command, "--check-only")
                if checked is None:
                    print("[static-plan] " + shlex.join(command))
                    continue
                command = checked
            print("$ " + shlex.join(command), flush=True)
            if args.mode != "plan":
                try:
                    subprocess.run(command, cwd=ROOT, env=env, check=True)
                except subprocess.CalledProcessError as exc:
                    raise SystemExit(
                        f"[fail] {spec.id} {args.mode} command exited with status "
                        f"{exc.returncode}: {shlex.join(command)}"
                    ) from None
            executed += 1
    verb = "planned" if args.mode == "plan" else "checked" if args.mode in CHECK_MODES else "completed"
    print(
        f"[ok] {verb} {executed} {args.phase} command(s) across {len(specs)} spec(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
