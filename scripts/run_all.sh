#!/usr/bin/env bash
# Registry-driven entrypoint for light rendering or full paper measurements.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${VECIO_EXPERIMENT_PROFILE:-full}"
ONLY=""
LANCE_ROOT="${VECIO_LANCE_ROOT:-}"
SKIP_BUILD=0
RUN_CHECK=1
RUN_PREFLIGHT=1
RUN_PREPARE=1
RUN_RENDER=1
FORCE=0
PLAN_ONLY=0

load_env_file() {
  local path="$1"
  if [[ -n "$path" && -f "$path" ]]; then
    echo "[env] loading $path"
    set -a
    # shellcheck disable=SC1090
    source "$path"
    set +a
  fi
}

if [[ -n "${VECIO_ENV_FILE:-}" ]]; then
  load_env_file "$VECIO_ENV_FILE"
elif [[ -f "$ROOT/.vecio-run.env" ]]; then
  load_env_file "$ROOT/.vecio-run.env"
elif [[ -f "$ROOT/scripts/run_all.env" ]]; then
  load_env_file "$ROOT/scripts/run_all.env"
fi

# Re-read values that may have come from the selected env file.
PROFILE="${VECIO_EXPERIMENT_PROFILE:-$PROFILE}"
LANCE_ROOT="${VECIO_LANCE_ROOT:-$LANCE_ROOT}"
if [[ -n "${VECIO_PYTHON:-}" ]]; then
  PYTHON="$VECIO_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi
command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "Python interpreter is not executable: $PYTHON" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_all.sh [options]

  --profile light|full   light renders committed evidence; full runs measurements.
  --only ID[,ID...]     Run selected registry specs and their dependencies.
  --lance-root PATH     Use an existing paper Lance companion checkout.
  --skip-build          Reuse already-built Lance binaries.
  --no-check            Skip the offline repository checks.
  --no-preflight        Skip runner --check-only/static plans before a full run.
  --no-prepare          Reuse already-prepared datasets and their local manifests.
  --no-render           Do not render fresh data after a full run.
  --force               Forward --force only to runners that declare it.
  --plan                Print the dependency-ordered full plan without executing it.
  -h, --help            Show this message.

Full mode prepares public datasets and companion tools declared by each selected
experiment, then runs its measurement. It never replaces committed data/paper
snapshots; fresh outputs go to data/latest and figures/tables to _latest.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --lance-root) LANCE_ROOT="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --no-check) RUN_CHECK=0; shift ;;
    --no-preflight) RUN_PREFLIGHT=0; shift ;;
    --no-prepare) RUN_PREPARE=0; shift ;;
    --no-render) RUN_RENDER=0; shift ;;
    --force) FORCE=1; shift ;;
    --plan) PLAN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$PROFILE" != "light" && "$PROFILE" != "full" ]]; then
  echo "--profile must be light or full" >&2
  exit 2
fi

cd "$ROOT"

if [[ "$PLAN_ONLY" -eq 1 ]]; then
  plan_args=("$PYTHON" -m artifact orchestrate --mode plan --profile full)
  if [[ -n "$ONLY" ]]; then
    plan_args+=(--only "$ONLY")
  fi
  "${plan_args[@]}" --phase prepare
  "${plan_args[@]}" --phase run
  exit 0
fi

if [[ "$RUN_CHECK" -eq 1 ]]; then
  "$PYTHON" -m artifact check
fi

rebuild_args=("$PYTHON" -m artifact rebuild --profile "$PROFILE")
if [[ -n "$ONLY" ]]; then
  rebuild_args+=(--only "$ONLY")
fi

if [[ "$PROFILE" == "light" ]]; then
  echo "[light] rendering every selected paper artifact from data/paper"
  "${rebuild_args[@]}" --data-slot paper
  echo "[ok] light reproduction finished; authoritative artifacts were untouched"
  exit 0
fi

requirements_args=("$PYTHON" -m artifact requirements)
if [[ -n "$ONLY" ]]; then
  requirements_args+=(--only "$ONLY")
fi
read -r NEEDS_LANCE NEEDS_LANCE_PYTHON NEEDS_S3 HAS_PREPARE NEEDS_FIG4_TOPOLOGIES < <("${requirements_args[@]}")
echo "[requirements] lance=$NEEDS_LANCE lance_python=$NEEDS_LANCE_PYTHON s3=$NEEDS_S3 prepare=$HAS_PREPARE fig4_topologies=$NEEDS_FIG4_TOPOLOGIES"

# Figure 4 is the only global spec whose required topology cannot be prepared
# on this host. Reject an incomplete environment before Cargo builds or large
# public-dataset/S3 preparation begins.
if [[ "$RUN_PREFLIGHT" -eq 1 && "$NEEDS_FIG4_TOPOLOGIES" -eq 1 ]]; then
  "$PYTHON" runners/figure-04-object-store-io/run_data.py --check-only
fi

if [[ "$NEEDS_LANCE" -eq 1 && -z "$LANCE_ROOT" ]]; then
  LANCE_ROOT="$("$PYTHON" - <<'PY'
from artifact.environment import discover_lance_root
root = discover_lance_root()
if root is None:
    raise SystemExit(
        "paper Lance companion checkout not found; set VECIO_LANCE_ROOT or --lance-root"
    )
print(root)
PY
)"
fi
if [[ "$NEEDS_LANCE" -eq 1 ]]; then
  LANCE_ROOT="$(cd "$LANCE_ROOT" && pwd)"
  export VECIO_LANCE_ROOT="$LANCE_ROOT"
  echo "[env] lance_root=$LANCE_ROOT"
fi
echo "[env] s3_bucket=${VECIO_S3_BUCKET:-<unset>}"
echo "[env] s3_region=${VECIO_S3_REGION:-<unset>}"

orchestrate_args=("$PYTHON" -m artifact orchestrate --profile full)
if [[ -n "$LANCE_ROOT" ]]; then
  orchestrate_args+=(--local-lance-root "$LANCE_ROOT")
fi
if [[ -n "$ONLY" ]]; then
  orchestrate_args+=(--only "$ONLY")
fi
if [[ -n "${VECIO_S3_BUCKET:-}" ]]; then
  orchestrate_args+=(--s3-bucket "$VECIO_S3_BUCKET")
fi
if [[ -n "${VECIO_S3_REGION:-}" ]]; then
  orchestrate_args+=(--s3-region "$VECIO_S3_REGION")
fi
if [[ "$FORCE" -eq 1 ]]; then
  orchestrate_args+=(--force)
fi
if [[ "$SKIP_BUILD" -eq 1 ]]; then
  export VECIO_SKIP_LANCE_BUILD=1
fi

if [[ "$RUN_PREFLIGHT" -eq 1 && "$RUN_PREPARE" -eq 1 ]]; then
  "${orchestrate_args[@]}" --phase prepare --mode server-check
fi

if [[ "$NEEDS_LANCE" -eq 1 && "$SKIP_BUILD" -eq 0 ]]; then
  if ! command -v cargo >/dev/null 2>&1 && [[ -f "$HOME/.cargo/env" ]]; then
    # setup_ec2.sh installs Rustup here, but a fresh non-interactive shell may
    # not have loaded its PATH update yet.
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
  command -v cargo >/dev/null 2>&1 || {
    echo "cargo is required for full mode; run scripts/setup_ec2.sh --with-cargo" >&2
    exit 2
  }
  echo "[build] paper Lance companion examples"
  (
    cd "$LANCE_ROOT"
    # Run inside the checkout so Rustup honors Lance's rust-toolchain.toml.
    cargo build --release -p lance \
      --example full_pipeline_real_s3 --example full_pipeline_recall_s3 \
      --target-dir "$LANCE_ROOT/target"
    cargo build --release -p lance-io \
      --example strict_budget_real_s3 --example planner_overhead_real \
      --example lance_scheduler_qps_s3 --example lance_scheduler_qps_s3_per_iop \
      --target-dir "$LANCE_ROOT/target"
  )
  "$PYTHON" runners/shared/lance_build_receipt.py \
    --lance-root "$LANCE_ROOT" \
    --binary "$LANCE_ROOT/target/release/examples/full_pipeline_real_s3" \
    --package lance \
    --example full_pipeline_real_s3
  # Standalone `artifact prepare` owns the Figure 11 build when used directly.
  # The unified entrypoint has just built the same target directory, so its
  # preparation phase should verify those binaries instead of invoking Cargo twice.
  export VECIO_SKIP_LANCE_BUILD=1
fi

if [[ "$NEEDS_LANCE_PYTHON" -eq 1 && "$RUN_PREPARE" -eq 1 ]]; then
  bash scripts/setup_lance_python.sh --lance-root "$LANCE_ROOT"
fi

if [[ "$RUN_PREPARE" -eq 1 && "$HAS_PREPARE" -eq 1 ]]; then
  "${orchestrate_args[@]}" --phase prepare --mode server-run
fi

if [[ "$RUN_PREFLIGHT" -eq 1 ]]; then
  "${orchestrate_args[@]}" --phase run --mode server-check
fi

"${orchestrate_args[@]}" --phase run --mode server-run

if [[ "$RUN_RENDER" -eq 1 ]]; then
  echo "[artifact] rendering fresh runner outputs from data/latest"
  "${rebuild_args[@]}" --data-slot latest
fi

echo "[ok] full registry workflow finished; committed paper evidence was untouched"
