#!/usr/bin/env bash
#
# scripts/setup_ec2.sh — one-click EC2 environment bootstrap.
#
# Idempotent. Safe to re-run. Installs the Python + system deps needed
# for both light-mode reproduction and full-mode reruns.
#
# - Python deps: pinned light rendering plus full dataset/Lance preparation
#                dependencies from requirements-full.txt.
# - System deps: git, rsync, ssh, build-essential, and cargo/rustc for
#                building the companion Lance examples.
#
# After running this, you should be able to:
#   python3 -m artifact check
#   bash scripts/run_all.sh --profile light
#
# Full mode still needs AWS credentials, public raw datasets, and a writable bucket.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log()  { echo "[setup_ec2] $*"; }
warn() { echo "[setup_ec2] warn: $*" >&2; }

need_sudo=0
if [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
  need_sudo=1
fi
sudo_wrap() { if [[ "$need_sudo" -eq 1 ]]; then sudo "$@"; else "$@"; fi; }

INSTALL_SYSTEM=${VECIO_INSTALL_SYSTEM:-1}
INSTALL_PYTHON=${VECIO_INSTALL_PYTHON:-1}
INSTALL_CARGO=${VECIO_INSTALL_CARGO:-yes}   # yes | no | auto

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-system)       INSTALL_SYSTEM=0 ;;
    --no-python)       INSTALL_PYTHON=0 ;;
    --with-cargo)      INSTALL_CARGO=yes ;;
    --no-cargo)        INSTALL_CARGO=no ;;
    -h|--help)
      cat <<USAGE
Usage: bash scripts/setup_ec2.sh [--no-system] [--no-python] [--with-cargo|--no-cargo]

Without flags, this installs all system packages, all Python deps, and Cargo,
so both light and full reproduction prerequisites are present. Use --no-cargo
for a render-only host.

Env overrides: VECIO_INSTALL_SYSTEM / VECIO_INSTALL_PYTHON /
               VECIO_INSTALL_CARGO (auto|yes|no).
USAGE
      exit 0 ;;
    *) warn "unknown flag: $1"; exit 1 ;;
  esac
  shift
done

log "host: $(uname -srm)"
log "python: $(python3 --version 2>&1)"

# ---- system packages (Debian/Ubuntu) ---------------------------------------
if [[ "$INSTALL_SYSTEM" -eq 1 ]] && command -v apt-get >/dev/null 2>&1; then
  log "installing system packages via apt-get"
  sudo_wrap apt-get update -y
  sudo_wrap apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git rsync openssh-client build-essential \
    cmake pkg-config libssl-dev protobuf-compiler \
    curl ca-certificates \
    poppler-utils
fi

# ---- Python deps -----------------------------------------------------------
if [[ "$INSTALL_PYTHON" -eq 1 ]]; then
  # Pin matplotlib to 3.10.x so regenerated figures match the paper snapshot.
  log "installing pinned Python dependency ranges from requirements-full.txt"
  # Ubuntu 24.04+ ships PEP 668 "externally-managed" python; --break-system-packages
  # is the documented escape hatch for artifact hosts.
  python3 -m pip install --upgrade --break-system-packages -r requirements-full.txt 2>/dev/null \
    || python3 -m pip install --upgrade -r requirements-full.txt
fi

# ---- cargo (optional, for Lance build) ------------------------------------
case "$INSTALL_CARGO" in
  yes) NEED_CARGO=1 ;;
  no)  NEED_CARGO=0 ;;
  auto)
    if [[ -d "third_party/lance" ]]; then NEED_CARGO=1; else NEED_CARGO=0; fi
    ;;
  *) NEED_CARGO=0 ;;
esac
if [[ "$NEED_CARGO" -eq 1 ]]; then
  if ! command -v protoc >/dev/null 2>&1; then
    warn "protoc is required for Lance; install protobuf-compiler before building"
    exit 1
  fi
  log "protobuf: $(protoc --version)"
fi
if [[ "$NEED_CARGO" -eq 1 ]] && ! command -v cargo >/dev/null 2>&1; then
  log "installing cargo via rustup (minimal profile)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi
if command -v cargo >/dev/null 2>&1; then
  log "cargo: $(cargo --version)"
fi

# ---- sanity: can we import every required module? -------------------------
python3 - <<'PY'
import importlib, sys
required = [
    "numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml", "boto3",
    "PIL", "h5py", "faiss", "pyarrow",
]
missing = []
for m in required:
    try:
        importlib.import_module(m)
    except ImportError as e:
        missing.append((m, str(e)))
if missing:
    print("[setup_ec2] MISSING python modules:", file=sys.stderr)
    for m, err in missing:
        print(f"  {m}: {err}", file=sys.stderr)
    sys.exit(1)
print("[setup_ec2] python dep check ok")
PY

log "setup complete. Next steps:"
log "  python3 -m artifact check"
log "  bash scripts/run_all.sh --profile light"
