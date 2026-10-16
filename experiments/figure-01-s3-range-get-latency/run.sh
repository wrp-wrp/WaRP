#!/usr/bin/env bash
# Paper Experiment wrapper; registry: artifact/registry.py.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
PROFILE="${1:-full}"
exec python3 -m artifact run figure-01-s3-range-get-latency --profile "$PROFILE"
