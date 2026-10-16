#!/usr/bin/env bash
# Run the full Rust benchmark suite and archive CSV + log artifacts.
#
# Output:
#   target/bench-artifacts/
#     ivf_workload.csv             — P × dist × strategy table
#     ivf_workload.stderr.log      — human-readable quality table
#     plan_quality.stderr.log
#     plan_time.stderr.log
#     main_solution_speed.stderr.log
#     planner_smoke.stderr.log
#   target/criterion/              — Criterion's HTML reports
set -euo pipefail
cd "$(dirname "$0")/.."

ARTIFACT_DIR="target/bench-artifacts"
mkdir -p "${ARTIFACT_DIR}"

BENCH_FAILURES=()

run_bench() {
    local name="$1"
    local log="${ARTIFACT_DIR}/${name}.stderr.log"
    echo "=== cargo bench --bench ${name} ==="
    local status=0
    (cd rust && cargo bench --bench "${name}" 2> >(tee "../${log}" >&2)) || status=$?
    if [[ ${status} -ne 0 ]]; then
        BENCH_FAILURES+=("${name}:${status}")
        echo "[bench] ${name} failed with exit code ${status}" >&2
    fi
}

run_bench "ivf_workload"
run_bench "plan_quality"
run_bench "plan_time"
run_bench "main_solution_speed"
run_bench "planner_smoke"

echo
echo "=== artifacts ==="
ls -lh "${ARTIFACT_DIR}/" 2>/dev/null || true
if [[ -f "${ARTIFACT_DIR}/ivf_workload.csv" ]]; then
    echo
    echo "=== ivf_workload.csv (first 20 rows) ==="
    head -20 "${ARTIFACT_DIR}/ivf_workload.csv"
fi

if [[ ${#BENCH_FAILURES[@]} -ne 0 ]]; then
    echo
    echo "=== failed benches ===" >&2
    printf '  %s\n' "${BENCH_FAILURES[@]}" >&2
    exit 1
fi
