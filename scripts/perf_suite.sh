#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-quick}" # quick | full

WARMUP_SECS="${WARMUP_SECS:-1}"
MEASURE_SECS="${MEASURE_SECS:-3}"

if [[ "$MODE" != "quick" && "$MODE" != "full" ]]; then
  echo "unknown mode: $MODE (use quick|full)"
  exit 1
fi

echo "[perf] mode=$MODE warmup=${WARMUP_SECS}s measure=${MEASURE_SECS}s"

echo "[perf] cargo bench hot_path"
(
  cd "$ROOT_DIR"
  cargo bench --bench hot_path -- --warm-up-time "$WARMUP_SECS" --measurement-time "$MEASURE_SECS"
)

if [[ "$MODE" == "full" ]]; then
  echo "[perf] cargo bench transcode"
  (
    cd "$ROOT_DIR"
    cargo bench --bench transcode -- --warm-up-time "$WARMUP_SECS" --measurement-time "$MEASURE_SECS"
  )

  echo "[perf] e2e forwarding"
  (
    cd "$ROOT_DIR"
    REQS="${REQS:-2000}" \
    CONCURRENCY="${CONCURRENCY:-16}" \
    WORKER_THREADS="${WORKER_THREADS:-auto}" \
    scripts/bench_forwarding.sh all
  )

  echo "[perf] e2e fc_inject"
  (
    cd "$ROOT_DIR"
    REQS="${REQS:-2000}" \
    CONCURRENCY="${CONCURRENCY:-16}" \
    WORKER_THREADS="${WORKER_THREADS:-auto}" \
    scripts/bench_fc_inject.sh all
  )
fi

echo "[perf] done"
