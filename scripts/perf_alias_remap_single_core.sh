#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DURATION="${DURATION:-10s}"
WRK_THREADS="${WRK_THREADS:-1}"
CONNECTIONS="${CONNECTIONS:-8}"
WORKER_THREADS="${WORKER_THREADS:-1}"
UPSTREAM_TRANSPORT="${UPSTREAM_TRANSPORT:-h2c}"
REQUIRE_UPSTREAM_H2="${REQUIRE_UPSTREAM_H2:-1}"
MOCK_SCENARIO="${MOCK_SCENARIO:-text}"
MODE="${1:-nonstream}" # nonstream | stream | all

if [[ "$WORKER_THREADS" != "1" ]]; then
  echo "This profile is single-core only. Set WORKER_THREADS=1."
  exit 1
fi

if [[ "$WRK_THREADS" != "1" || "$CONNECTIONS" != "8" ]]; then
  echo "This profile is fixed to wrk_threads=1 and connections=8."
  echo "Set WRK_THREADS=1 and CONNECTIONS=8."
  exit 1
fi

(
  cd "$ROOT_DIR"
  DURATION="$DURATION" \
  WRK_THREADS="$WRK_THREADS" \
  CONNECTIONS="$CONNECTIONS" \
  WORKER_THREADS="$WORKER_THREADS" \
  UPSTREAM_TRANSPORT="$UPSTREAM_TRANSPORT" \
  REQUIRE_UPSTREAM_H2="$REQUIRE_UPSTREAM_H2" \
  MOCK_SCENARIO="$MOCK_SCENARIO" \
  scripts/bench_alias_remap_wrk.sh "$MODE"
)
