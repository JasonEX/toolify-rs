#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Unified benchmark profile (single-core historical baseline):
# - wrk threads: 1
# - wrk connections: 8
DURATION="${DURATION:-10s}"
WRK_THREADS="${WRK_THREADS:-1}"
CONNECTIONS="${CONNECTIONS:-8}"
WORKER_THREADS="${WORKER_THREADS:-1}"
UPSTREAM_TRANSPORT="${UPSTREAM_TRANSPORT:-h2c}"
REQUIRE_UPSTREAM_H2="${REQUIRE_UPSTREAM_H2:-1}"
MOCK_SCENARIO="${MOCK_SCENARIO:-text}"
AUTO_BUILD_RELEASE="${AUTO_BUILD_RELEASE:-1}"

if [[ "$WORKER_THREADS" != "1" ]]; then
  echo "This profile is single-core only. Set WORKER_THREADS=1."
  exit 1
fi

if [[ "$WRK_THREADS" != "1" || "$CONNECTIONS" != "8" ]]; then
  echo "This standard profile is fixed to wrk_threads=1 and connections=8."
  echo "Set WRK_THREADS=1 and CONNECTIONS=8."
  exit 1
fi

echo "[standard] profile=single_core_1x8 duration=$DURATION wrk_threads=$WRK_THREADS connections=$CONNECTIONS worker_threads=$WORKER_THREADS"
echo "[standard] upstream_transport=$UPSTREAM_TRANSPORT require_upstream_h2=$REQUIRE_UPSTREAM_H2 mock_scenario=$MOCK_SCENARIO"

echo "[standard] forwarding_wrk"
(
  cd "$ROOT_DIR"
  DURATION="$DURATION" \
  WRK_THREADS="$WRK_THREADS" \
  CONNECTIONS="$CONNECTIONS" \
  WORKER_THREADS="$WORKER_THREADS" \
  AUTO_BUILD_RELEASE="$AUTO_BUILD_RELEASE" \
  UPSTREAM_TRANSPORT="$UPSTREAM_TRANSPORT" \
  REQUIRE_UPSTREAM_H2="$REQUIRE_UPSTREAM_H2" \
  MOCK_SCENARIO="$MOCK_SCENARIO" \
  scripts/bench_forwarding_wrk.sh all
)

echo "[standard] fc_inject_wrk"
(
  cd "$ROOT_DIR"
  DURATION="$DURATION" \
  WRK_THREADS="$WRK_THREADS" \
  CONNECTIONS="$CONNECTIONS" \
  WORKER_THREADS="$WORKER_THREADS" \
  AUTO_BUILD_RELEASE="$AUTO_BUILD_RELEASE" \
  UPSTREAM_TRANSPORT="$UPSTREAM_TRANSPORT" \
  REQUIRE_UPSTREAM_H2="$REQUIRE_UPSTREAM_H2" \
  MOCK_SCENARIO="$MOCK_SCENARIO" \
  scripts/bench_fc_inject_wrk.sh all
)

echo "[standard] done"
