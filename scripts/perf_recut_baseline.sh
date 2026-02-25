#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROUNDS="${ROUNDS:-9}"
DURATION="${DURATION:-10s}"
WRK_THREADS="${WRK_THREADS:-1}"
CONNECTIONS="${CONNECTIONS:-8}"
WORKER_THREADS="${WORKER_THREADS:-1}"
UPSTREAM_TRANSPORT="${UPSTREAM_TRANSPORT:-h2c}"
REQUIRE_UPSTREAM_H2="${REQUIRE_UPSTREAM_H2:-1}"
MOCK_SCENARIO="${MOCK_SCENARIO:-text}"
OUT_FILE="${OUT_FILE:-$ROOT_DIR/artifacts/perf/baseline_single_core_1x8_pinned.md}"
INCLUDE_ALIAS_REMAP_SCENARIO="${INCLUDE_ALIAS_REMAP_SCENARIO:-0}" # 0 | 1

SCENARIOS=(
  "forward_nonstream_wrk"
  "forward_stream_wrk"
  "fc_inject_nonstream_wrk"
  "fc_inject_stream_wrk"
)
if [[ "$INCLUDE_ALIAS_REMAP_SCENARIO" == "1" ]]; then
  SCENARIOS+=(
    "alias_remap_nonstream_wrk"
    "alias_remap_stream_wrk"
  )
fi

if [[ "$WORKER_THREADS" != "1" ]]; then
  echo "This baseline profile is single-core only. Set WORKER_THREADS=1."
  exit 1
fi

if [[ "$WRK_THREADS" != "1" || "$CONNECTIONS" != "8" ]]; then
  echo "This baseline profile is fixed to wrk_threads=1 and connections=8."
  echo "Set WRK_THREADS=1 and CONNECTIONS=8."
  exit 1
fi

mkdir -p "$(dirname "$OUT_FILE")"
TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

latency_to_us() {
  local raw="${1// /}"
  raw="${raw,,}"
  if [[ -z "$raw" || "$raw" == "n/a" ]]; then
    return 1
  fi
  if [[ "$raw" =~ ^([0-9]+([.][0-9]+)?)(ns|us|ms|s)$ ]]; then
    local value="${BASH_REMATCH[1]}"
    local unit="${BASH_REMATCH[3]}"
    local factor
    case "$unit" in
      ns) factor="0.001" ;;
      us) factor="1" ;;
      ms) factor="1000" ;;
      s) factor="1000000" ;;
      *) return 1 ;;
    esac
    awk -v v="$value" -v f="$factor" 'BEGIN { printf "%.6f", v * f }'
    return 0
  fi
  return 1
}

us_to_pretty_ms() {
  local us="$1"
  awk -v v="$us" 'BEGIN { printf "%.2fms", v / 1000.0 }'
}

median_of_file() {
  local file="$1"
  sort -n "$file" | awk '
    { v[NR] = $1 }
    END {
      if (NR == 0) { exit 1 }
      if (NR % 2 == 1) {
        printf "%.6f", v[(NR + 1) / 2]
      } else {
        printf "%.6f", (v[NR / 2] + v[NR / 2 + 1]) / 2
      }
    }
  '
}

for (( round = 1; round <= ROUNDS; round += 1 )); do
  echo "[baseline] round $round/$ROUNDS"
  run_log="$TMP_DIR/round_${round}.log"
  (
    cd "$ROOT_DIR"
    DURATION="$DURATION" \
    WRK_THREADS="$WRK_THREADS" \
    CONNECTIONS="$CONNECTIONS" \
    WORKER_THREADS="$WORKER_THREADS" \
    UPSTREAM_TRANSPORT="$UPSTREAM_TRANSPORT" \
    REQUIRE_UPSTREAM_H2="$REQUIRE_UPSTREAM_H2" \
    MOCK_SCENARIO="$MOCK_SCENARIO" \
    scripts/perf_single_core_standard.sh
    if [[ "$INCLUDE_ALIAS_REMAP_SCENARIO" == "1" ]]; then
      DURATION="$DURATION" \
      WRK_THREADS="$WRK_THREADS" \
      CONNECTIONS="$CONNECTIONS" \
      WORKER_THREADS="$WORKER_THREADS" \
      UPSTREAM_TRANSPORT="$UPSTREAM_TRANSPORT" \
      REQUIRE_UPSTREAM_H2="$REQUIRE_UPSTREAM_H2" \
      MOCK_SCENARIO="$MOCK_SCENARIO" \
      scripts/perf_alias_remap_single_core.sh all
    fi
  ) | tee "$run_log"

  declare -A round_rps round_p99_us round_cpu round_rss
  while IFS= read -r line; do
    if [[ "$line" =~ ^([a-z_]+)[[:space:]]+wrk_requests=[^[:space:]]+[[:space:]]+wrk_rps=([0-9]+([.][0-9]+)?)[[:space:]]+wrk_p99=([^[:space:]]+) ]]; then
      scenario="${BASH_REMATCH[1]}"
      case "$scenario" in
        forward_nonstream_wrk|forward_stream_wrk|fc_inject_nonstream_wrk|fc_inject_stream_wrk|alias_remap_nonstream_wrk|alias_remap_stream_wrk)
          round_rps["$scenario"]="${BASH_REMATCH[2]}"
          p99_raw="${BASH_REMATCH[4]}"
          if p99_us="$(latency_to_us "$p99_raw")"; then
            round_p99_us["$scenario"]="$p99_us"
          fi
          ;;
        *) ;;
      esac
      continue
    fi

    if [[ "$line" =~ ^([a-z_]+)[[:space:]]+cpu_pct=([0-9]+([.][0-9]+)?)[[:space:]]+peak_rss_kb=([0-9]+)$ ]]; then
      scenario="${BASH_REMATCH[1]}"
      case "$scenario" in
        forward_nonstream_wrk|forward_stream_wrk|fc_inject_nonstream_wrk|fc_inject_stream_wrk|alias_remap_nonstream_wrk|alias_remap_stream_wrk)
          round_cpu["$scenario"]="${BASH_REMATCH[2]}"
          round_rss["$scenario"]="${BASH_REMATCH[4]}"
          ;;
        *) ;;
      esac
    fi
  done <"$run_log"

  for scenario in "${SCENARIOS[@]}"; do
    if [[ -z "${round_rps[$scenario]:-}" || -z "${round_p99_us[$scenario]:-}" || -z "${round_cpu[$scenario]:-}" || -z "${round_rss[$scenario]:-}" ]]; then
      echo "missing metrics in round $round for scenario: $scenario" >&2
      exit 1
    fi
    printf '%s\n' "${round_rps[$scenario]}" >> "$TMP_DIR/${scenario}.rps"
    printf '%s\n' "${round_p99_us[$scenario]}" >> "$TMP_DIR/${scenario}.p99_us"
    printf '%s\n' "${round_cpu[$scenario]}" >> "$TMP_DIR/${scenario}.cpu"
    printf '%s\n' "${round_rss[$scenario]}" >> "$TMP_DIR/${scenario}.rss"
  done
done

declare -A med_rps med_p99_us med_cpu med_rss
for scenario in "${SCENARIOS[@]}"; do
  med_rps["$scenario"]="$(median_of_file "$TMP_DIR/${scenario}.rps")"
  med_p99_us["$scenario"]="$(median_of_file "$TMP_DIR/${scenario}.p99_us")"
  med_cpu["$scenario"]="$(median_of_file "$TMP_DIR/${scenario}.cpu")"
  med_rss["$scenario"]="$(median_of_file "$TMP_DIR/${scenario}.rss")"
done

{
  echo "# Baseline Snapshot"
  echo
  echo "- Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
  echo "- Profile: single-core \`1x8\`"
  echo "- Rounds: $ROUNDS"
  echo "- Duration per scenario: $DURATION"
  echo "- Upstream transport: $UPSTREAM_TRANSPORT"
  echo "- Require upstream h2: $REQUIRE_UPSTREAM_H2"
  echo "- Mock scenario: $MOCK_SCENARIO"
  echo
  echo "## Results"
  echo
  echo "| Scenario | RPS | p99 | Avg Latency | CPU | RSS | Notes |"
  echo "|---|---:|---:|---:|---:|---:|---|"
  for scenario in "${SCENARIOS[@]}"; do
    p99_pretty="$(us_to_pretty_ms "${med_p99_us[$scenario]}")"
    printf '| `%s` | %.2f | %s | n/a | %.2f%% | %.0fKB | median of %d rounds |\n' \
      "$scenario" \
      "${med_rps[$scenario]}" \
      "$p99_pretty" \
      "${med_cpu[$scenario]}" \
      "${med_rss[$scenario]}" \
      "$ROUNDS"
  done
} > "$OUT_FILE"

echo "[baseline] wrote $OUT_FILE"
