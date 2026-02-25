#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_FILE="${BASELINE_FILE:-$ROOT_DIR/artifacts/perf/baseline_single_core_1x8_pinned.md}"
ROUNDS="${ROUNDS:-9}"
MIN_PASS_ROUNDS="${MIN_PASS_ROUNDS:-}"
MIN_PASS_RATIO="${MIN_PASS_RATIO:-0.7778}"
MAX_CV_PERCENT="${MAX_CV_PERCENT:-5.0}"
MAX_EXTRA_ROUNDS="${MAX_EXTRA_ROUNDS:-3}"
DURATION="${DURATION:-10s}"
RSS_LIMIT_KB="${RSS_LIMIT_KB:-10240}"
INCLUDE_ALIAS_REMAP_SCENARIO="${INCLUDE_ALIAS_REMAP_SCENARIO:-0}" # 0 | 1
OUT_DIR="${OUT_DIR:-$ROOT_DIR/artifacts/perf}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_MD="$OUT_DIR/zero_regression_gate_${TIMESTAMP}.md"
OUT_LATEST="$OUT_DIR/zero_regression_gate_latest.md"

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

if [[ ! -f "$BASELINE_FILE" ]]; then
  echo "baseline file not found: $BASELINE_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

declare -A BASE_RPS BASE_P99_US BASE_CPU BASE_RSS BASE_CPU_PER_KRPS
declare -A PASS_ROUNDS
declare -A MEDIAN_RPS MEDIAN_P99_US MEDIAN_CPU MEDIAN_RSS MEDIAN_CPU_PER_KRPS RPS_CV_PCT

for scenario in "${SCENARIOS[@]}"; do
  PASS_ROUNDS["$scenario"]=0
done

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

float_ge() {
  local left="$1"
  local right="$2"
  awk -v l="$left" -v r="$right" 'BEGIN { exit(l + 0 >= r + 0 ? 0 : 1) }'
}

float_le() {
  local left="$1"
  local right="$2"
  awk -v l="$left" -v r="$right" 'BEGIN { exit(l + 0 <= r + 0 ? 0 : 1) }'
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

cv_pct_of_file() {
  local file="$1"
  awk '
    {
      x = $1 + 0
      sum += x
      sum_sq += x * x
      n += 1
    }
    END {
      if (n == 0 || sum == 0) {
        printf "0.000000"
        exit
      }
      mean = sum / n
      variance = (sum_sq / n) - (mean * mean)
      if (variance < 0) {
        variance = 0
      }
      stddev = sqrt(variance)
      printf "%.6f", (stddev / mean) * 100.0
    }
  ' "$file"
}

required_pass_rounds() {
  local rounds="$1"
  local ratio_required
  ratio_required="$(awk -v total="$rounds" -v ratio="$MIN_PASS_RATIO" '
    BEGIN {
      v = total * ratio
      if (v == int(v)) {
        printf "%d", int(v)
      } else {
        printf "%d", int(v) + 1
      }
    }
  ')"

  if [[ -n "$MIN_PASS_ROUNDS" ]]; then
    if (( MIN_PASS_ROUNDS > ratio_required )); then
      printf '%d' "$MIN_PASS_ROUNDS"
      return
    fi
  fi
  printf '%d' "$ratio_required"
}

pct_delta() {
  local old="$1"
  local new="$2"
  awk -v o="$old" -v n="$new" '
    BEGIN {
      if (o == 0) { printf "0.00"; exit }
      printf "%.2f", ((n - o) / o) * 100.0
    }
  '
}

cpu_per_krps() {
  local cpu="$1"
  local rps="$2"
  awk -v c="$cpu" -v r="$rps" '
    BEGIN {
      if (r == 0) { printf "0.000000"; exit }
      printf "%.6f", (c * 1000.0) / r
    }
  '
}

parse_baseline() {
  while IFS=$'\t' read -r scenario rps p99 cpu rss; do
    case "$scenario" in
      forward_nonstream_wrk|forward_stream_wrk|fc_inject_nonstream_wrk|fc_inject_stream_wrk|alias_remap_nonstream_wrk|alias_remap_stream_wrk)
        BASE_RPS["$scenario"]="$rps"
        BASE_CPU["$scenario"]="$cpu"
        BASE_RSS["$scenario"]="$rss"
        if ! p99_us="$(latency_to_us "$p99")"; then
          echo "failed to parse baseline p99 for $scenario: $p99" >&2
          exit 1
        fi
        BASE_P99_US["$scenario"]="$p99_us"
        ;;
      *) ;;
    esac
  done < <(
    awk -F'|' '
      /^## Results$/ {
        in_results = 1
        next
      }
      /^## / {
        if (in_results) {
          in_results = 0
        }
      }
      !in_results { next }
      /^\| `[^`]+` \|/ {
        cpu_field = $6
        rss_field = $7
        if (cpu_field !~ /%/ || rss_field !~ /KB/) {
          next
        }

        scenario=$2
        gsub(/`/, "", scenario)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", scenario)

        rps=$3
        gsub(/[[:space:],]/, "", rps)

        p99=$4
        gsub(/[[:space:]]/, "", p99)

        cpu=$6
        gsub(/[[:space:],%]/, "", cpu)

        rss=$7
        gsub(/[[:space:],]/, "", rss)
        sub(/KB$/, "", rss)

        print scenario "\t" rps "\t" p99 "\t" cpu "\t" rss
      }
    ' "$BASELINE_FILE"
  )

  for scenario in "${SCENARIOS[@]}"; do
    if [[ -z "${BASE_RPS[$scenario]:-}" || -z "${BASE_P99_US[$scenario]:-}" || -z "${BASE_CPU[$scenario]:-}" || -z "${BASE_RSS[$scenario]:-}" ]]; then
      echo "baseline missing scenario: $scenario" >&2
      exit 1
    fi
    BASE_CPU_PER_KRPS["$scenario"]="$(cpu_per_krps "${BASE_CPU[$scenario]}" "${BASE_RPS[$scenario]}")"
  done
}

run_one_round() {
  local round="$1"
  local run_log="$TMP_DIR/round_${round}.log"
  declare -A round_rps round_p99_us round_cpu round_rss

  echo "[gate] round $round/${target_rounds:-$ROUNDS}"
  (
    cd "$ROOT_DIR"
    DURATION="$DURATION" \
      WRK_THREADS=1 \
      CONNECTIONS=8 \
      WORKER_THREADS=1 \
      scripts/perf_single_core_standard.sh
    if [[ "$INCLUDE_ALIAS_REMAP_SCENARIO" == "1" ]]; then
      DURATION="$DURATION" \
        WRK_THREADS=1 \
        CONNECTIONS=8 \
        WORKER_THREADS=1 \
        scripts/perf_alias_remap_single_core.sh all
    fi
  ) | tee "$run_log"

  while IFS= read -r line; do
    if [[ "$line" =~ ^([a-z_]+)[[:space:]]+wrk_requests=[^[:space:]]+[[:space:]]+wrk_rps=([0-9]+([.][0-9]+)?)[[:space:]]+wrk_p99=([^[:space:]]+) ]]; then
      local scenario="${BASH_REMATCH[1]}"
      local rps="${BASH_REMATCH[2]}"
      local p99_raw="${BASH_REMATCH[4]}"
      case "$scenario" in
        forward_nonstream_wrk|forward_stream_wrk|fc_inject_nonstream_wrk|fc_inject_stream_wrk|alias_remap_nonstream_wrk|alias_remap_stream_wrk)
          round_rps["$scenario"]="$rps"
          if p99_us="$(latency_to_us "$p99_raw")"; then
            round_p99_us["$scenario"]="$p99_us"
          fi
          ;;
        *) ;;
      esac
      continue
    fi

    if [[ "$line" =~ ^([a-z_]+)[[:space:]]+cpu_pct=([0-9]+([.][0-9]+)?)[[:space:]]+peak_rss_kb=([0-9]+)$ ]]; then
      local scenario="${BASH_REMATCH[1]}"
      local cpu="${BASH_REMATCH[2]}"
      local rss="${BASH_REMATCH[4]}"
      case "$scenario" in
        forward_nonstream_wrk|forward_stream_wrk|fc_inject_nonstream_wrk|fc_inject_stream_wrk|alias_remap_nonstream_wrk|alias_remap_stream_wrk)
          round_cpu["$scenario"]="$cpu"
          round_rss["$scenario"]="$rss"
          ;;
        *) ;;
      esac
    fi
  done <"$run_log"

  for scenario in "${SCENARIOS[@]}"; do
    if [[ -z "${round_rps[$scenario]:-}" || -z "${round_p99_us[$scenario]:-}" || -z "${round_cpu[$scenario]:-}" || -z "${round_rss[$scenario]:-}" ]]; then
      echo "round $round missing metrics for scenario: $scenario" >&2
      exit 1
    fi
    printf '%s\n' "${round_rps[$scenario]}" >>"$TMP_DIR/${scenario}.rps"
    printf '%s\n' "${round_p99_us[$scenario]}" >>"$TMP_DIR/${scenario}.p99_us"
    printf '%s\n' "${round_cpu[$scenario]}" >>"$TMP_DIR/${scenario}.cpu"
    printf '%s\n' "${round_rss[$scenario]}" >>"$TMP_DIR/${scenario}.rss"

    local round_ok=true
    if ! float_ge "${round_rps[$scenario]}" "${BASE_RPS[$scenario]}"; then
      round_ok=false
    fi
    if ! float_le "${round_p99_us[$scenario]}" "${BASE_P99_US[$scenario]}"; then
      round_ok=false
    fi
    round_cpu_per_krps="$(cpu_per_krps "${round_cpu[$scenario]}" "${round_rps[$scenario]}")"
    if ! float_le "$round_cpu_per_krps" "${BASE_CPU_PER_KRPS[$scenario]}"; then
      round_ok=false
    fi
    if ! float_le "${round_rss[$scenario]}" "$RSS_LIMIT_KB"; then
      round_ok=false
    fi
    if [[ "$round_ok" == true ]]; then
      PASS_ROUNDS["$scenario"]=$((PASS_ROUNDS["$scenario"] + 1))
    fi
  done
}

needs_extra_round() {
  local high_jitter=false
  for scenario in "${SCENARIOS[@]}"; do
    local cv
    cv="$(cv_pct_of_file "$TMP_DIR/${scenario}.rps")"
    if ! float_le "$cv" "$MAX_CV_PERCENT"; then
      high_jitter=true
      break
    fi
  done
  [[ "$high_jitter" == true ]]
}

parse_baseline

target_rounds="$ROUNDS"
max_total_rounds=$((ROUNDS + MAX_EXTRA_ROUNDS))
round=1
while (( round <= target_rounds )); do
  run_one_round "$round"
  if (( round == target_rounds )) && (( target_rounds < max_total_rounds )) && needs_extra_round; then
    target_rounds=$((target_rounds + 1))
    echo "[gate] detected high RPS jitter (CV > ${MAX_CV_PERCENT}%), extending to $target_rounds rounds"
  fi
  round=$((round + 1))
done

required_pass="$(required_pass_rounds "$target_rounds")"

overall_ok=true
for scenario in "${SCENARIOS[@]}"; do
  MEDIAN_RPS["$scenario"]="$(median_of_file "$TMP_DIR/${scenario}.rps")"
  MEDIAN_P99_US["$scenario"]="$(median_of_file "$TMP_DIR/${scenario}.p99_us")"
  MEDIAN_CPU["$scenario"]="$(median_of_file "$TMP_DIR/${scenario}.cpu")"
  MEDIAN_RSS["$scenario"]="$(median_of_file "$TMP_DIR/${scenario}.rss")"
  MEDIAN_CPU_PER_KRPS["$scenario"]="$(cpu_per_krps "${MEDIAN_CPU[$scenario]}" "${MEDIAN_RPS[$scenario]}")"
  RPS_CV_PCT["$scenario"]="$(cv_pct_of_file "$TMP_DIR/${scenario}.rps")"

  if ! float_ge "${MEDIAN_RPS[$scenario]}" "${BASE_RPS[$scenario]}"; then
    overall_ok=false
  fi
  if ! float_le "${MEDIAN_P99_US[$scenario]}" "${BASE_P99_US[$scenario]}"; then
    overall_ok=false
  fi
  if ! float_le "${MEDIAN_CPU_PER_KRPS[$scenario]}" "${BASE_CPU_PER_KRPS[$scenario]}"; then
    overall_ok=false
  fi
  if ! float_le "${MEDIAN_RSS[$scenario]}" "$RSS_LIMIT_KB"; then
    overall_ok=false
  fi
  if (( PASS_ROUNDS[$scenario] < required_pass )); then
    overall_ok=false
  fi
done

{
  echo "# Zero Regression Gate Result"
  echo
  echo "- Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
  echo "- Baseline: \`$BASELINE_FILE\`"
  echo "- Profile: single-core \`1x8\`"
  echo "- Rounds planned: $ROUNDS"
  echo "- Rounds executed: $target_rounds"
  echo "- Extra rounds allowed: $MAX_EXTRA_ROUNDS"
  echo "- Jitter threshold (RPS CV): ${MAX_CV_PERCENT}%"
  echo "- Min passing rounds per scenario: $required_pass/$target_rounds (ratio=${MIN_PASS_RATIO})"
  echo "- RSS limit: ${RSS_LIMIT_KB} KB"
  echo "- Contract priority: p99 (no regression) -> throughput (no regression) -> CPU efficiency (no regression) -> memory cap"
  echo
  echo "| Scenario | Baseline RPS | Median RPS | ΔRPS% | RPS CV% | Baseline p99(us) | Median p99(us) | Δp99% | Baseline CPU% | Median CPU% | ΔCPU% | Baseline CPU/kRPS | Median CPU/kRPS | ΔCPU/kRPS% | Baseline RSS KB | Median RSS KB | ΔRSS% | Pass Rounds |"
  echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
  for scenario in "${SCENARIOS[@]}"; do
    drps="$(pct_delta "${BASE_RPS[$scenario]}" "${MEDIAN_RPS[$scenario]}")"
    dp99="$(pct_delta "${BASE_P99_US[$scenario]}" "${MEDIAN_P99_US[$scenario]}")"
    dcpu="$(pct_delta "${BASE_CPU[$scenario]}" "${MEDIAN_CPU[$scenario]}")"
    dcpuk="$(pct_delta "${BASE_CPU_PER_KRPS[$scenario]}" "${MEDIAN_CPU_PER_KRPS[$scenario]}")"
    drss="$(pct_delta "${BASE_RSS[$scenario]}" "${MEDIAN_RSS[$scenario]}")"
    printf '| `%s` | %.2f | %.2f | %s%% | %.2f | %.2f | %.2f | %s%% | %.2f | %.2f | %s%% | %.4f | %.4f | %s%% | %.0f | %.0f | %s%% | %d/%d |\n' \
      "$scenario" \
      "${BASE_RPS[$scenario]}" "${MEDIAN_RPS[$scenario]}" "$drps" "${RPS_CV_PCT[$scenario]}" \
      "${BASE_P99_US[$scenario]}" "${MEDIAN_P99_US[$scenario]}" "$dp99" \
      "${BASE_CPU[$scenario]}" "${MEDIAN_CPU[$scenario]}" "$dcpu" \
      "${BASE_CPU_PER_KRPS[$scenario]}" "${MEDIAN_CPU_PER_KRPS[$scenario]}" "$dcpuk" \
      "${BASE_RSS[$scenario]}" "${MEDIAN_RSS[$scenario]}" "$drss" \
      "${PASS_ROUNDS[$scenario]}" "$target_rounds"
  done
  echo
  if [[ "$overall_ok" == true ]]; then
    echo "## Verdict"
    echo
    echo "PASS: zero-regression contract satisfied."
  else
    echo "## Verdict"
    echo
    echo "FAIL: zero-regression contract violated."
  fi
} >"$OUT_MD"

cp "$OUT_MD" "$OUT_LATEST"
cat "$OUT_MD"

if [[ "$overall_ok" != true ]]; then
  exit 1
fi
