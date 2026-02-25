# Performance Contract (Zero Regression)

This repository uses a strict zero-regression policy for hot-path changes.

## Fixed Benchmark Profile

- Script: `scripts/perf_single_core_standard.sh`
- Fixed parameters:
  - `WORKER_THREADS=1`
  - `WRK_THREADS=1`
  - `CONNECTIONS=8`
  - `DURATION=10s` (unless explicitly overridden)
  - `WARMUP_REQUESTS=200`
  - `RUNTIME_THREAD_STACK_SIZE_KB=512`
- Scenarios:
  - `forward_nonstream_wrk`
  - `forward_stream_wrk`
  - `fc_inject_nonstream_wrk`
  - `fc_inject_stream_wrk`

## Baseline Source

- Canonical baseline file: `artifacts/perf/baseline_single_core_1x8_pinned.md`

## Gate Rules

- Round count: `9` by default, with jitter-adaptive extension up to `12` (`MAX_EXTRA_ROUNDS=3`).
- For each scenario, median metrics across rounds must satisfy:
  - `p99_new <= p99_base`
  - `rps_new >= rps_base`
  - `cpu_per_krps_new <= cpu_per_krps_base`
  - `rss_new <= rss_base`
- Per-scenario round stability rule:
  - At least `ceil(total_rounds * 0.7778)` rounds must be non-regressing (or higher when `MIN_PASS_ROUNDS` is explicitly set).
- Jitter control:
  - If any scenario has `RPS CV% > 5.0`, gate auto-adds rounds until stable or `MAX_EXTRA_ROUNDS` is reached.
- Memory hard limit:
  - `Peak RSS < 10 MB` (`10240 KB`) in the fixed profile.

## Local Gate Command

```bash
scripts/perf_dual_gate.sh
```

## CI / Nightly Gate

- Workflow: `.github/workflows/nightly-perf.yml`
- Pinned gate is blocking.
- Unpinned gate is observe-only and non-blocking.

## Docker PGO Default

- `Dockerfile` supports `PGO_MODE=auto|on|off` (default: `auto`).
- `auto`: use `PGO_PROFDATA` when present; fallback to non-PGO release if missing or invalid.
- `on`: require `PGO_PROFDATA` and fail build if unavailable.
- `off`: force non-PGO release build.
- Default `PGO_PROFDATA`: `artifacts/pgo/merged.profdata`.
