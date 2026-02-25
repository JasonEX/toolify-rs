# syntax=docker/dockerfile:1.7
# Build stage
FROM rust:bookworm AS builder
WORKDIR /app

ARG PGO_MODE=auto
ARG PGO_PROFDATA=artifacts/pgo/merged.profdata

# Cache dependencies separately from source changes.
COPY Cargo.toml Cargo.lock ./
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    set -eux; \
    if [ "${PGO_MODE}" = "on" ]; then \
      echo "skip dependency preheat for PGO_MODE=on"; \
    else \
      mkdir -p src; \
      printf 'pub fn __docker_cache_placeholder() {}\n' > src/lib.rs; \
      printf 'fn main() {}\n' > src/main.rs; \
      cargo build --locked --release; \
      rm -rf src; \
    fi

COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    set -eux; \
    resolve_profdata_path() { \
      case "$1" in \
        /*) printf '%s\n' "$1" ;; \
        *) printf '%s/%s\n' "$(pwd -P)" "$1" ;; \
      esac; \
    }; \
    build_release() { cargo build --locked --release; }; \
    build_with_pgo() { \
      profdata_path="$(resolve_profdata_path "$1")"; \
      test -f "${profdata_path}"; \
      RUSTFLAGS="-Cprofile-use=${profdata_path} -Cllvm-args=-pgo-warn-missing-function" cargo build --locked --release; \
    }; \
    case "${PGO_MODE}" in \
      off) \
        build_release; \
        ;; \
      on) \
        build_with_pgo "${PGO_PROFDATA}"; \
        ;; \
      auto) \
        if [ -f "${PGO_PROFDATA}" ]; then \
          if ! build_with_pgo "${PGO_PROFDATA}"; then \
            echo "PGO build failed, fallback to non-PGO release build."; \
            build_release; \
          fi; \
        else \
          echo "PGO profile not found at ${PGO_PROFDATA}, fallback to non-PGO release build."; \
          build_release; \
        fi; \
        ;; \
      *) \
        echo "invalid PGO_MODE=${PGO_MODE}, expected auto|on|off"; \
        exit 2; \
        ;; \
    esac

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
RUN useradd -r -s /bin/false toolify
WORKDIR /app
COPY --from=builder /app/target/release/toolify /app/toolify
COPY config.example.yaml /app/config.example.yaml
COPY config.example.yaml /app/config.yaml
USER toolify
EXPOSE 8000
CMD ["/app/toolify"]
