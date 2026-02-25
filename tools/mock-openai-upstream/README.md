# mock-openai-upstream

Local benchmark-only mock upstream for toolify-rs.

Supports:

- OpenAI Chat: `/v1/chat/completions`, `/chat/completions`
- OpenAI Responses: `/v1/responses`, `/responses`
- Anthropic Messages: `/v1/messages`, `/messages`
- Gemini native: `/v1beta/models/*:generateContent`, `*:streamGenerateContent`
- Stats endpoint: `GET /_mock/stats`, reset endpoint: `POST /_mock/reset`

## Build

```bash
cargo build --release \
  --manifest-path tools/mock-openai-upstream/Cargo.toml \
  --target-dir target
```

## Run

```bash
MOCK_MODE=nonstream MOCK_TRANSPORT=h2c UPSTREAM_PORT=19001 target/release/mock_openai_upstream
```

`MOCK_MODE`:

- `nonstream`
- `stream`

`MOCK_TRANSPORT`:

- `auto` (default, accepts HTTP/1 and HTTP/2)
- `h2c` (HTTP/2 cleartext only, for local reproducible perf)

`MOCK_SCENARIO`:

- `text` (default)
- `full` (more complete payload/events)
- `error` (always return retriable upstream error)
