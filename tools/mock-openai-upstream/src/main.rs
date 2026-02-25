use std::convert::Infallible;
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use http::{header, HeaderValue, Method, Request, Response, StatusCode, Version};
use http_body_util::BodyExt;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as AutoBuilder;
use tokio::net::TcpListener;

const DEFAULT_UPSTREAM_PORT: u16 = 19_001;

#[derive(Copy, Clone)]
enum MockMode {
    Nonstream,
    Stream,
}

#[derive(Copy, Clone)]
enum MockScenario {
    Text,
    Code,
    Full,
    Error,
}

#[derive(Copy, Clone)]
enum MockTransport {
    Auto,
    H2c,
}

#[derive(Copy, Clone)]
enum ProviderApi {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
    GeminiGenerateContent,
}

struct ProtocolStats {
    h1: AtomicU64,
    h2: AtomicU64,
    other: AtomicU64,
}

impl ProtocolStats {
    const fn new() -> Self {
        Self {
            h1: AtomicU64::new(0),
            h2: AtomicU64::new(0),
            other: AtomicU64::new(0),
        }
    }

    fn record(&self, version: Version) {
        match version {
            Version::HTTP_10 | Version::HTTP_11 => {
                self.h1.fetch_add(1, Ordering::Relaxed);
            }
            Version::HTTP_2 => {
                self.h2.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                self.other.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.h1.load(Ordering::Relaxed),
            self.h2.load(Ordering::Relaxed),
            self.other.load(Ordering::Relaxed),
        )
    }

    fn reset(&self) {
        self.h1.store(0, Ordering::Relaxed);
        self.h2.store(0, Ordering::Relaxed);
        self.other.store(0, Ordering::Relaxed);
    }
}

struct MockState {
    mode: MockMode,
    scenario: MockScenario,
    transport: MockTransport,
    stats: ProtocolStats,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let port = env_u16("UPSTREAM_PORT", DEFAULT_UPSTREAM_PORT);
    let mode = parse_mode();
    let scenario = parse_scenario();
    let transport = parse_transport();
    let state = Arc::new(MockState {
        mode,
        scenario,
        transport,
        stats: ProtocolStats::new(),
    });

    let listener = TcpListener::bind(("127.0.0.1", port))
        .await
        .unwrap_or_else(|err| panic!("failed to bind mock upstream on 127.0.0.1:{port}: {err}"));

    let conn_builder = match transport {
        MockTransport::Auto => AutoBuilder::new(TokioExecutor::new()),
        MockTransport::H2c => AutoBuilder::new(TokioExecutor::new()).http2_only(),
    };

    loop {
        let (stream, remote_addr) = match listener.accept().await {
            Ok((stream, remote_addr)) => (stream, remote_addr),
            Err(err) => {
                eprintln!("accept error: {err}");
                continue;
            }
        };
        let io = TokioIo::new(stream);
        let conn_builder = conn_builder.clone();
        let service_state = Arc::clone(&state);
        let service = service_fn(move |request: Request<Incoming>| {
            let state_ref = Arc::clone(&service_state);
            async move { Ok::<_, Infallible>(handle_request(request, &state_ref).await) }
        });

        tokio::spawn(async move {
            if let Err(err) = conn_builder.serve_connection(io, service).await {
                eprintln!("mock upstream connection error from {remote_addr}: {err}");
            }
        });
    }
}

fn env_u16(name: &str, default: u16) -> u16 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(default)
}

fn parse_mode() -> MockMode {
    match env::var("MOCK_MODE").as_deref() {
        Ok("stream") => MockMode::Stream,
        Ok("nonstream") | Err(_) => MockMode::Nonstream,
        Ok(other) => {
            eprintln!("unknown MOCK_MODE '{other}', fallback to nonstream");
            MockMode::Nonstream
        }
    }
}

fn parse_scenario() -> MockScenario {
    match env::var("MOCK_SCENARIO").as_deref() {
        Ok("code") => MockScenario::Code,
        Ok("full") => MockScenario::Full,
        Ok("error") => MockScenario::Error,
        Ok("text") | Err(_) => MockScenario::Text,
        Ok(other) => {
            eprintln!("unknown MOCK_SCENARIO '{other}', fallback to text");
            MockScenario::Text
        }
    }
}

fn parse_transport() -> MockTransport {
    match env::var("MOCK_TRANSPORT").as_deref() {
        Ok("h2c") => MockTransport::H2c,
        Ok("auto") | Err(_) => MockTransport::Auto,
        Ok(other) => {
            eprintln!("unknown MOCK_TRANSPORT '{other}', fallback to auto");
            MockTransport::Auto
        }
    }
}

async fn handle_request(request: Request<Incoming>, state: &Arc<MockState>) -> Response<Full<Bytes>> {
    let (parts, body) = request.into_parts();
    state.stats.record(parts.version);
    drain_request_body(body).await;

    let method = parts.method;
    let path = parts.uri.path();

    if method == Method::GET && path == "/_mock/stats" {
        return stats_response(state);
    }
    if method == Method::POST && path == "/_mock/reset" {
        state.stats.reset();
        return simple_response_static(StatusCode::OK, "application/json", br#"{"ok":true}"#);
    }
    if method != Method::POST {
        return simple_response_static(
            StatusCode::METHOD_NOT_ALLOWED,
            "application/json",
            br#"{"error":"method_not_allowed"}"#,
        );
    }

    let Some(provider) = provider_for_path(path) else {
        return simple_response_static(
            StatusCode::NOT_FOUND,
            "application/json",
            br#"{"error":"not_found"}"#,
        );
    };

    if matches!(state.scenario, MockScenario::Error) {
        return simple_response_static(
            StatusCode::SERVICE_UNAVAILABLE,
            "application/json",
            br#"{"error":"mock_injected_error"}"#,
        );
    }

    let is_stream = matches!(state.mode, MockMode::Stream);
    if is_stream {
        streaming_response(provider, state.scenario)
    } else {
        non_streaming_response(provider, state.scenario)
    }
}

async fn drain_request_body(mut body: Incoming) {
    while let Some(frame_result) = body.frame().await {
        if frame_result.is_err() {
            break;
        }
    }
}

fn provider_for_path(path: &str) -> Option<ProviderApi> {
    match path {
        "/v1/chat/completions" | "/chat/completions" => Some(ProviderApi::OpenAiChat),
        "/v1/responses" | "/responses" => Some(ProviderApi::OpenAiResponses),
        "/v1/messages" | "/messages" => Some(ProviderApi::AnthropicMessages),
        _ if path.starts_with("/v1beta/models/")
            && (path.contains(":generateContent") || path.contains(":streamGenerateContent")) =>
        {
            Some(ProviderApi::GeminiGenerateContent)
        }
        _ => None,
    }
}

fn stats_response(state: &MockState) -> Response<Full<Bytes>> {
    let (h1, h2, other) = state.stats.snapshot();
    let mode = match state.mode {
        MockMode::Nonstream => "nonstream",
        MockMode::Stream => "stream",
    };
    let scenario = match state.scenario {
        MockScenario::Text => "text",
        MockScenario::Code => "code",
        MockScenario::Full => "full",
        MockScenario::Error => "error",
    };
    let transport = match state.transport {
        MockTransport::Auto => "auto",
        MockTransport::H2c => "h2c",
    };
    let body = format!(
        "{{\"mode\":\"{mode}\",\"scenario\":\"{scenario}\",\"transport\":\"{transport}\",\"h1\":{h1},\"h2\":{h2},\"other\":{other}}}"
    );
    simple_response(
        StatusCode::OK,
        "application/json",
        Bytes::from(body.into_bytes()),
    )
}

fn non_streaming_response(provider: ProviderApi, scenario: MockScenario) -> Response<Full<Bytes>> {
    let body = match (provider, scenario) {
        (ProviderApi::OpenAiChat, MockScenario::Text) => OPENAI_CHAT_NONSTREAM_TEXT,
        (ProviderApi::OpenAiChat, MockScenario::Code) => OPENAI_CHAT_NONSTREAM_CODE,
        (ProviderApi::OpenAiChat, MockScenario::Full) => OPENAI_CHAT_NONSTREAM_FULL,
        (ProviderApi::OpenAiResponses, MockScenario::Code) => OPENAI_RESPONSES_NONSTREAM_TEXT,
        (ProviderApi::OpenAiResponses, MockScenario::Text) => OPENAI_RESPONSES_NONSTREAM_TEXT,
        (ProviderApi::OpenAiResponses, MockScenario::Full) => OPENAI_RESPONSES_NONSTREAM_FULL,
        (ProviderApi::AnthropicMessages, MockScenario::Code) => ANTHROPIC_NONSTREAM_TEXT,
        (ProviderApi::AnthropicMessages, MockScenario::Text) => ANTHROPIC_NONSTREAM_TEXT,
        (ProviderApi::AnthropicMessages, MockScenario::Full) => ANTHROPIC_NONSTREAM_FULL,
        (ProviderApi::GeminiGenerateContent, MockScenario::Code) => GEMINI_NONSTREAM_TEXT,
        (ProviderApi::GeminiGenerateContent, MockScenario::Text) => GEMINI_NONSTREAM_TEXT,
        (ProviderApi::GeminiGenerateContent, MockScenario::Full) => GEMINI_NONSTREAM_FULL,
        (_, MockScenario::Error) => br#"{"error":"mock_injected_error"}"#,
    };
    simple_response_static(StatusCode::OK, "application/json", body)
}

fn streaming_response(provider: ProviderApi, scenario: MockScenario) -> Response<Full<Bytes>> {
    let body = match (provider, scenario) {
        (ProviderApi::OpenAiChat, MockScenario::Text) => OPENAI_CHAT_STREAM_TEXT,
        (ProviderApi::OpenAiChat, MockScenario::Code) => OPENAI_CHAT_STREAM_CODE,
        (ProviderApi::OpenAiChat, MockScenario::Full) => OPENAI_CHAT_STREAM_FULL,
        (ProviderApi::OpenAiResponses, MockScenario::Code) => OPENAI_RESPONSES_STREAM_TEXT,
        (ProviderApi::OpenAiResponses, MockScenario::Text) => OPENAI_RESPONSES_STREAM_TEXT,
        (ProviderApi::OpenAiResponses, MockScenario::Full) => OPENAI_RESPONSES_STREAM_FULL,
        (ProviderApi::AnthropicMessages, MockScenario::Code) => ANTHROPIC_STREAM_TEXT,
        (ProviderApi::AnthropicMessages, MockScenario::Text) => ANTHROPIC_STREAM_TEXT,
        (ProviderApi::AnthropicMessages, MockScenario::Full) => ANTHROPIC_STREAM_FULL,
        (ProviderApi::GeminiGenerateContent, MockScenario::Code) => GEMINI_STREAM_TEXT,
        (ProviderApi::GeminiGenerateContent, MockScenario::Text) => GEMINI_STREAM_TEXT,
        (ProviderApi::GeminiGenerateContent, MockScenario::Full) => GEMINI_STREAM_FULL,
        (_, MockScenario::Error) => b"data: {\"error\":\"mock_injected_error\"}\n\n",
    };
    let mut response = simple_response_static(StatusCode::OK, "text/event-stream", body);
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}

fn simple_response(
    status: StatusCode,
    content_type: &'static str,
    body: Bytes,
) -> Response<Full<Bytes>> {
    let mut response = Response::new(Full::new(body));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    response
}

fn simple_response_static(
    status: StatusCode,
    content_type: &'static str,
    body: &'static [u8],
) -> Response<Full<Bytes>> {
    simple_response(status, content_type, Bytes::from_static(body))
}

const OPENAI_CHAT_NONSTREAM_TEXT: &[u8] = br#"{"id":"chatcmpl-mock","object":"chat.completion","created":1,"model":"m1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#;
const OPENAI_CHAT_NONSTREAM_CODE: &[u8] = br#"{"id":"chatcmpl-mock","object":"chat.completion","created":1,"model":"m1","choices":[{"index":0,"message":{"role":"assistant","content":"```html\n<div>ok</div>\n```"},"finish_reason":"stop"}]}"#;
const OPENAI_CHAT_NONSTREAM_FULL: &[u8] = br#"{"id":"chatcmpl-mock","object":"chat.completion","created":1,"model":"m1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
const OPENAI_CHAT_STREAM_TEXT: &[u8] = b"data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"o\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"k\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
const OPENAI_CHAT_STREAM_CODE: &[u8] = b"data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"```html\\n<div>\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok</div>\\n```\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
const OPENAI_CHAT_STREAM_FULL: &[u8] = b"data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\ndata: [DONE]\n\n";

const OPENAI_RESPONSES_NONSTREAM_TEXT: &[u8] = br#"{"id":"resp_mock","object":"response","created_at":1,"model":"m1","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}]}"#;
const OPENAI_RESPONSES_NONSTREAM_FULL: &[u8] = br#"{"id":"resp_mock","object":"response","created_at":1,"model":"m1","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}"#;
const OPENAI_RESPONSES_STREAM_TEXT: &[u8] = b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_mock\",\"model\":\"m1\",\"output\":[]}}\n\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"ok\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_mock\",\"status\":\"completed\"}}\n\ndata: [DONE]\n\n";
const OPENAI_RESPONSES_STREAM_FULL: &[u8] = b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_mock\",\"model\":\"m1\",\"output\":[]}}\n\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"ok\"}\n\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]}}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_mock\",\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\ndata: [DONE]\n\n";

const ANTHROPIC_NONSTREAM_TEXT: &[u8] = br#"{"id":"msg_mock","type":"message","role":"assistant","model":"claude-3-5-haiku-latest","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#;
const ANTHROPIC_NONSTREAM_FULL: &[u8] = br#"{"id":"msg_mock","type":"message","role":"assistant","model":"claude-3-5-haiku-latest","content":[{"type":"thinking","thinking":"analysis"},{"type":"text","text":"ok"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":2}}"#;
const ANTHROPIC_STREAM_TEXT: &[u8] = b"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_mock\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-5-haiku-latest\",\"content\":[]}}\n\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\ndata: {\"type\":\"message_stop\"}\n\n";
const ANTHROPIC_STREAM_FULL: &[u8] = b"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_mock\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-5-haiku-latest\",\"content\":[]}}\n\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"analysis\"}}\n\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\ndata: {\"type\":\"message_stop\"}\n\n";

const GEMINI_NONSTREAM_TEXT: &[u8] = br#"{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}"#;
const GEMINI_NONSTREAM_FULL: &[u8] = br#"{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP","index":0,"safetyRatings":[]}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}"#;
const GEMINI_STREAM_TEXT: &[u8] = b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"o\"}],\"role\":\"model\"},\"index\":0}]}\n\ndata: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"k\"}],\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1,\"totalTokenCount\":2}}\n\n";
const GEMINI_STREAM_FULL: &[u8] = b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}],\"role\":\"model\"},\"index\":0}]}\n\ndata: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}],\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0,\"safetyRatings\":[]}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1,\"totalTokenCount\":2}}\n\n";
