use axum::http::HeaderMap;
use axum::response::Response;

use crate::error::CanonicalError;
use crate::protocol::canonical::{IngressApi, ProviderKind};
use crate::state::AppState;

pub(crate) fn is_protocol_passthrough(provider: ProviderKind, ingress: IngressApi) -> bool {
    matches!(
        (provider, ingress),
        (
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi,
            IngressApi::OpenAiChat
        ) | (ProviderKind::OpenAiResponses, IngressApi::OpenAiResponses)
            | (ProviderKind::Anthropic, IngressApi::Anthropic)
            | (ProviderKind::Gemini, IngressApi::Gemini)
    )
}

/// Raw non-streaming passthrough: forward request JSON and return upstream body bytes directly.
pub(crate) async fn passthrough_non_streaming_bytes(
    state: &AppState,
    url: &str,
    proxy_url: Option<&str>,
    upstream_headers: &HeaderMap,
    upstream_body: bytes::Bytes,
) -> Result<Response, CanonicalError> {
    if state.transport.hyper_passthrough_enabled_for(proxy_url) {
        let response = state
            .transport
            .send_request_uri_str(url, http::Method::POST, upstream_headers, upstream_body)
            .await?;
        return build_passthrough_non_streaming_hyper_response(response).await;
    }

    let response = state
        .transport
        .send_request(
            url,
            http::Method::POST,
            upstream_headers,
            upstream_body,
            proxy_url,
        )
        .await?;
    build_passthrough_non_streaming_response(response).await
}

/// Raw non-streaming passthrough using a pre-parsed URL.
pub(crate) async fn passthrough_non_streaming_url_bytes(
    state: &AppState,
    url: &url::Url,
    proxy_url: Option<&str>,
    upstream_headers: &HeaderMap,
    upstream_body: bytes::Bytes,
) -> Result<Response, CanonicalError> {
    let response = state
        .transport
        .send_request_url(
            url,
            http::Method::POST,
            upstream_headers,
            upstream_body,
            proxy_url,
        )
        .await?;
    build_passthrough_non_streaming_response(response).await
}

/// Raw non-streaming passthrough using a pre-parsed URI and hyper transport.
pub(crate) async fn passthrough_non_streaming_uri_bytes(
    state: &AppState,
    uri: &http::Uri,
    upstream_headers: &HeaderMap,
    upstream_body: bytes::Bytes,
) -> Result<Response, CanonicalError> {
    let response = state
        .transport
        .send_request_uri(uri, http::Method::POST, upstream_headers, upstream_body)
        .await?;
    build_passthrough_non_streaming_hyper_response(response).await
}

async fn build_passthrough_non_streaming_response(
    response: reqwest::Response,
) -> Result<Response, CanonicalError> {
    let status = response.status();

    if !status.is_success() {
        let body_bytes = response
            .bytes()
            .await
            .map_err(|e| CanonicalError::Transport(format!("Failed to read response body: {e}")))?;
        return Err(CanonicalError::Upstream {
            status: status.as_u16(),
            message: sanitize_upstream_error(&body_bytes),
        });
    }

    let content_type = response.headers().get(http::header::CONTENT_TYPE).cloned();
    let body = axum::body::Body::from_stream(response.bytes_stream());
    let mut passthrough = Response::new(body);
    *passthrough.status_mut() = status;
    passthrough.headers_mut().insert(
        http::header::CONTENT_TYPE,
        content_type.unwrap_or_else(|| http::HeaderValue::from_static("application/json")),
    );
    Ok(passthrough)
}

async fn build_passthrough_non_streaming_hyper_response(
    response: http::Response<hyper::body::Incoming>,
) -> Result<Response, CanonicalError> {
    use http_body_util::BodyExt;

    let status = response.status();
    let content_type = response.headers().get(http::header::CONTENT_TYPE).cloned();
    let (_, body) = response.into_parts();

    if !status.is_success() {
        let body_bytes = body
            .collect()
            .await
            .map(http_body_util::Collected::to_bytes)
            .map_err(|e| CanonicalError::Transport(format!("Failed to read response body: {e}")))?;
        return Err(CanonicalError::Upstream {
            status: status.as_u16(),
            message: sanitize_upstream_error(&body_bytes),
        });
    }

    let body = axum::body::Body::new(body);
    let mut passthrough = Response::new(body);
    *passthrough.status_mut() = status;
    passthrough.headers_mut().insert(
        http::header::CONTENT_TYPE,
        content_type.unwrap_or_else(|| http::HeaderValue::from_static("application/json")),
    );
    Ok(passthrough)
}

/// Raw streaming passthrough: forward request JSON and stream upstream bytes directly.
pub(crate) async fn passthrough_streaming_bytes(
    state: &AppState,
    url: &str,
    proxy_url: Option<&str>,
    upstream_headers: &HeaderMap,
    upstream_body: bytes::Bytes,
) -> Result<Response, CanonicalError> {
    if state.transport.hyper_passthrough_enabled_for(proxy_url) {
        let response = state
            .transport
            .send_stream_uri_str(url, http::Method::POST, upstream_headers, upstream_body)
            .await?;
        return build_passthrough_streaming_hyper_response(response).await;
    }

    let response = state
        .transport
        .send_stream(
            url,
            http::Method::POST,
            upstream_headers,
            upstream_body,
            proxy_url,
        )
        .await?;
    build_passthrough_streaming_response(response).await
}

/// Raw streaming passthrough using a pre-parsed URL.
pub(crate) async fn passthrough_streaming_url_bytes(
    state: &AppState,
    url: &url::Url,
    proxy_url: Option<&str>,
    upstream_headers: &HeaderMap,
    upstream_body: bytes::Bytes,
) -> Result<Response, CanonicalError> {
    let response = state
        .transport
        .send_stream_url(
            url,
            http::Method::POST,
            upstream_headers,
            upstream_body,
            proxy_url,
        )
        .await?;
    build_passthrough_streaming_response(response).await
}

/// Raw streaming passthrough using a pre-parsed URI and hyper transport.
pub(crate) async fn passthrough_streaming_uri_bytes(
    state: &AppState,
    uri: &http::Uri,
    upstream_headers: &HeaderMap,
    upstream_body: bytes::Bytes,
) -> Result<Response, CanonicalError> {
    let response = state
        .transport
        .send_stream_uri(uri, http::Method::POST, upstream_headers, upstream_body)
        .await?;
    build_passthrough_streaming_hyper_response(response).await
}

async fn build_passthrough_streaming_response(
    response: reqwest::Response,
) -> Result<Response, CanonicalError> {
    let status = response.status();
    if !status.is_success() {
        let body_bytes = response
            .bytes()
            .await
            .map_err(|e| CanonicalError::Transport(format!("Failed to read error body: {e}")))?;
        return Err(CanonicalError::Upstream {
            status: status.as_u16(),
            message: sanitize_upstream_error(&body_bytes),
        });
    }

    let content_type = response
        .headers()
        .get(http::header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| http::HeaderValue::from_static("text/event-stream"));
    let byte_stream = response.bytes_stream();
    let body = axum::body::Body::from_stream(byte_stream);

    let mut passthrough = Response::new(body);
    *passthrough.status_mut() = status;
    let headers = passthrough.headers_mut();
    headers.insert(http::header::CONTENT_TYPE, content_type);
    headers.insert(
        http::header::CACHE_CONTROL,
        http::HeaderValue::from_static("no-cache"),
    );
    headers.insert(
        http::header::CONNECTION,
        http::HeaderValue::from_static("keep-alive"),
    );
    Ok(passthrough)
}

async fn build_passthrough_streaming_hyper_response(
    response: http::Response<hyper::body::Incoming>,
) -> Result<Response, CanonicalError> {
    use http_body_util::BodyExt;

    let status = response.status();
    let content_type = response
        .headers()
        .get(http::header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| http::HeaderValue::from_static("text/event-stream"));

    let (_, body) = response.into_parts();
    if !status.is_success() {
        let collected = body
            .collect()
            .await
            .map_err(|e| CanonicalError::Transport(format!("Failed to read error body: {e}")))?;
        let body_bytes = collected.to_bytes();
        return Err(CanonicalError::Upstream {
            status: status.as_u16(),
            message: sanitize_upstream_error(&body_bytes),
        });
    }

    let body = axum::body::Body::new(body);

    let mut passthrough = Response::new(body);
    *passthrough.status_mut() = status;
    let headers = passthrough.headers_mut();
    headers.insert(http::header::CONTENT_TYPE, content_type);
    headers.insert(
        http::header::CACHE_CONTROL,
        http::HeaderValue::from_static("no-cache"),
    );
    headers.insert(
        http::header::CONNECTION,
        http::HeaderValue::from_static("keep-alive"),
    );
    Ok(passthrough)
}

/// Sanitize an upstream error body to avoid leaking internal details.
///
/// Attempts to extract just the `error.message` field from JSON responses.
/// Falls back to a truncated UTF-8 representation capped at 500 chars.
pub(crate) fn sanitize_upstream_error(body: &[u8]) -> String {
    const MAX_LEN: usize = 500;

    // Try to extract error.message from JSON
    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(body) {
        if let Some(msg) = json
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            let truncated = if msg.len() > MAX_LEN {
                format!("{}...", &msg[..MAX_LEN])
            } else {
                msg.to_string()
            };
            return truncated;
        }
        // Anthropic-style: { "type": "error", "error": { "message": "..." } }
        if let Some(msg) = json
            .get("error")
            .and_then(|e| e.as_object())
            .and_then(|o| o.get("message"))
            .and_then(|m| m.as_str())
        {
            let truncated = if msg.len() > MAX_LEN {
                format!("{}...", &msg[..MAX_LEN])
            } else {
                msg.to_string()
            };
            return truncated;
        }
    }

    // Fallback: lossy UTF-8 truncated
    let raw = String::from_utf8_lossy(body);
    if raw.len() > MAX_LEN {
        format!("{}...", &raw[..MAX_LEN])
    } else {
        raw.to_string()
    }
}
