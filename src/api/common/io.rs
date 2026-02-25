use axum::http::HeaderMap;

use crate::error::CanonicalError;
use crate::protocol::canonical::ProviderKind;
use crate::state::AppState;
use crate::transport::{
    build_provider_headers_prepared, build_upstream_url_prepared, static_parsed_upstream_uri,
    static_parsed_upstream_url, PreparedUpstream,
};

#[derive(Clone, Copy)]
pub(crate) struct UpstreamIoRequest<'a> {
    pub(crate) state: &'a AppState,
    pub(crate) url: &'a str,
    pub(crate) parsed_url: Option<&'a url::Url>,
    pub(crate) parsed_hyper_uri: Option<&'a http::Uri>,
    pub(crate) proxy_url: Option<&'a str>,
    pub(crate) preconfigured_proxy_client: Option<&'a reqwest::Client>,
    pub(crate) upstream_headers: &'a HeaderMap,
    pub(crate) provider: ProviderKind,
    pub(crate) client_model: &'a str,
}

pub(crate) struct PreparedUpstreamIoRequest<'a> {
    state: &'a AppState,
    url: std::borrow::Cow<'a, str>,
    parsed_url: Option<&'a url::Url>,
    parsed_hyper_uri: Option<&'a http::Uri>,
    proxy_url: Option<&'a str>,
    preconfigured_proxy_client: Option<&'a reqwest::Client>,
    upstream_headers: &'a HeaderMap,
    provider: ProviderKind,
}

impl PreparedUpstreamIoRequest<'_> {
    #[inline]
    pub(crate) fn io_ctx<'a>(&'a self, client_model: &'a str) -> UpstreamIoRequest<'a> {
        UpstreamIoRequest {
            state: self.state,
            url: self.url.as_ref(),
            parsed_url: self.parsed_url,
            parsed_hyper_uri: self.parsed_hyper_uri,
            proxy_url: self.proxy_url,
            preconfigured_proxy_client: self.preconfigured_proxy_client,
            upstream_headers: self.upstream_headers,
            provider: self.provider,
            client_model,
        }
    }
}

#[inline]
pub(crate) fn prepare_upstream_io_request<'a>(
    state: &'a AppState,
    prepared_upstream: &'a PreparedUpstream,
    actual_model: &'a str,
    stream: bool,
) -> PreparedUpstreamIoRequest<'a> {
    let proxy_url = prepared_upstream.proxy_for(stream);
    PreparedUpstreamIoRequest {
        state,
        url: build_upstream_url_prepared(prepared_upstream, actual_model, stream),
        parsed_url: static_parsed_upstream_url(prepared_upstream, actual_model, stream),
        parsed_hyper_uri: static_parsed_upstream_uri(prepared_upstream, actual_model, stream),
        proxy_url,
        preconfigured_proxy_client: state.transport.preconfigured_proxy_client(proxy_url),
        upstream_headers: build_provider_headers_prepared(prepared_upstream),
        provider: prepared_upstream.provider_kind(),
    }
}

#[inline]
pub(crate) async fn send_non_streaming_bytes(
    state: &AppState,
    url: &str,
    parsed_url: Option<&url::Url>,
    parsed_hyper_uri: Option<&http::Uri>,
    proxy_url: Option<&str>,
    preconfigured_proxy_client: Option<&reqwest::Client>,
    upstream_headers: &HeaderMap,
    upstream_body: bytes::Bytes,
) -> Result<(http::StatusCode, bytes::Bytes), CanonicalError> {
    if state.transport.hyper_passthrough_enabled_for(proxy_url) {
        use http_body_util::BodyExt as _;

        let response = if let Some(parsed_hyper_uri) = parsed_hyper_uri {
            state
                .transport
                .send_request_uri(
                    parsed_hyper_uri,
                    http::Method::POST,
                    upstream_headers,
                    upstream_body,
                )
                .await?
        } else {
            state
                .transport
                .send_request_uri_str(url, http::Method::POST, upstream_headers, upstream_body)
                .await?
        };
        let status = response.status();
        let (_, body) = response.into_parts();
        let body_bytes = body
            .collect()
            .await
            .map(http_body_util::Collected::to_bytes)
            .map_err(|e| CanonicalError::Transport(format!("Failed to read response body: {e}")))?;
        return Ok((status, body_bytes));
    }

    let response = if let Some(parsed_url) = parsed_url {
        state
            .transport
            .send_request_url_with_client(
                parsed_url,
                http::Method::POST,
                upstream_headers,
                upstream_body,
                proxy_url,
                preconfigured_proxy_client,
            )
            .await?
    } else {
        state
            .transport
            .send_request_with_client(
                url,
                http::Method::POST,
                upstream_headers,
                upstream_body,
                proxy_url,
                preconfigured_proxy_client,
            )
            .await?
    };
    let status = response.status();
    let body_bytes = response
        .bytes()
        .await
        .map_err(|e| CanonicalError::Transport(format!("Failed to read response body: {e}")))?;
    Ok((status, body_bytes))
}
