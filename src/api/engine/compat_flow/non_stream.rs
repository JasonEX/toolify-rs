use std::sync::Arc;

use axum::response::Response;

use crate::api::common::CommonProbeRanges;
use crate::api::engine::pipeline::UpstreamIoRequest;
use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalRequest, CanonicalToolSpec, ProviderKind};
use crate::routing::RouteTarget;
use crate::state::AppState;
use crate::transport::{
    build_provider_headers_prepared, build_upstream_url_prepared, static_parsed_upstream_uri,
    static_parsed_upstream_url, PreparedUpstream,
};

use super::types::{AutoFallbackInput, CompatFlowSpec};

pub(crate) struct NonStreamFallbackInput<'a, S: CompatFlowSpec> {
    pub(crate) state: &'a Arc<AppState>,
    pub(crate) body: &'a bytes::Bytes,
    pub(crate) wire_request: Option<&'a S::WireRequest>,
    pub(crate) requested_model: &'a str,
    pub(crate) probe_ranges: Option<&'a CommonProbeRanges>,
    pub(crate) route: RouteTarget<'a>,
    pub(crate) prepared_upstream: &'a PreparedUpstream,
    pub(crate) provider: ProviderKind,
    pub(crate) client_model: &'a str,
    pub(crate) upstream_canonical: &'a CanonicalRequest,
    pub(crate) fc_active: bool,
    pub(crate) saved_tools: &'a [CanonicalToolSpec],
    pub(crate) auto_fallback_allowed: bool,
    pub(crate) request_seq: u64,
    pub(crate) request_id: uuid::Uuid,
}

pub(crate) async fn run_non_stream_with_fallback<S: CompatFlowSpec>(
    input: NonStreamFallbackInput<'_, S>,
) -> Result<Response, CanonicalError> {
    let url = build_upstream_url_prepared(
        input.prepared_upstream,
        input.route.actual_model,
        input.upstream_canonical.stream,
    );
    let parsed_url = static_parsed_upstream_url(
        input.prepared_upstream,
        input.route.actual_model,
        input.upstream_canonical.stream,
    );
    let parsed_uri = static_parsed_upstream_uri(
        input.prepared_upstream,
        input.route.actual_model,
        input.upstream_canonical.stream,
    );
    let proxy_url = input
        .prepared_upstream
        .proxy_for(input.upstream_canonical.stream);
    let upstream_headers = build_provider_headers_prepared(input.prepared_upstream);
    let io_ctx = UpstreamIoRequest {
        state: input.state.as_ref(),
        url: url.as_ref(),
        parsed_url,
        parsed_hyper_uri: parsed_uri,
        proxy_url,
        preconfigured_proxy_client: input.state.transport.preconfigured_proxy_client(proxy_url),
        upstream_headers,
        provider: input.provider,
        client_model: input.client_model,
    };

    let primary_result = S::handle_non_streaming(
        io_ctx,
        input.upstream_canonical,
        input.fc_active,
        input.saved_tools,
        input.route.actual_model == input.client_model,
    )
    .await;

    if !input.auto_fallback_allowed || input.fc_active {
        input.state.record_upstream_outcome(
            input.route.upstream_index,
            input.requested_model,
            &primary_result,
        );
        return primary_result;
    }

    match primary_result {
        Ok(response) => {
            input
                .state
                .record_upstream_success(input.route.upstream_index, input.requested_model);
            Ok(response)
        }
        Err(err) => {
            let request_ref = input.wire_request.ok_or_else(|| {
                CanonicalError::Internal("wire request missing for fallback retry".to_string())
            })?;
            S::run_auto_fallback(AutoFallbackInput {
                state: input.state.as_ref(),
                body: input.body,
                wire_request: request_ref,
                route: input.route,
                client_model: input.client_model,
                requested_model: input.requested_model,
                request_seq: input.request_seq,
                request_id: input.request_id,
                probe_ranges: input.probe_ranges,
                err,
            })
            .await
        }
    }
}
