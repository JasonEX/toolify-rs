use axum::response::Response;

use crate::api::common::CommonProbeRanges;
use crate::api::engine::pipeline::UpstreamIoRequest;
use crate::error::CanonicalError;
use crate::protocol::canonical::ProviderKind;
use crate::routing::RouteTarget;
use crate::state::AppState;
use crate::transport::{
    build_provider_headers_prepared, build_upstream_url_prepared, static_parsed_upstream_uri,
    static_parsed_upstream_url, PreparedUpstream,
};

use super::types::CompatFlowSpec;

pub(crate) async fn try_raw_inject_fast_path<S: CompatFlowSpec>(
    state: &AppState,
    body: &bytes::Bytes,
    prepared_upstream: &PreparedUpstream,
    route: RouteTarget<'_>,
    provider: ProviderKind,
    client_model: &str,
    request_seq: Option<u64>,
    probe_ranges: Option<&CommonProbeRanges>,
    enabled: bool,
) -> Result<Option<Response>, CanonicalError> {
    if !enabled {
        return Ok(None);
    }

    let Some(raw_fast) = S::try_raw_inject_fast_path(
        body,
        route.actual_model,
        &state.config.features,
        probe_ranges,
    )?
    else {
        return Ok(None);
    };

    let inject_url =
        build_upstream_url_prepared(prepared_upstream, route.actual_model, raw_fast.stream);
    let inject_parsed_url =
        static_parsed_upstream_url(prepared_upstream, route.actual_model, raw_fast.stream);
    let inject_hyper_uri =
        static_parsed_upstream_uri(prepared_upstream, route.actual_model, raw_fast.stream);
    let proxy_url = prepared_upstream.proxy_for(raw_fast.stream);
    let inject_headers = build_provider_headers_prepared(prepared_upstream);
    let io_ctx = UpstreamIoRequest {
        state,
        url: inject_url.as_ref(),
        parsed_url: inject_parsed_url,
        parsed_hyper_uri: inject_hyper_uri,
        proxy_url,
        preconfigured_proxy_client: state.transport.preconfigured_proxy_client(proxy_url),
        upstream_headers: inject_headers,
        provider,
        client_model,
    };

    if raw_fast.stream {
        let stream_request_seq = request_seq.unwrap_or_else(|| state.next_request_seq());
        let response = S::handle_streaming(
            io_ctx,
            raw_fast.body,
            stream_request_seq,
            raw_fast.fc_active,
            &raw_fast.saved_tools,
        )
        .await?;
        return Ok(Some(response));
    }

    let response = S::handle_non_streaming_preencoded(
        io_ctx,
        raw_fast.body,
        raw_fast.fc_active,
        &raw_fast.saved_tools,
        route.actual_model == client_model,
    )
    .await?;
    Ok(Some(response))
}
