use axum::response::Response;

use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::anthropic::decoder::decode_anthropic_request;
use crate::protocol::anthropic::AnthropicRequest;
use crate::protocol::canonical::ProviderKind;
use crate::routing::RouteTarget;
use crate::state::AppState;

use crate::api::common::prepare_upstream_io_request;
use crate::api::engine::fallback_common::{
    run_auto_inject_fallback, run_canonical_retry, run_preencoded_retry,
};
use crate::api::ingress::anthropic::fc::apply_fc_inject_anthropic_wire;
use crate::api::ingress::anthropic::io::{
    handle_non_streaming, handle_non_streaming_preencoded, handle_streaming,
};

pub(crate) struct AnthropicAutoFallbackCtx<'a> {
    pub(crate) state: &'a AppState,
    pub(crate) wire_request: &'a AnthropicRequest,
    pub(crate) route: RouteTarget<'a>,
    pub(crate) client_model: &'a str,
    pub(crate) requested_model: &'a str,
    pub(crate) request_seq: u64,
    pub(crate) request_id: uuid::Uuid,
}

pub(crate) async fn run_anthropic_auto_fallback(
    ctx: AnthropicAutoFallbackCtx<'_>,
    err: CanonicalError,
) -> Result<Response, CanonicalError> {
    let prefer_wire = !ctx.state.config.features.enable_fc_error_retry
        && matches!(
            ctx.state.prepared_upstreams[ctx.route.upstream_index].provider_kind(),
            ProviderKind::Anthropic
        );
    run_auto_inject_fallback(
        ctx.state,
        ctx.route,
        ctx.requested_model,
        ctx.request_seq,
        err,
        prefer_wire,
        || async {
            let prepared_upstream = &ctx.state.prepared_upstreams[ctx.route.upstream_index];
            run_anthropic_wire_retry(&ctx, prepared_upstream).await
        },
        || async {
            let prepared_upstream = &ctx.state.prepared_upstreams[ctx.route.upstream_index];
            let provider = prepared_upstream.provider_kind();
            run_anthropic_canonical_retry(&ctx, prepared_upstream, provider).await
        },
    )
    .await
}

async fn run_anthropic_wire_retry(
    ctx: &AnthropicAutoFallbackCtx<'_>,
    prepared_upstream: &crate::transport::PreparedUpstream,
) -> Result<Response, CanonicalError> {
    let mut inject_wire = ctx.wire_request.clone();
    inject_wire.model.clear();
    inject_wire.model.push_str(ctx.route.actual_model);
    let inject_saved_tools =
        apply_fc_inject_anthropic_wire(&mut inject_wire, &ctx.state.config.features)?;
    let inject_stream = inject_wire.stream.unwrap_or(false);
    let inject_body = serde_json::to_vec(&inject_wire)
        .map(bytes::Bytes::from)
        .map_err(|e| CanonicalError::Translation(format!("Serialization error: {e}")))?;
    let io_target = prepare_upstream_io_request(
        ctx.state,
        prepared_upstream,
        ctx.route.actual_model,
        inject_stream,
    );
    run_preencoded_retry(
        &io_target,
        ctx.client_model,
        ctx.request_seq,
        inject_body,
        inject_stream,
        true,
        &inject_saved_tools,
        handle_streaming,
        handle_non_streaming_preencoded,
    )
    .await
}

async fn run_anthropic_canonical_retry(
    ctx: &AnthropicAutoFallbackCtx<'_>,
    prepared_upstream: &crate::transport::PreparedUpstream,
    provider: ProviderKind,
) -> Result<Response, CanonicalError> {
    let mut inject_canonical = decode_anthropic_request(ctx.wire_request, ctx.request_id)?;
    inject_canonical.model.clear();
    inject_canonical.model.push_str(ctx.route.actual_model);
    let inject_saved_tools =
        fc::apply_fc_inject_take_tools(&mut inject_canonical, &ctx.state.config.features)?;
    let io_target = prepare_upstream_io_request(
        ctx.state,
        prepared_upstream,
        ctx.route.actual_model,
        inject_canonical.stream,
    );
    run_canonical_retry(
        &io_target,
        ctx.client_model,
        ctx.request_seq,
        provider,
        &inject_canonical,
        true,
        &inject_saved_tools,
        handle_streaming,
        handle_non_streaming,
    )
    .await
}
