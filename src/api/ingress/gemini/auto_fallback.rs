use axum::response::Response;

use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::canonical::ProviderKind;
use crate::protocol::gemini::decoder::decode_gemini_request;
use crate::protocol::gemini::GeminiRequest;
use crate::routing::RouteTarget;
use crate::state::AppState;

use crate::api::common::prepare_upstream_io_request;
use crate::api::engine::fallback_common::{
    run_auto_inject_fallback, run_canonical_retry, run_preencoded_retry,
};
use crate::api::ingress::gemini::fc::apply_fc_inject_gemini_wire;
use crate::api::ingress::gemini::io::{
    handle_non_streaming, handle_non_streaming_preencoded, handle_streaming,
};

pub(crate) struct GeminiAutoFallbackCtx<'a> {
    pub(crate) state: &'a AppState,
    pub(crate) wire_request: &'a GeminiRequest,
    pub(crate) route: RouteTarget<'a>,
    pub(crate) model: &'a str,
    pub(crate) request_seq: u64,
    pub(crate) request_id: uuid::Uuid,
    pub(crate) is_stream: bool,
}

pub(crate) async fn run_gemini_auto_fallback(
    ctx: GeminiAutoFallbackCtx<'_>,
    err: CanonicalError,
) -> Result<Response, CanonicalError> {
    let prefer_wire = !ctx.state.config.features.enable_fc_error_retry
        && matches!(
            ctx.state.prepared_upstreams[ctx.route.upstream_index].provider_kind(),
            ProviderKind::Gemini
        );
    run_auto_inject_fallback(
        ctx.state,
        ctx.route,
        ctx.model,
        ctx.request_seq,
        err,
        prefer_wire,
        || async {
            let prepared_upstream = &ctx.state.prepared_upstreams[ctx.route.upstream_index];
            run_gemini_wire_retry(&ctx, prepared_upstream).await
        },
        || async {
            let prepared_upstream = &ctx.state.prepared_upstreams[ctx.route.upstream_index];
            let provider = prepared_upstream.provider_kind();
            run_gemini_canonical_retry(&ctx, prepared_upstream, provider).await
        },
    )
    .await
}

async fn run_gemini_wire_retry(
    ctx: &GeminiAutoFallbackCtx<'_>,
    prepared_upstream: &crate::transport::PreparedUpstream,
) -> Result<Response, CanonicalError> {
    let mut inject_wire = ctx.wire_request.clone();
    let inject_saved_tools =
        apply_fc_inject_gemini_wire(&mut inject_wire, &ctx.state.config.features)?;
    let inject_body = serde_json::to_vec(&inject_wire)
        .map(bytes::Bytes::from)
        .map_err(|e| CanonicalError::Translation(format!("Serialization error: {e}")))?;
    let io_target = prepare_upstream_io_request(
        ctx.state,
        prepared_upstream,
        ctx.route.actual_model,
        ctx.is_stream,
    );
    run_preencoded_retry(
        &io_target,
        ctx.model,
        ctx.request_seq,
        inject_body,
        ctx.is_stream,
        true,
        &inject_saved_tools,
        handle_streaming,
        handle_non_streaming_preencoded,
    )
    .await
}

async fn run_gemini_canonical_retry(
    ctx: &GeminiAutoFallbackCtx<'_>,
    prepared_upstream: &crate::transport::PreparedUpstream,
    provider: ProviderKind,
) -> Result<Response, CanonicalError> {
    let mut inject_canonical = decode_gemini_request(ctx.wire_request, ctx.model, ctx.request_id)?;
    inject_canonical.stream = ctx.is_stream;
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
        ctx.model,
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
