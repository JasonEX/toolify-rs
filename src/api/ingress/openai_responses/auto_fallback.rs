use axum::response::Response;

use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::canonical::ProviderKind;
use crate::protocol::openai_responses::decoder::decode_responses_request;
use crate::protocol::openai_responses::ResponsesRequest;
use crate::routing::RouteTarget;
use crate::state::AppState;

use crate::api::common::prepare_upstream_io_request;
use crate::api::engine::fallback_common::{
    run_auto_inject_fallback, run_canonical_retry, run_preencoded_retry,
};
use crate::api::ingress::openai_responses::fc::apply_fc_inject_responses_wire;
use crate::api::ingress::openai_responses::io::{
    handle_non_streaming, handle_non_streaming_preencoded, handle_streaming,
};

pub(crate) struct OpenAiResponsesAutoFallbackCtx<'a> {
    pub(crate) state: &'a AppState,
    pub(crate) wire_request: &'a ResponsesRequest,
    pub(crate) route: RouteTarget<'a>,
    pub(crate) client_model: &'a str,
    pub(crate) requested_model: &'a str,
    pub(crate) request_seq: u64,
    pub(crate) request_id: uuid::Uuid,
}

pub(crate) async fn run_openai_responses_auto_fallback(
    ctx: OpenAiResponsesAutoFallbackCtx<'_>,
    err: CanonicalError,
) -> Result<Response, CanonicalError> {
    let prefer_wire = !ctx.state.config.features.enable_fc_error_retry
        && matches!(
            ctx.state.prepared_upstreams[ctx.route.upstream_index].provider_kind(),
            ProviderKind::OpenAiResponses
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
            run_openai_responses_wire_retry(&ctx, prepared_upstream).await
        },
        || async {
            let prepared_upstream = &ctx.state.prepared_upstreams[ctx.route.upstream_index];
            let provider = prepared_upstream.provider_kind();
            run_openai_responses_canonical_retry(&ctx, prepared_upstream, provider).await
        },
    )
    .await
}

async fn run_openai_responses_wire_retry(
    ctx: &OpenAiResponsesAutoFallbackCtx<'_>,
    prepared_upstream: &crate::transport::PreparedUpstream,
) -> Result<Response, CanonicalError> {
    let mut inject_wire = ctx.wire_request.clone();
    inject_wire.model.clear();
    inject_wire.model.push_str(ctx.route.actual_model);
    let inject_saved_tools =
        apply_fc_inject_responses_wire(&mut inject_wire, &ctx.state.config.features)?;
    let inject_fc_active = !inject_saved_tools.is_empty();
    let inject_stream = inject_wire.stream.unwrap_or(false);
    let inject_body = serde_json::to_vec(&inject_wire)
        .map(bytes::Bytes::from)
        .map_err(|e| CanonicalError::Translation(format!("Serialization error: {e}")))?;
    let response_model_passthrough_ok = ctx.route.actual_model == ctx.client_model;
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
        inject_fc_active,
        &inject_saved_tools,
        handle_streaming,
        |io_ctx, body, fc_active, saved_tools| async move {
            handle_non_streaming_preencoded(
                io_ctx,
                body,
                fc_active,
                saved_tools,
                response_model_passthrough_ok,
            )
            .await
        },
    )
    .await
}

async fn run_openai_responses_canonical_retry(
    ctx: &OpenAiResponsesAutoFallbackCtx<'_>,
    prepared_upstream: &crate::transport::PreparedUpstream,
    provider: ProviderKind,
) -> Result<Response, CanonicalError> {
    let mut inject_canonical = decode_responses_request(ctx.wire_request, ctx.request_id)?;
    inject_canonical.model.clear();
    inject_canonical.model.push_str(ctx.route.actual_model);
    let inject_saved_tools =
        fc::apply_fc_inject_take_tools(&mut inject_canonical, &ctx.state.config.features)?;
    let inject_fc_active = !inject_saved_tools.is_empty();
    let io_target = prepare_upstream_io_request(
        ctx.state,
        prepared_upstream,
        ctx.route.actual_model,
        inject_canonical.stream,
    );
    let response_model_passthrough_ok = ctx.route.actual_model == ctx.client_model;
    run_canonical_retry(
        &io_target,
        ctx.client_model,
        ctx.request_seq,
        provider,
        &inject_canonical,
        inject_fc_active,
        &inject_saved_tools,
        handle_streaming,
        |io_ctx, canonical, fc_active, saved_tools| async move {
            handle_non_streaming(
                io_ctx,
                canonical,
                fc_active,
                saved_tools,
                response_model_passthrough_ok,
            )
            .await
        },
    )
    .await
}
