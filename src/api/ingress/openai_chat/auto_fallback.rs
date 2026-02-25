use axum::response::Response;

use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::canonical::{CanonicalToolSpec, ProviderKind};
use crate::protocol::openai_chat::decoder::decode_openai_chat_request;
use crate::protocol::openai_chat::OpenAiChatRequest;
use crate::routing::RouteTarget;
use crate::state::AppState;

use crate::api::common::{prepare_upstream_io_request, CommonProbeRanges};
use crate::api::engine::fallback_common::{
    run_auto_inject_fallback, run_canonical_retry, run_preencoded_retry,
};
use crate::api::ingress::openai_chat::fc::{
    apply_fc_inject_openai_wire, try_build_openai_simple_fc_inject_body_from_raw,
};
use crate::api::ingress::openai_chat::io::{
    handle_non_streaming, handle_non_streaming_preencoded, handle_streaming,
};

pub(crate) struct OpenAiChatAutoFallbackCtx<'a> {
    pub(crate) state: &'a AppState,
    pub(crate) body: &'a bytes::Bytes,
    pub(crate) wire_request: &'a OpenAiChatRequest,
    pub(crate) route: RouteTarget<'a>,
    pub(crate) client_model: &'a str,
    pub(crate) requested_model: &'a str,
    pub(crate) request_seq: u64,
    pub(crate) probe_ranges: Option<&'a CommonProbeRanges>,
}

pub(crate) async fn run_openai_chat_auto_fallback(
    ctx: OpenAiChatAutoFallbackCtx<'_>,
    err: CanonicalError,
) -> Result<Response, CanonicalError> {
    let prefer_wire = !ctx.state.config.features.enable_fc_error_retry
        && matches!(
            ctx.state.prepared_upstreams[ctx.route.upstream_index].provider_kind(),
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi
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
            run_openai_chat_wire_retry(&ctx, prepared_upstream).await
        },
        || async {
            let prepared_upstream = &ctx.state.prepared_upstreams[ctx.route.upstream_index];
            let provider = prepared_upstream.provider_kind();
            run_openai_chat_canonical_retry(&ctx, prepared_upstream, provider).await
        },
    )
    .await
}

async fn run_openai_chat_wire_retry(
    ctx: &OpenAiChatAutoFallbackCtx<'_>,
    prepared_upstream: &crate::transport::PreparedUpstream,
) -> Result<Response, CanonicalError> {
    if let Some((inject_body, inject_saved_tools, inject_stream)) =
        try_build_openai_simple_fc_inject_body_from_raw(
            ctx.body,
            ctx.route.actual_model,
            &ctx.state.config.features,
            ctx.probe_ranges,
        )?
    {
        return send_openai_chat_preencoded_retry(
            ctx,
            prepared_upstream,
            inject_body,
            inject_stream,
            inject_saved_tools.as_ref(),
        )
        .await;
    }

    let mut inject_wire = ctx.wire_request.clone();
    inject_wire.model.clear();
    inject_wire.model.push_str(ctx.route.actual_model);
    let inject_saved_tools =
        apply_fc_inject_openai_wire(&mut inject_wire, &ctx.state.config.features)?;
    let inject_stream = inject_wire.stream.unwrap_or(false);
    let inject_body = serde_json::to_vec(&inject_wire)
        .map(bytes::Bytes::from)
        .map_err(|e| CanonicalError::Translation(format!("Serialization error: {e}")))?;

    send_openai_chat_preencoded_retry(
        ctx,
        prepared_upstream,
        inject_body,
        inject_stream,
        &inject_saved_tools,
    )
    .await
}

async fn send_openai_chat_preencoded_retry(
    ctx: &OpenAiChatAutoFallbackCtx<'_>,
    prepared_upstream: &crate::transport::PreparedUpstream,
    inject_body: bytes::Bytes,
    inject_stream: bool,
    inject_saved_tools: &[CanonicalToolSpec],
) -> Result<Response, CanonicalError> {
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
        true,
        inject_saved_tools,
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

async fn run_openai_chat_canonical_retry(
    ctx: &OpenAiChatAutoFallbackCtx<'_>,
    prepared_upstream: &crate::transport::PreparedUpstream,
    provider: ProviderKind,
) -> Result<Response, CanonicalError> {
    let mut inject_canonical =
        decode_openai_chat_request(ctx.wire_request, ctx.state.request_uuid(ctx.request_seq))?;
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
    let response_model_passthrough_ok = ctx.route.actual_model == ctx.client_model;
    run_canonical_retry(
        &io_target,
        ctx.client_model,
        ctx.request_seq,
        provider,
        &inject_canonical,
        true,
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
