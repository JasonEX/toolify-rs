use std::sync::Arc;

use axum::response::Response;

use crate::api::engine::channel_b::core::{ChannelBFastPathOutcome, ChannelBPlan};
use crate::api::engine::compat_flow::{
    AutoFallbackInput, CompatFlowSpec, FcNonStreamCtx, NoToolsCtx,
};
use crate::api::engine::failover::{
    run_openai_responses_fc_non_stream, run_openai_responses_no_tools_non_stream,
};
use crate::api::engine::pipeline::{CommonRequestProbe, UpstreamIoRequest};
use crate::config::FeaturesConfig;
use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalRequest, CanonicalToolSpec, IngressApi, ProviderKind};
use crate::protocol::openai_responses::decoder::{
    decode_responses_request, decode_responses_request_owned,
};
use crate::protocol::openai_responses::ResponsesRequest;
use crate::state::AppState;

use super::auto_fallback::{run_openai_responses_auto_fallback, OpenAiResponsesAutoFallbackCtx};
use super::channel_b::run_channel_b_fast_path;
use super::fc::apply_fc_inject_responses_wire;
use super::io::{handle_non_streaming, handle_non_streaming_preencoded, handle_streaming};
use super::parse::parse_openai_responses_probe;

pub(super) struct OpenAiResponsesSpec;

impl CompatFlowSpec for OpenAiResponsesSpec {
    type WireRequest = ResponsesRequest;

    const INGRESS: IngressApi = IngressApi::OpenAiResponses;

    fn parse_probe(body: &bytes::Bytes) -> Result<CommonRequestProbe<'_>, CanonicalError> {
        parse_openai_responses_probe(body)
    }

    fn parse_wire_request(body: &bytes::Bytes) -> Result<Self::WireRequest, CanonicalError> {
        serde_json::from_slice(body).map_err(|e| {
            CanonicalError::InvalidRequest(format!("Invalid OpenAI Responses request body: {e}"))
        })
    }

    async fn run_channel_b_fast_path<'a>(
        state: &'a Arc<AppState>,
        body: &'a bytes::Bytes,
        request_seq: &mut Option<u64>,
        plan: ChannelBPlan<'a>,
    ) -> ChannelBFastPathOutcome<'a> {
        run_channel_b_fast_path(state, body, request_seq, plan).await
    }

    fn supports_wire_inject_provider(provider: ProviderKind) -> bool {
        matches!(provider, ProviderKind::OpenAiResponses)
    }

    fn set_wire_model(wire_request: &mut Self::WireRequest, actual_model: &str) {
        wire_request.model.clear();
        wire_request.model.push_str(actual_model);
    }

    fn apply_wire_inject(
        wire_request: &mut Self::WireRequest,
        features: &FeaturesConfig,
    ) -> Result<Vec<CanonicalToolSpec>, CanonicalError> {
        apply_fc_inject_responses_wire(wire_request, features)
    }

    fn wire_stream_requested(wire_request: &Self::WireRequest) -> bool {
        wire_request.stream.unwrap_or(false)
    }

    fn encode_wire(wire_request: &Self::WireRequest) -> Result<bytes::Bytes, CanonicalError> {
        serde_json::to_vec(wire_request)
            .map(bytes::Bytes::from)
            .map_err(|e| CanonicalError::Translation(format!("Serialization error: {e}")))
    }

    fn decode_wire_ref(
        wire_request: &Self::WireRequest,
        request_id: uuid::Uuid,
    ) -> Result<CanonicalRequest, CanonicalError> {
        decode_responses_request(wire_request, request_id)
    }

    fn decode_wire_owned(
        wire_request: Self::WireRequest,
        request_id: uuid::Uuid,
    ) -> Result<CanonicalRequest, CanonicalError> {
        decode_responses_request_owned(wire_request, request_id)
    }

    async fn run_no_tools_non_stream(ctx: NoToolsCtx<'_>) -> Result<Response, CanonicalError> {
        run_openai_responses_no_tools_non_stream(
            ctx.state,
            ctx.body,
            ctx.model_value_range,
            ctx.route_candidates,
            ctx.route,
            ctx.request_id,
            ctx.client_model,
        )
        .await
    }

    async fn run_fc_non_stream(ctx: FcNonStreamCtx<'_>) -> Result<Response, CanonicalError> {
        run_openai_responses_fc_non_stream(
            ctx.state,
            ctx.route_candidates,
            ctx.route,
            ctx.upstream_canonical,
            ctx.saved_tools,
            ctx.client_model,
        )
        .await
    }

    async fn run_auto_fallback(
        input: AutoFallbackInput<'_, Self::WireRequest>,
    ) -> Result<Response, CanonicalError> {
        run_openai_responses_auto_fallback(
            OpenAiResponsesAutoFallbackCtx {
                state: input.state,
                wire_request: input.wire_request,
                route: input.route,
                client_model: input.client_model,
                requested_model: input.requested_model,
                request_seq: input.request_seq,
                request_id: input.request_id,
            },
            input.err,
        )
        .await
    }

    async fn handle_streaming<'a>(
        io_ctx: UpstreamIoRequest<'a>,
        upstream_body: bytes::Bytes,
        request_seq: u64,
        fc_active: bool,
        saved_tools: &'a [CanonicalToolSpec],
    ) -> Result<Response, CanonicalError> {
        handle_streaming(io_ctx, upstream_body, request_seq, fc_active, saved_tools).await
    }

    async fn handle_non_streaming<'a>(
        io_ctx: UpstreamIoRequest<'a>,
        upstream_canonical: &'a CanonicalRequest,
        fc_active: bool,
        saved_tools: &'a [CanonicalToolSpec],
        same_model: bool,
    ) -> Result<Response, CanonicalError> {
        handle_non_streaming(
            io_ctx,
            upstream_canonical,
            fc_active,
            saved_tools,
            same_model,
        )
        .await
    }

    async fn handle_non_streaming_preencoded<'a>(
        io_ctx: UpstreamIoRequest<'a>,
        upstream_body: bytes::Bytes,
        fc_active: bool,
        saved_tools: &'a [CanonicalToolSpec],
        same_model: bool,
    ) -> Result<Response, CanonicalError> {
        handle_non_streaming_preencoded(io_ctx, upstream_body, fc_active, saved_tools, same_model)
            .await
    }

    fn wire_inject_fc_active(saved_tools: &[CanonicalToolSpec]) -> bool {
        !saved_tools.is_empty()
    }

    fn canonical_fc_active(fc_active: bool, saved_tools: &[CanonicalToolSpec]) -> bool {
        fc_active && !saved_tools.is_empty()
    }
}
