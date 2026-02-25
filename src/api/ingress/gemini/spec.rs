use std::borrow::Cow;
use std::sync::Arc;

use axum::response::Response;

use crate::api::engine::channel_b::core::{ChannelBFastPathOutcome, ChannelBPlan};
use crate::api::engine::compat_flow::{
    AutoFallbackInput, CompatFlowSpec, FcNonStreamCtx, NoToolsCtx,
};
use crate::api::engine::failover::{run_gemini_fc_non_stream, run_gemini_no_tools_non_stream};
use crate::api::engine::pipeline::{CommonRequestProbe, UpstreamIoRequest};
use crate::config::FeaturesConfig;
use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalRequest, CanonicalToolSpec, IngressApi, ProviderKind};
use crate::protocol::gemini::decoder::{decode_gemini_request, decode_gemini_request_owned};
use crate::protocol::gemini::GeminiRequest;
use crate::state::AppState;

use super::auto_fallback::{run_gemini_auto_fallback, GeminiAutoFallbackCtx};
use super::channel_b::run_channel_b_fast_path;
use super::fc::apply_fc_inject_gemini_wire;
use super::io::{handle_non_streaming, handle_non_streaming_preencoded, handle_streaming};
use super::parse::parse_gemini_has_tools;

pub(super) const INGRESS: IngressApi = IngressApi::Gemini;

#[derive(Clone)]
pub(super) struct GeminiWireRequest {
    request: GeminiRequest,
    requested_model: String,
    stream_requested: bool,
}

pub(super) struct GeminiSpec;

impl CompatFlowSpec for GeminiSpec {
    type WireRequest = GeminiWireRequest;

    const INGRESS: IngressApi = INGRESS;

    fn parse_probe(body: &bytes::Bytes) -> Result<CommonRequestProbe<'_>, CanonicalError> {
        Ok(CommonRequestProbe {
            model: Cow::Borrowed(""),
            stream: None,
            has_tools: parse_gemini_has_tools(body)?,
            ranges: None,
        })
    }

    fn parse_wire_request(body: &bytes::Bytes) -> Result<Self::WireRequest, CanonicalError> {
        let request = serde_json::from_slice(body).map_err(|e| {
            CanonicalError::InvalidRequest(format!("Invalid Gemini request body: {e}"))
        })?;
        Ok(Self::WireRequest {
            request,
            requested_model: String::new(),
            stream_requested: false,
        })
    }

    fn set_request_context(
        wire_request: &mut Self::WireRequest,
        requested_model: &str,
        stream_requested: bool,
    ) {
        wire_request.requested_model.clear();
        wire_request.requested_model.push_str(requested_model);
        wire_request.stream_requested = stream_requested;
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
        matches!(provider, ProviderKind::Gemini)
    }

    fn set_wire_model(_wire_request: &mut Self::WireRequest, _actual_model: &str) {}

    fn apply_wire_inject(
        wire_request: &mut Self::WireRequest,
        features: &FeaturesConfig,
    ) -> Result<Vec<CanonicalToolSpec>, CanonicalError> {
        apply_fc_inject_gemini_wire(&mut wire_request.request, features)
    }

    fn wire_stream_requested(wire_request: &Self::WireRequest) -> bool {
        wire_request.stream_requested
    }

    fn encode_wire(wire_request: &Self::WireRequest) -> Result<bytes::Bytes, CanonicalError> {
        serde_json::to_vec(&wire_request.request)
            .map(bytes::Bytes::from)
            .map_err(|e| CanonicalError::Translation(format!("Serialization error: {e}")))
    }

    fn decode_wire_ref(
        wire_request: &Self::WireRequest,
        request_id: uuid::Uuid,
    ) -> Result<CanonicalRequest, CanonicalError> {
        if wire_request.requested_model.is_empty() {
            return Err(CanonicalError::Internal(
                "Gemini requested model context is missing".to_string(),
            ));
        }
        let mut canonical = decode_gemini_request(
            &wire_request.request,
            &wire_request.requested_model,
            request_id,
        )?;
        canonical.stream = wire_request.stream_requested;
        Ok(canonical)
    }

    fn decode_wire_owned(
        wire_request: Self::WireRequest,
        request_id: uuid::Uuid,
    ) -> Result<CanonicalRequest, CanonicalError> {
        if wire_request.requested_model.is_empty() {
            return Err(CanonicalError::Internal(
                "Gemini requested model context is missing".to_string(),
            ));
        }
        let mut canonical = decode_gemini_request_owned(
            wire_request.request,
            wire_request.requested_model,
            request_id,
        )?;
        canonical.stream = wire_request.stream_requested;
        Ok(canonical)
    }

    async fn run_no_tools_non_stream(ctx: NoToolsCtx<'_>) -> Result<Response, CanonicalError> {
        run_gemini_no_tools_non_stream(
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
        run_gemini_fc_non_stream(
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
        run_gemini_auto_fallback(
            GeminiAutoFallbackCtx {
                state: input.state,
                wire_request: &input.wire_request.request,
                route: input.route,
                model: input.requested_model,
                request_seq: input.request_seq,
                request_id: input.request_id,
                is_stream: input.wire_request.stream_requested,
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
        _same_model: bool,
    ) -> Result<Response, CanonicalError> {
        handle_non_streaming(io_ctx, upstream_canonical, fc_active, saved_tools).await
    }

    async fn handle_non_streaming_preencoded<'a>(
        io_ctx: UpstreamIoRequest<'a>,
        upstream_body: bytes::Bytes,
        fc_active: bool,
        saved_tools: &'a [CanonicalToolSpec],
        _same_model: bool,
    ) -> Result<Response, CanonicalError> {
        handle_non_streaming_preencoded(io_ctx, upstream_body, fc_active, saved_tools).await
    }
}

pub(super) struct GeminiAction<'a> {
    pub(super) model: &'a str,
    pub(super) is_stream: bool,
}

#[must_use]
pub(super) fn parse_model_action(model_action: &str) -> GeminiAction<'_> {
    match model_action.rsplit_once(':') {
        Some((model_name, "streamGenerateContent")) => GeminiAction {
            model: model_name,
            is_stream: true,
        },
        Some((model_name, _)) => GeminiAction {
            model: model_name,
            is_stream: false,
        },
        None => GeminiAction {
            model: model_action,
            is_stream: false,
        },
    }
}
