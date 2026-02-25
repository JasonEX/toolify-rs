use std::sync::Arc;

use axum::response::Response;

use crate::config::FeaturesConfig;
use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalRequest, CanonicalToolSpec, IngressApi, ProviderKind};
use crate::routing::RouteTarget;
use crate::state::AppState;

use crate::api::common::{CommonProbeRanges, CommonRequestProbe};
use crate::api::engine::channel_b::core::{ChannelBFastPathOutcome, ChannelBPlan};
use crate::api::engine::pipeline::UpstreamIoRequest;

pub(crate) struct RawInjectPayload {
    pub(crate) body: bytes::Bytes,
    pub(crate) saved_tools: Arc<[CanonicalToolSpec]>,
    pub(crate) stream: bool,
    pub(crate) fc_active: bool,
}

pub(crate) struct NoToolsCtx<'a> {
    pub(crate) state: &'a AppState,
    pub(crate) body: &'a bytes::Bytes,
    pub(crate) model_value_range: Option<&'a std::ops::Range<usize>>,
    pub(crate) route_candidates: &'a [RouteTarget<'a>],
    pub(crate) route: RouteTarget<'a>,
    pub(crate) request_id: uuid::Uuid,
    pub(crate) client_model: &'a str,
}

pub(crate) struct FcNonStreamCtx<'a> {
    pub(crate) state: &'a AppState,
    pub(crate) route_candidates: &'a [RouteTarget<'a>],
    pub(crate) route: RouteTarget<'a>,
    pub(crate) upstream_canonical: &'a CanonicalRequest,
    pub(crate) saved_tools: &'a [CanonicalToolSpec],
    pub(crate) client_model: &'a str,
}

pub(crate) struct AutoFallbackInput<'a, W> {
    pub(crate) state: &'a AppState,
    pub(crate) body: &'a bytes::Bytes,
    pub(crate) wire_request: &'a W,
    pub(crate) route: RouteTarget<'a>,
    pub(crate) client_model: &'a str,
    pub(crate) requested_model: &'a str,
    pub(crate) request_seq: u64,
    pub(crate) request_id: uuid::Uuid,
    pub(crate) probe_ranges: Option<&'a CommonProbeRanges>,
    pub(crate) err: CanonicalError,
}

pub(crate) trait CompatFlowSpec {
    type WireRequest;

    const INGRESS: IngressApi;

    fn parse_probe(body: &bytes::Bytes) -> Result<CommonRequestProbe<'_>, CanonicalError>;
    fn parse_wire_request(body: &bytes::Bytes) -> Result<Self::WireRequest, CanonicalError>;
    fn set_request_context(
        _wire_request: &mut Self::WireRequest,
        _requested_model: &str,
        _stream_requested: bool,
    ) {
    }

    async fn run_channel_b_fast_path<'a>(
        state: &'a Arc<AppState>,
        body: &'a bytes::Bytes,
        request_seq: &mut Option<u64>,
        plan: ChannelBPlan<'a>,
    ) -> ChannelBFastPathOutcome<'a>;

    fn supports_wire_inject_provider(provider: ProviderKind) -> bool;
    fn set_wire_model(wire_request: &mut Self::WireRequest, actual_model: &str);
    fn apply_wire_inject(
        wire_request: &mut Self::WireRequest,
        features: &FeaturesConfig,
    ) -> Result<Vec<CanonicalToolSpec>, CanonicalError>;
    fn wire_stream_requested(wire_request: &Self::WireRequest) -> bool;
    fn encode_wire(wire_request: &Self::WireRequest) -> Result<bytes::Bytes, CanonicalError>;
    fn decode_wire_ref(
        wire_request: &Self::WireRequest,
        request_id: uuid::Uuid,
    ) -> Result<CanonicalRequest, CanonicalError>;
    fn decode_wire_owned(
        wire_request: Self::WireRequest,
        request_id: uuid::Uuid,
    ) -> Result<CanonicalRequest, CanonicalError>;

    async fn run_no_tools_non_stream(ctx: NoToolsCtx<'_>) -> Result<Response, CanonicalError>;
    async fn run_fc_non_stream(ctx: FcNonStreamCtx<'_>) -> Result<Response, CanonicalError>;
    async fn run_auto_fallback(
        input: AutoFallbackInput<'_, Self::WireRequest>,
    ) -> Result<Response, CanonicalError>;

    async fn handle_streaming<'a>(
        io_ctx: UpstreamIoRequest<'a>,
        upstream_body: bytes::Bytes,
        request_seq: u64,
        fc_active: bool,
        saved_tools: &'a [CanonicalToolSpec],
    ) -> Result<Response, CanonicalError>;
    async fn handle_non_streaming<'a>(
        io_ctx: UpstreamIoRequest<'a>,
        upstream_canonical: &'a CanonicalRequest,
        fc_active: bool,
        saved_tools: &'a [CanonicalToolSpec],
        same_model: bool,
    ) -> Result<Response, CanonicalError>;
    async fn handle_non_streaming_preencoded<'a>(
        io_ctx: UpstreamIoRequest<'a>,
        upstream_body: bytes::Bytes,
        fc_active: bool,
        saved_tools: &'a [CanonicalToolSpec],
        same_model: bool,
    ) -> Result<Response, CanonicalError>;

    fn try_raw_inject_fast_path(
        _body: &bytes::Bytes,
        _actual_model: &str,
        _features: &FeaturesConfig,
        _probe_ranges: Option<&CommonProbeRanges>,
    ) -> Result<Option<RawInjectPayload>, CanonicalError> {
        Ok(None)
    }

    fn wire_inject_fc_active(saved_tools: &[CanonicalToolSpec]) -> bool {
        !saved_tools.is_empty()
    }

    fn canonical_fc_active(fc_active: bool, saved_tools: &[CanonicalToolSpec]) -> bool {
        fc_active && !saved_tools.is_empty()
    }
}
