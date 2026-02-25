//! Shared API pipeline primitives reused across ingress handlers.
//!
//! This module re-exports shared helper surface from `api::common`.
//! Behavioral logic remains unchanged.

use axum::http::HeaderMap;
use smallvec::SmallVec;

use crate::error::CanonicalError;
use crate::protocol::canonical::{IngressApi, ProviderKind};
use crate::routing::session::SessionClass;
use crate::routing::RouteTarget;
use crate::state::{AppState, FcDecision};

pub(crate) use crate::api::common::{
    encode_for_provider, find_top_level_field_value_range, handle_non_streaming_common,
    handle_non_streaming_preencoded_common, handle_streaming_request, parse_common_request_probe,
    prepare_upstream_io_request, raw_tools_field_has_items, CommonProbeRanges, CommonRequestProbe,
    PreparedUpstreamIoRequest, UpstreamIoRequest,
};

pub(crate) struct FlowBootstrap<'a> {
    pub(crate) route_candidates: SmallVec<[RouteTarget<'a>; 4]>,
    pub(crate) route: RouteTarget<'a>,
    pub(crate) provider: ProviderKind,
    pub(crate) fc_decision: FcDecision,
}

/// Resolve route candidates and FC policy for an ingress request.
///
/// # Errors
///
/// Returns [`CanonicalError::InvalidRequest`] when no route can be resolved.
pub(crate) fn bootstrap_flow<'a>(
    state: &'a AppState,
    ingress: IngressApi,
    headers: &HeaderMap,
    model: &'a str,
    prompt_prefix: &[u8],
    session_class: SessionClass,
    has_tools: bool,
) -> Result<FlowBootstrap<'a>, CanonicalError> {
    let route_hash = if state.model_router.requires_request_hash_for_ordering(model) {
        state.route_sticky_hash(ingress, headers, model, prompt_prefix)
    } else {
        0
    };
    let route_candidates = state.resolve_routes_with_policy(model, route_hash, session_class)?;
    let route = *route_candidates
        .first()
        .ok_or_else(|| CanonicalError::InvalidRequest(format!("No upstream for '{model}'")))?;
    let provider = state.prepared_upstreams[route.upstream_index].provider_kind();
    let fc_decision = if has_tools {
        state.fc_decision(&route, true)
    } else {
        FcDecision {
            fc_active: false,
            auto_fallback_allowed: false,
        }
    };

    Ok(FlowBootstrap {
        route_candidates,
        route,
        provider,
        fc_decision,
    })
}
