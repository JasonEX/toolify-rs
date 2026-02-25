use std::sync::Arc;

use axum::http::HeaderMap;
use axum::response::Response;
use smallvec::{smallvec, SmallVec};

use crate::api::common::{
    is_protocol_passthrough, passthrough_non_streaming_bytes, passthrough_non_streaming_uri_bytes,
    passthrough_non_streaming_url_bytes, passthrough_streaming_bytes,
    passthrough_streaming_uri_bytes, passthrough_streaming_url_bytes,
    rewrite_model_field_in_json_body_with_range,
};
use crate::api::engine::channel_b::core::{ChannelBFastPathOutcome, ChannelBPlan, ChannelBState};
use crate::api::engine::fallback_common::run_preencoded_retry;
use crate::api::engine::pipeline::{
    bootstrap_flow, prepare_upstream_io_request, CommonProbeRanges, UpstreamIoRequest,
};
use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::canonical::CanonicalToolSpec;
use crate::routing::session;
use crate::routing::RouteTarget;
use crate::state::AppState;

use super::bootstrap::probe_messages_range;
use super::non_stream::{run_non_stream_with_fallback, NonStreamFallbackInput};
use super::raw_inject::try_raw_inject_fast_path;
use super::stream_failover::{run_stream_failover, StreamFailoverInput};
use super::types::{CompatFlowSpec, FcNonStreamCtx, NoToolsCtx};

struct SingleCandidateCtx<'a> {
    route: RouteTarget<'a>,
    provider: crate::protocol::canonical::ProviderKind,
    fc_decision: crate::state::FcDecision,
}

struct BootstrapResolved<'a> {
    route_candidates: SmallVec<[RouteTarget<'a>; 4]>,
    route: RouteTarget<'a>,
    provider: crate::protocol::canonical::ProviderKind,
    fc_decision: crate::state::FcDecision,
}

pub(crate) async fn run_compat_handler<S: CompatFlowSpec>(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: bytes::Bytes,
) -> Result<Response, CanonicalError> {
    run_compat_handler_with_route::<S>(state, headers, body, None, None).await
}

pub(crate) async fn run_compat_handler_with_route<S: CompatFlowSpec>(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: bytes::Bytes,
    requested_model_override: Option<&str>,
    stream_requested_override: Option<bool>,
) -> Result<Response, CanonicalError> {
    let mut request_seq: Option<u64> = None;

    state.authenticate(S::INGRESS, &headers)?;

    let probe = S::parse_probe(&body)?;
    let requested_model = requested_model_override.unwrap_or(probe.model.as_ref());
    let stream_requested = stream_requested_override.unwrap_or(probe.stream.unwrap_or(false));
    let single_candidate_ctx =
        resolve_single_candidate_ctx(state.as_ref(), requested_model, probe.has_tools)?;
    if let Some(response) = try_single_candidate_fast_path::<S>(
        &state,
        &body,
        requested_model,
        stream_requested,
        probe.has_tools,
        probe.ranges.as_ref(),
        single_candidate_ctx.as_ref(),
    )
    .await?
    {
        return Ok(response);
    }

    let resolved = if let Some(single_ctx) = single_candidate_ctx {
        BootstrapResolved {
            route_candidates: smallvec![single_ctx.route],
            route: single_ctx.route,
            provider: single_ctx.provider,
            fc_decision: single_ctx.fc_decision,
        }
    } else {
        bootstrap_multi_candidate_flow::<S>(
            state.as_ref(),
            &headers,
            &body,
            requested_model,
            probe.ranges.as_ref(),
            probe.has_tools,
        )?
    };
    let route_candidates = resolved.route_candidates;
    let mut route = resolved.route;
    let mut provider = resolved.provider;
    let mut fc_active = resolved.fc_decision.fc_active;
    let auto_fallback_allowed = resolved.fc_decision.auto_fallback_allowed;

    let channel_b_plan = ChannelBPlan {
        model: requested_model,
        model_value_range: probe
            .ranges
            .as_ref()
            .and_then(|ranges| ranges.model.as_ref()),
        route_candidates: &route_candidates,
        state: ChannelBState {
            route,
            provider,
            fc_active,
        },
        auto_fallback_allowed,
        stream_requested,
    };
    match S::run_channel_b_fast_path(&state, &body, &mut request_seq, channel_b_plan).await {
        ChannelBFastPathOutcome::Continue(next_state) => {
            route = next_state.route;
            provider = next_state.provider;
            fc_active = next_state.fc_active;
        }
        ChannelBFastPathOutcome::Return(response) => return Ok(response),
        ChannelBFastPathOutcome::Error(err) => return Err(err),
    }

    let prepared_upstream = &state.prepared_upstreams[route.upstream_index];
    let request_seq_opt = request_seq;
    let client_model = requested_model;

    if !probe.has_tools && !stream_requested {
        return S::run_no_tools_non_stream(NoToolsCtx {
            state: state.as_ref(),
            body: &body,
            model_value_range: probe
                .ranges
                .as_ref()
                .and_then(|ranges| ranges.model.as_ref()),
            route_candidates: &route_candidates,
            route,
            // No-tools non-stream hot path is predominantly raw passthrough,
            // where request_id is not observed. Avoid per-request sequence/uuid
            // generation on this path.
            request_id: uuid::Uuid::nil(),
            client_model,
        })
        .await;
    }

    let raw_fast_enabled = fc_active
        && !state.config.features.enable_fc_error_retry
        && route_candidates.len() == 1
        && S::supports_wire_inject_provider(provider);
    if let Some(response) = try_raw_inject_fast_path::<S>(
        state.as_ref(),
        &body,
        prepared_upstream,
        route,
        provider,
        client_model,
        request_seq_opt,
        probe.ranges.as_ref(),
        raw_fast_enabled,
    )
    .await?
    {
        return Ok(response);
    }
    let request_seq = request_seq_opt.unwrap_or_else(|| state.next_request_seq());

    let mut wire_request = S::parse_wire_request(&body)?;
    S::set_request_context(&mut wire_request, requested_model, stream_requested);

    if fc_active
        && !state.config.features.enable_fc_error_retry
        && route_candidates.len() == 1
        && S::supports_wire_inject_provider(provider)
    {
        let mut inject_wire = wire_request;
        S::set_wire_model(&mut inject_wire, route.actual_model);
        let inject_saved_tools = S::apply_wire_inject(&mut inject_wire, &state.config.features)?;
        let inject_fc_active = S::wire_inject_fc_active(&inject_saved_tools);
        let inject_stream = S::wire_stream_requested(&inject_wire);
        let inject_body = S::encode_wire(&inject_wire)?;
        let same_model = route.actual_model == client_model;
        let io_target = prepare_upstream_io_request(
            state.as_ref(),
            prepared_upstream,
            route.actual_model,
            inject_stream,
        );
        return run_preencoded_retry(
            &io_target,
            client_model,
            request_seq,
            inject_body,
            inject_stream,
            inject_fc_active,
            &inject_saved_tools,
            S::handle_streaming,
            |io_ctx, body, fc_active, saved_tools| async move {
                S::handle_non_streaming_preencoded(io_ctx, body, fc_active, saved_tools, same_model)
                    .await
            },
        )
        .await;
    }

    let needs_fallback_decode = auto_fallback_allowed && !fc_active;
    let mut wire_request = Some(wire_request);
    let request_id = state.request_uuid(request_seq);
    let mut upstream_canonical = if needs_fallback_decode {
        let request_ref = wire_request.as_ref().ok_or_else(|| {
            CanonicalError::Internal("wire request missing for fallback decode".to_string())
        })?;
        S::decode_wire_ref(request_ref, request_id)?
    } else {
        let request = wire_request.take().ok_or_else(|| {
            CanonicalError::Internal("wire request missing for owned decode".to_string())
        })?;
        S::decode_wire_owned(request, request_id)?
    };
    upstream_canonical.model.clear();
    upstream_canonical.model.push_str(route.actual_model);
    let saved_tools: Arc<[CanonicalToolSpec]> = if fc_active {
        fc::apply_fc_inject_take_tools(&mut upstream_canonical, &state.config.features)?
    } else {
        Arc::from([])
    };
    fc_active = S::canonical_fc_active(fc_active, &saved_tools);

    if fc_active && !upstream_canonical.stream {
        return S::run_fc_non_stream(FcNonStreamCtx {
            state: state.as_ref(),
            route_candidates: &route_candidates,
            route,
            upstream_canonical: &upstream_canonical,
            saved_tools: &saved_tools,
            client_model,
        })
        .await;
    }

    if upstream_canonical.stream {
        return run_stream_failover(StreamFailoverInput::<S> {
            state: &state,
            body: &body,
            requested_model,
            probe_ranges: probe.ranges.as_ref(),
            route_candidates: &route_candidates,
            route,
            upstream_canonical: &upstream_canonical,
            saved_tools: &saved_tools,
            fc_active,
            auto_fallback_allowed,
            request_seq,
            request_id,
            client_model,
            wire_request: wire_request.as_ref(),
        })
        .await;
    }

    run_non_stream_with_fallback(NonStreamFallbackInput::<S> {
        state: &state,
        body: &body,
        wire_request: wire_request.as_ref(),
        requested_model,
        probe_ranges: probe.ranges.as_ref(),
        route,
        prepared_upstream,
        provider,
        client_model,
        upstream_canonical: &upstream_canonical,
        fc_active,
        saved_tools: &saved_tools,
        auto_fallback_allowed,
        request_seq,
        request_id,
    })
    .await
}

async fn try_single_candidate_fast_path<S: CompatFlowSpec>(
    state: &Arc<AppState>,
    body: &bytes::Bytes,
    requested_model: &str,
    stream_requested: bool,
    has_tools: bool,
    probe_ranges: Option<&CommonProbeRanges>,
    single_candidate_ctx: Option<&SingleCandidateCtx<'_>>,
) -> Result<Option<Response>, CanonicalError> {
    let Some(single_ctx) = single_candidate_ctx else {
        return Ok(None);
    };
    let route = single_ctx.route;
    let provider = single_ctx.provider;
    let fc_decision = single_ctx.fc_decision;
    let prepared_upstream = &state.prepared_upstreams[route.upstream_index];

    if !has_tools && !fc_decision.fc_active && is_protocol_passthrough(provider, S::INGRESS) {
        let passthrough_body = if route.actual_model == requested_model {
            body.clone()
        } else {
            rewrite_model_field_in_json_body_with_range(
                body,
                route.actual_model,
                "request",
                probe_ranges.and_then(|ranges| ranges.model.as_ref()),
            )?
        };
        let io_target = prepare_upstream_io_request(
            state.as_ref(),
            prepared_upstream,
            route.actual_model,
            stream_requested,
        );
        let io_ctx = io_target.io_ctx(requested_model);
        if stream_requested {
            let response = passthrough_streaming_fast(io_ctx, passthrough_body).await?;
            return Ok(Some(response));
        }
        let response = passthrough_non_streaming_fast(io_ctx, passthrough_body).await?;
        return Ok(Some(response));
    }

    let raw_fast_enabled = fc_decision.fc_active
        && !state.config.features.enable_fc_error_retry
        && S::supports_wire_inject_provider(provider);
    if !raw_fast_enabled {
        return Ok(None);
    }

    try_raw_inject_fast_path::<S>(
        state.as_ref(),
        body,
        prepared_upstream,
        route,
        provider,
        requested_model,
        None,
        probe_ranges,
        true,
    )
    .await
}

async fn passthrough_non_streaming_fast(
    io_ctx: UpstreamIoRequest<'_>,
    body: bytes::Bytes,
) -> Result<Response, CanonicalError> {
    if io_ctx
        .state
        .transport
        .hyper_passthrough_enabled_for(io_ctx.proxy_url)
    {
        if let Some(parsed_uri) = io_ctx.parsed_hyper_uri {
            return passthrough_non_streaming_uri_bytes(
                io_ctx.state,
                parsed_uri,
                io_ctx.upstream_headers,
                body,
            )
            .await;
        }
        return passthrough_non_streaming_bytes(
            io_ctx.state,
            io_ctx.url,
            io_ctx.proxy_url,
            io_ctx.upstream_headers,
            body,
        )
        .await;
    }

    if let Some(parsed_url) = io_ctx.parsed_url {
        return passthrough_non_streaming_url_bytes(
            io_ctx.state,
            parsed_url,
            io_ctx.proxy_url,
            io_ctx.upstream_headers,
            body,
        )
        .await;
    }

    passthrough_non_streaming_bytes(
        io_ctx.state,
        io_ctx.url,
        io_ctx.proxy_url,
        io_ctx.upstream_headers,
        body,
    )
    .await
}

async fn passthrough_streaming_fast(
    io_ctx: UpstreamIoRequest<'_>,
    body: bytes::Bytes,
) -> Result<Response, CanonicalError> {
    if io_ctx
        .state
        .transport
        .hyper_passthrough_enabled_for(io_ctx.proxy_url)
    {
        if let Some(parsed_uri) = io_ctx.parsed_hyper_uri {
            return passthrough_streaming_uri_bytes(
                io_ctx.state,
                parsed_uri,
                io_ctx.upstream_headers,
                body,
            )
            .await;
        }
        return passthrough_streaming_bytes(
            io_ctx.state,
            io_ctx.url,
            io_ctx.proxy_url,
            io_ctx.upstream_headers,
            body,
        )
        .await;
    }

    if let Some(parsed_url) = io_ctx.parsed_url {
        return passthrough_streaming_url_bytes(
            io_ctx.state,
            parsed_url,
            io_ctx.proxy_url,
            io_ctx.upstream_headers,
            body,
        )
        .await;
    }

    passthrough_streaming_bytes(
        io_ctx.state,
        io_ctx.url,
        io_ctx.proxy_url,
        io_ctx.upstream_headers,
        body,
    )
    .await
}

fn resolve_single_candidate_ctx<'a>(
    state: &'a AppState,
    requested_model: &'a str,
    has_tools: bool,
) -> Result<Option<SingleCandidateCtx<'a>>, CanonicalError> {
    let Some(route) = state
        .model_router
        .resolve_if_single_candidate(requested_model)?
    else {
        return Ok(None);
    };
    let prepared_upstream = &state.prepared_upstreams[route.upstream_index];
    let provider = prepared_upstream.provider_kind();
    let fc_decision = if has_tools {
        state.fc_decision(&route, true)
    } else {
        crate::state::FcDecision {
            fc_active: false,
            auto_fallback_allowed: false,
        }
    };
    Ok(Some(SingleCandidateCtx {
        route,
        provider,
        fc_decision,
    }))
}

fn bootstrap_multi_candidate_flow<'a, S: CompatFlowSpec>(
    state: &'a AppState,
    headers: &HeaderMap,
    body: &'a bytes::Bytes,
    requested_model: &'a str,
    probe_ranges: Option<&'a CommonProbeRanges>,
    has_tools: bool,
) -> Result<BootstrapResolved<'a>, CanonicalError> {
    let hash_required = state
        .model_router
        .requires_request_hash_for_ordering(requested_model);
    let messages_range = if hash_required {
        probe_messages_range(probe_ranges)
    } else {
        None
    };
    let session_class = if hash_required {
        session::classify_session_class(body.as_ref(), messages_range.is_some())
    } else {
        session::SessionClass::Portable
    };
    let prompt_prefix = if hash_required {
        session::route_prompt_prefix_bytes(body.as_ref(), messages_range)
    } else {
        &[]
    };
    let flow = bootstrap_flow(
        state,
        S::INGRESS,
        headers,
        requested_model,
        prompt_prefix,
        session_class,
        has_tools,
    )?;
    Ok(BootstrapResolved {
        route_candidates: flow.route_candidates,
        route: flow.route,
        provider: flow.provider,
        fc_decision: flow.fc_decision,
    })
}
