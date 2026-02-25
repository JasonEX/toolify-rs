use std::sync::Arc;

use axum::http::HeaderMap;
use axum::response::Response;

use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::canonical::{IngressApi, ProviderKind};
use crate::routing::RouteTarget;
use crate::state::AppState;
use crate::transport::{
    build_provider_headers_prepared, build_upstream_url_prepared, PreparedUpstream,
};

use crate::api::common::{
    is_protocol_passthrough, passthrough_non_streaming_bytes, passthrough_non_streaming_uri_bytes,
    passthrough_non_streaming_url_bytes, passthrough_streaming_bytes,
    passthrough_streaming_uri_bytes, passthrough_streaming_url_bytes,
    rewrite_model_field_in_json_body_with_range,
};

pub(crate) type UriGetter = for<'a> fn(&'a PreparedUpstream) -> Option<&'a http::Uri>;
pub(crate) type UrlGetter = for<'a> fn(&'a PreparedUpstream) -> Option<&'a url::Url>;

#[derive(Clone, Copy)]
pub(crate) struct UriUrlEndpointConfig {
    pub(crate) ingress: IngressApi,
    pub(crate) request_label: &'static str,
    pub(crate) uri_getter: UriGetter,
    pub(crate) url_getter: UrlGetter,
    pub(crate) rewrite_model_field: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct ChannelBState<'a> {
    pub(crate) route: RouteTarget<'a>,
    pub(crate) provider: ProviderKind,
    pub(crate) fc_active: bool,
}

pub(crate) struct ChannelBPlan<'a> {
    pub(crate) model: &'a str,
    pub(crate) model_value_range: Option<&'a std::ops::Range<usize>>,
    pub(crate) route_candidates: &'a [RouteTarget<'a>],
    pub(crate) state: ChannelBState<'a>,
    pub(crate) auto_fallback_allowed: bool,
    pub(crate) stream_requested: bool,
}

pub(crate) enum ChannelBFastPathOutcome<'a> {
    Continue(ChannelBState<'a>),
    Return(Response),
    Error(CanonicalError),
}

enum NoAutoFallbackDecision {
    Return(Response),
    RetryNext(CanonicalError),
    Error(CanonicalError),
}

enum NativeDecision {
    Return(Response),
    RetryNext(CanonicalError),
    Error(CanonicalError),
    ContinueAfterInject,
}

struct PassthroughAttempt<'a> {
    stream_requested: bool,
    parsed_passthrough_uri: Option<&'a http::Uri>,
    parsed_passthrough_url: Option<&'a url::Url>,
    url: Option<std::borrow::Cow<'a, str>>,
    proxy_url: Option<&'a str>,
    upstream_headers: &'a HeaderMap,
    passthrough_body: bytes::Bytes,
}

pub(crate) async fn run_channel_b_fast_path_uri_url<'a>(
    state: &Arc<AppState>,
    body: &bytes::Bytes,
    request_seq: &mut Option<u64>,
    mut plan: ChannelBPlan<'a>,
    config: UriUrlEndpointConfig,
) -> ChannelBFastPathOutcome<'a> {
    if plan.state.fc_active || !is_protocol_passthrough(plan.state.provider, config.ingress) {
        return ChannelBFastPathOutcome::Continue(plan.state);
    }
    let mut last_passthrough_err: Option<CanonicalError> = None;
    for (route_idx, candidate_route) in plan.route_candidates.iter().copied().enumerate() {
        let candidate_prepared_upstream = &state.prepared_upstreams[candidate_route.upstream_index];
        let candidate_provider = candidate_prepared_upstream.provider_kind();
        if !is_protocol_passthrough(candidate_provider, config.ingress) {
            if last_passthrough_err.is_some() && !plan.stream_requested {
                plan.state.route = candidate_route;
                plan.state.provider = candidate_provider;
                last_passthrough_err = None;
                break;
            }
            continue;
        }
        let proxy_url = candidate_prepared_upstream.proxy_for(plan.stream_requested);
        let parsed_passthrough_uri = if state.transport.hyper_passthrough_enabled_for(proxy_url) {
            (config.uri_getter)(candidate_prepared_upstream)
        } else {
            None
        };
        let parsed_passthrough_url = if parsed_passthrough_uri.is_none() {
            (config.url_getter)(candidate_prepared_upstream)
        } else {
            None
        };
        let candidate_url = if parsed_passthrough_uri.is_none() && parsed_passthrough_url.is_none()
        {
            Some(build_upstream_url_prepared(
                candidate_prepared_upstream,
                candidate_route.actual_model,
                plan.stream_requested,
            ))
        } else {
            None
        };
        let upstream_headers = build_provider_headers_prepared(candidate_prepared_upstream);
        let passthrough_body =
            if !config.rewrite_model_field || candidate_route.actual_model == plan.model {
                body.clone()
            } else {
                match rewrite_model_field_in_json_body_with_range(
                    body,
                    candidate_route.actual_model,
                    config.request_label,
                    plan.model_value_range,
                ) {
                    Ok(rewritten) => rewritten,
                    Err(err) => return ChannelBFastPathOutcome::Error(err),
                }
            };
        if !plan.auto_fallback_allowed {
            let attempt = PassthroughAttempt {
                stream_requested: plan.stream_requested,
                parsed_passthrough_uri,
                parsed_passthrough_url,
                url: candidate_url,
                proxy_url,
                upstream_headers,
                passthrough_body,
            };
            match handle_no_auto_fallback_attempt(state, &plan, route_idx, candidate_route, attempt)
                .await
            {
                NoAutoFallbackDecision::Return(response) => {
                    return ChannelBFastPathOutcome::Return(response);
                }
                NoAutoFallbackDecision::RetryNext(err) => {
                    last_passthrough_err = Some(err);
                    continue;
                }
                NoAutoFallbackDecision::Error(err) => {
                    return ChannelBFastPathOutcome::Error(err);
                }
            }
        }
        let attempt = PassthroughAttempt {
            stream_requested: plan.stream_requested,
            parsed_passthrough_uri,
            parsed_passthrough_url,
            url: candidate_url,
            proxy_url,
            upstream_headers,
            passthrough_body,
        };
        let native_result = dispatch_attempt(state, attempt).await;
        match handle_native_attempt(
            state,
            request_seq,
            &mut plan,
            route_idx,
            candidate_route,
            candidate_provider,
            native_result,
        ) {
            NativeDecision::Return(response) => return ChannelBFastPathOutcome::Return(response),
            NativeDecision::RetryNext(err) => {
                last_passthrough_err = Some(err);
            }
            NativeDecision::Error(err) => return ChannelBFastPathOutcome::Error(err),
            NativeDecision::ContinueAfterInject => break,
        }
    }
    if !plan.state.fc_active {
        if let Some(err) = last_passthrough_err {
            return ChannelBFastPathOutcome::Error(err);
        }
    }
    ChannelBFastPathOutcome::Continue(plan.state)
}

async fn handle_no_auto_fallback_attempt(
    state: &Arc<AppState>,
    plan: &ChannelBPlan<'_>,
    route_idx: usize,
    candidate_route: RouteTarget<'_>,
    attempt: PassthroughAttempt<'_>,
) -> NoAutoFallbackDecision {
    let passthrough_result = dispatch_attempt(state, attempt).await;
    state.record_upstream_outcome(
        candidate_route.upstream_index,
        plan.model,
        &passthrough_result,
    );
    match passthrough_result {
        Ok(response) => NoAutoFallbackDecision::Return(response),
        Err(err) => {
            if !state.should_try_alternate_upstream(&err)
                || route_idx + 1 >= plan.route_candidates.len()
            {
                NoAutoFallbackDecision::Error(err)
            } else {
                NoAutoFallbackDecision::RetryNext(err)
            }
        }
    }
}

async fn dispatch_attempt(
    state: &Arc<AppState>,
    attempt: PassthroughAttempt<'_>,
) -> Result<Response, CanonicalError> {
    dispatch_passthrough(
        state,
        attempt.stream_requested,
        attempt.parsed_passthrough_uri,
        attempt.parsed_passthrough_url,
        attempt.url.as_deref(),
        attempt.proxy_url,
        attempt.upstream_headers,
        attempt.passthrough_body,
    )
    .await
}

fn handle_native_attempt<'a>(
    state: &Arc<AppState>,
    request_seq: &mut Option<u64>,
    plan: &mut ChannelBPlan<'a>,
    route_idx: usize,
    candidate_route: RouteTarget<'a>,
    candidate_provider: ProviderKind,
    native_result: Result<Response, CanonicalError>,
) -> NativeDecision {
    match native_result {
        Ok(response) => {
            state.record_upstream_success(candidate_route.upstream_index, plan.model);
            NativeDecision::Return(response)
        }
        Err(err) if fc::should_auto_fallback_to_inject(&err) => {
            state.record_upstream_failure(candidate_route.upstream_index, plan.model, &err);
            state.mark_auto_inject(&candidate_route);
            let request_seq_value = *request_seq.get_or_insert_with(|| state.next_request_seq());
            tracing::debug!(
                request_id = %request_seq_value,
                upstream = %state.upstream_name(candidate_route.upstream_index),
                "fc_mode=auto: native tool path unsupported, retrying with inject mode"
            );
            plan.state.route = candidate_route;
            plan.state.provider = candidate_provider;
            plan.state.fc_active = true;
            NativeDecision::ContinueAfterInject
        }
        Err(err) => {
            state.record_upstream_failure(candidate_route.upstream_index, plan.model, &err);
            if !state.should_try_alternate_upstream(&err)
                || route_idx + 1 >= plan.route_candidates.len()
            {
                NativeDecision::Error(err)
            } else {
                NativeDecision::RetryNext(err)
            }
        }
    }
}

async fn dispatch_passthrough(
    state: &Arc<AppState>,
    stream_requested: bool,
    parsed_passthrough_uri: Option<&http::Uri>,
    parsed_passthrough_url: Option<&url::Url>,
    url: Option<&str>,
    proxy_url: Option<&str>,
    upstream_headers: &HeaderMap,
    passthrough_body: bytes::Bytes,
) -> Result<Response, CanonicalError> {
    if stream_requested {
        if let Some(parsed_uri) = parsed_passthrough_uri {
            return passthrough_streaming_uri_bytes(
                state,
                parsed_uri,
                upstream_headers,
                passthrough_body,
            )
            .await;
        }
        if let Some(parsed_url) = parsed_passthrough_url {
            return passthrough_streaming_url_bytes(
                state,
                parsed_url,
                proxy_url,
                upstream_headers,
                passthrough_body,
            )
            .await;
        }
        let Some(url) = url else {
            return Err(CanonicalError::Internal(
                "missing passthrough URL for streaming request".to_string(),
            ));
        };
        passthrough_streaming_bytes(state, url, proxy_url, upstream_headers, passthrough_body).await
    } else if let Some(parsed_uri) = parsed_passthrough_uri {
        passthrough_non_streaming_uri_bytes(state, parsed_uri, upstream_headers, passthrough_body)
            .await
    } else if let Some(parsed_url) = parsed_passthrough_url {
        passthrough_non_streaming_url_bytes(
            state,
            parsed_url,
            proxy_url,
            upstream_headers,
            passthrough_body,
        )
        .await
    } else {
        let Some(url) = url else {
            return Err(CanonicalError::Internal(
                "missing passthrough URL for non-streaming request".to_string(),
            ));
        };
        passthrough_non_streaming_bytes(state, url, proxy_url, upstream_headers, passthrough_body)
            .await
    }
}
