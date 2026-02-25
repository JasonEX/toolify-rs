use axum::response::Response;

use crate::api::common::{
    is_protocol_passthrough, passthrough_non_streaming_bytes, passthrough_non_streaming_uri_bytes,
    passthrough_non_streaming_url_bytes, rewrite_model_field_in_json_body_with_range,
};
use crate::api::ingress::anthropic::io::handle_non_streaming as anthropic_handle_non_streaming;
use crate::api::ingress::gemini::io::handle_non_streaming as gemini_handle_non_streaming;
use crate::api::ingress::openai_chat::io::handle_non_streaming as openai_chat_handle_non_streaming;
use crate::api::ingress::openai_chat::parse::parse_openai_chat_request_wire;
use crate::api::ingress::openai_responses::io::handle_non_streaming as openai_responses_handle_non_streaming;
use crate::error::CanonicalError;
use crate::protocol::anthropic::decoder::decode_anthropic_request_owned;
use crate::protocol::anthropic::AnthropicRequest;
use crate::protocol::canonical::IngressApi;
use crate::protocol::gemini::decoder::decode_gemini_request_owned;
use crate::protocol::gemini::GeminiRequest;
use crate::protocol::openai_chat::decoder::decode_openai_chat_request_owned;
use crate::protocol::openai_responses::decoder::decode_responses_request_owned;
use crate::protocol::openai_responses::ResponsesRequest;
use crate::routing::RouteTarget;
use crate::state::AppState;

use super::{prepare_candidate_upstream_request, start_candidate_index};

const EXHAUSTED_MSG: &str = "No upstream candidate available for no-tools failover";

#[inline]
fn passthrough_body_for_model(
    body: &bytes::Bytes,
    target_model: &str,
    client_model: &str,
    request_name: &str,
    model_value_range: Option<&std::ops::Range<usize>>,
) -> Result<bytes::Bytes, CanonicalError> {
    if target_model == client_model {
        Ok(body.clone())
    } else {
        rewrite_model_field_in_json_body_with_range(
            body,
            target_model,
            request_name,
            model_value_range,
        )
    }
}

#[inline]
fn cached_passthrough_body_for_model<'a>(
    cache: &mut Option<(&'a str, bytes::Bytes)>,
    body: &bytes::Bytes,
    target_model: &'a str,
    client_model: &str,
    request_name: &str,
    model_value_range: Option<&std::ops::Range<usize>>,
) -> Result<bytes::Bytes, CanonicalError> {
    if let Some((cached_model, cached_body)) = cache.as_ref() {
        if *cached_model == target_model {
            return Ok(cached_body.clone());
        }
    }

    let rewritten = passthrough_body_for_model(
        body,
        target_model,
        client_model,
        request_name,
        model_value_range,
    )?;
    *cache = Some((target_model, rewritten.clone()));
    Ok(rewritten)
}

#[inline]
fn all_candidates_protocol_passthrough(
    state: &AppState,
    route_candidates: &[RouteTarget<'_>],
    ingress: IngressApi,
) -> bool {
    route_candidates.iter().all(|candidate_route| {
        let provider = state.prepared_upstreams[candidate_route.upstream_index].provider_kind();
        is_protocol_passthrough(provider, ingress)
    })
}

async fn run_passthrough_only_no_tools_failover<'a>(
    state: &AppState,
    body: &bytes::Bytes,
    model_value_range: Option<&std::ops::Range<usize>>,
    route_candidates: &[RouteTarget<'a>],
    route: RouteTarget<'a>,
    model_for_policy: &str,
    client_model: &str,
    request_name: &str,
) -> Result<Response, CanonicalError> {
    let start_idx = start_candidate_index(route_candidates, route);
    let mut last_err: Option<CanonicalError> = None;
    let mut passthrough_body_cache: Option<(&str, bytes::Bytes)> = None;
    for idx in start_idx..route_candidates.len() {
        let candidate_route = route_candidates[idx];
        let candidate_upstream = prepare_candidate_upstream_request(state, candidate_route, false);
        let candidate_passthrough_body = cached_passthrough_body_for_model(
            &mut passthrough_body_cache,
            body,
            candidate_route.actual_model,
            client_model,
            request_name,
            model_value_range,
        )?;
        let io_ctx = candidate_upstream.io_ctx(client_model);
        let attempt_result = passthrough_non_streaming_io(io_ctx, candidate_passthrough_body).await;

        state.record_upstream_outcome(
            candidate_route.upstream_index,
            model_for_policy,
            &attempt_result,
        );
        match attempt_result {
            Ok(response) => return Ok(response),
            Err(err) => {
                if idx + 1 < route_candidates.len() && state.should_try_alternate_upstream(&err) {
                    last_err = Some(err);
                    continue;
                }
                return Err(err);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| CanonicalError::Internal(EXHAUSTED_MSG.to_string())))
}

pub(crate) async fn run_openai_chat_no_tools_non_stream<'a>(
    state: &AppState,
    body: &bytes::Bytes,
    model_value_range: Option<&std::ops::Range<usize>>,
    route_candidates: &[RouteTarget<'a>],
    route: RouteTarget<'a>,
    request_id: uuid::Uuid,
    client_model: &str,
) -> Result<Response, CanonicalError> {
    if route_candidates.len() > 1
        && all_candidates_protocol_passthrough(state, route_candidates, IngressApi::OpenAiChat)
    {
        return run_passthrough_only_no_tools_failover(
            state,
            body,
            model_value_range,
            route_candidates,
            route,
            client_model,
            client_model,
            "OpenAI Chat request",
        )
        .await;
    }

    let route_provider = state.prepared_upstreams[route.upstream_index].provider_kind();
    if is_protocol_passthrough(route_provider, IngressApi::OpenAiChat) {
        let passthrough_body = passthrough_body_for_model(
            body,
            route.actual_model,
            client_model,
            "OpenAI Chat request",
            model_value_range,
        )?;
        let candidate_upstream = prepare_candidate_upstream_request(state, route, false);
        let io_ctx = candidate_upstream.io_ctx(client_model);
        return passthrough_non_streaming_io(io_ctx, passthrough_body).await;
    }

    let mut cached_upstream_canonical = None;
    if route_candidates.len() == 1 {
        if cached_upstream_canonical.is_none() {
            let base_request = parse_openai_chat_request_wire(body)?;
            cached_upstream_canonical =
                Some(decode_openai_chat_request_owned(base_request, request_id)?);
        }
        let upstream_canonical = cached_upstream_canonical
            .as_mut()
            .expect("cached canonical must exist");
        upstream_canonical.model.clear();
        upstream_canonical.model.push_str(route.actual_model);
        let candidate_upstream = prepare_candidate_upstream_request(state, route, false);
        let io_ctx = candidate_upstream.io_ctx(client_model);
        return openai_chat_handle_non_streaming(
            io_ctx,
            upstream_canonical,
            false,
            &[],
            route.actual_model == client_model,
        )
        .await;
    }

    let start_idx = start_candidate_index(route_candidates, route);
    let mut last_err: Option<CanonicalError> = None;
    let mut passthrough_body_cache: Option<(&str, bytes::Bytes)> = None;
    for idx in start_idx..route_candidates.len() {
        let candidate_route = route_candidates[idx];
        let candidate_upstream = prepare_candidate_upstream_request(state, candidate_route, false);
        let candidate_provider =
            state.prepared_upstreams[candidate_route.upstream_index].provider_kind();
        let attempt_result = if is_protocol_passthrough(candidate_provider, IngressApi::OpenAiChat)
        {
            let candidate_passthrough_body = cached_passthrough_body_for_model(
                &mut passthrough_body_cache,
                body,
                candidate_route.actual_model,
                client_model,
                "OpenAI Chat request",
                model_value_range,
            )?;
            let io_ctx = candidate_upstream.io_ctx(client_model);
            passthrough_non_streaming_io(io_ctx, candidate_passthrough_body).await
        } else {
            if cached_upstream_canonical.is_none() {
                let base_request = parse_openai_chat_request_wire(body)?;
                cached_upstream_canonical =
                    Some(decode_openai_chat_request_owned(base_request, request_id)?);
            }
            let upstream_canonical = cached_upstream_canonical
                .as_mut()
                .expect("cached canonical must exist");
            upstream_canonical.model.clear();
            upstream_canonical
                .model
                .push_str(candidate_route.actual_model);

            let model_matches = candidate_route.actual_model == client_model;
            let io_ctx = candidate_upstream.io_ctx(client_model);
            openai_chat_handle_non_streaming(io_ctx, upstream_canonical, false, &[], model_matches)
                .await
        };

        state.record_upstream_outcome(
            candidate_route.upstream_index,
            client_model,
            &attempt_result,
        );
        match attempt_result {
            Ok(response) => return Ok(response),
            Err(err) => {
                if idx + 1 < route_candidates.len() && state.should_try_alternate_upstream(&err) {
                    last_err = Some(err);
                    continue;
                }
                return Err(err);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| CanonicalError::Internal(EXHAUSTED_MSG.to_string())))
}

pub(crate) async fn run_openai_responses_no_tools_non_stream<'a>(
    state: &AppState,
    body: &bytes::Bytes,
    model_value_range: Option<&std::ops::Range<usize>>,
    route_candidates: &[RouteTarget<'a>],
    route: RouteTarget<'a>,
    request_id: uuid::Uuid,
    client_model: &str,
) -> Result<Response, CanonicalError> {
    if route_candidates.len() > 1
        && all_candidates_protocol_passthrough(state, route_candidates, IngressApi::OpenAiResponses)
    {
        return run_passthrough_only_no_tools_failover(
            state,
            body,
            model_value_range,
            route_candidates,
            route,
            client_model,
            client_model,
            "OpenAI Responses request",
        )
        .await;
    }

    let route_provider = state.prepared_upstreams[route.upstream_index].provider_kind();
    if is_protocol_passthrough(route_provider, IngressApi::OpenAiResponses) {
        let passthrough_body = passthrough_body_for_model(
            body,
            route.actual_model,
            client_model,
            "OpenAI Responses request",
            model_value_range,
        )?;
        let candidate_upstream = prepare_candidate_upstream_request(state, route, false);
        let io_ctx = candidate_upstream.io_ctx(client_model);
        return passthrough_non_streaming_io(io_ctx, passthrough_body).await;
    }

    let mut cached_upstream_canonical = None;
    if route_candidates.len() == 1 {
        if cached_upstream_canonical.is_none() {
            let request: ResponsesRequest = serde_json::from_slice(body).map_err(|e| {
                CanonicalError::InvalidRequest(format!(
                    "Invalid OpenAI Responses request body: {e}"
                ))
            })?;
            cached_upstream_canonical = Some(decode_responses_request_owned(request, request_id)?);
        }
        let upstream_canonical = cached_upstream_canonical
            .as_mut()
            .expect("cached canonical must exist");
        upstream_canonical.model.clear();
        upstream_canonical.model.push_str(route.actual_model);
        let candidate_upstream = prepare_candidate_upstream_request(state, route, false);
        let io_ctx = candidate_upstream.io_ctx(client_model);
        return openai_responses_handle_non_streaming(
            io_ctx,
            upstream_canonical,
            false,
            &[],
            route.actual_model == client_model,
        )
        .await;
    }

    let start_idx = start_candidate_index(route_candidates, route);
    let mut last_err: Option<CanonicalError> = None;
    let mut passthrough_body_cache: Option<(&str, bytes::Bytes)> = None;
    for idx in start_idx..route_candidates.len() {
        let candidate_route = route_candidates[idx];
        let candidate_upstream = prepare_candidate_upstream_request(state, candidate_route, false);
        let candidate_provider =
            state.prepared_upstreams[candidate_route.upstream_index].provider_kind();
        let attempt_result =
            if is_protocol_passthrough(candidate_provider, IngressApi::OpenAiResponses) {
                let candidate_passthrough_body = cached_passthrough_body_for_model(
                    &mut passthrough_body_cache,
                    body,
                    candidate_route.actual_model,
                    client_model,
                    "OpenAI Responses request",
                    model_value_range,
                )?;
                let io_ctx = candidate_upstream.io_ctx(client_model);
                passthrough_non_streaming_io(io_ctx, candidate_passthrough_body).await
            } else {
                if cached_upstream_canonical.is_none() {
                    let request: ResponsesRequest = serde_json::from_slice(body).map_err(|e| {
                        CanonicalError::InvalidRequest(format!(
                            "Invalid OpenAI Responses request body: {e}"
                        ))
                    })?;
                    cached_upstream_canonical =
                        Some(decode_responses_request_owned(request, request_id)?);
                }
                let upstream_canonical = cached_upstream_canonical
                    .as_mut()
                    .expect("cached canonical must exist");
                upstream_canonical.model.clear();
                upstream_canonical
                    .model
                    .push_str(candidate_route.actual_model);

                let model_matches = candidate_route.actual_model == client_model;
                let io_ctx = candidate_upstream.io_ctx(client_model);
                openai_responses_handle_non_streaming(
                    io_ctx,
                    upstream_canonical,
                    false,
                    &[],
                    model_matches,
                )
                .await
            };

        state.record_upstream_outcome(
            candidate_route.upstream_index,
            client_model,
            &attempt_result,
        );
        match attempt_result {
            Ok(response) => return Ok(response),
            Err(err) => {
                if idx + 1 < route_candidates.len() && state.should_try_alternate_upstream(&err) {
                    last_err = Some(err);
                    continue;
                }
                return Err(err);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| CanonicalError::Internal(EXHAUSTED_MSG.to_string())))
}

async fn passthrough_non_streaming_io(
    io_ctx: crate::api::engine::pipeline::UpstreamIoRequest<'_>,
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

pub(crate) async fn run_anthropic_no_tools_non_stream<'a>(
    state: &AppState,
    body: &bytes::Bytes,
    model_value_range: Option<&std::ops::Range<usize>>,
    route_candidates: &[RouteTarget<'a>],
    route: RouteTarget<'a>,
    request_id: uuid::Uuid,
    client_model: &str,
) -> Result<Response, CanonicalError> {
    if route_candidates.len() > 1
        && all_candidates_protocol_passthrough(state, route_candidates, IngressApi::Anthropic)
    {
        return run_passthrough_only_no_tools_failover(
            state,
            body,
            model_value_range,
            route_candidates,
            route,
            client_model,
            client_model,
            "Anthropic request",
        )
        .await;
    }

    let route_provider = state.prepared_upstreams[route.upstream_index].provider_kind();
    if is_protocol_passthrough(route_provider, IngressApi::Anthropic) {
        let passthrough_body = passthrough_body_for_model(
            body,
            route.actual_model,
            client_model,
            "Anthropic request",
            model_value_range,
        )?;
        let candidate_upstream = prepare_candidate_upstream_request(state, route, false);
        let io_ctx = candidate_upstream.io_ctx(client_model);
        return passthrough_non_streaming_io(io_ctx, passthrough_body).await;
    }

    let mut cached_upstream_canonical = None;
    if route_candidates.len() == 1 {
        if cached_upstream_canonical.is_none() {
            let request: AnthropicRequest = serde_json::from_slice(body).map_err(|e| {
                CanonicalError::InvalidRequest(format!("Invalid Anthropic request body: {e}"))
            })?;
            cached_upstream_canonical = Some(decode_anthropic_request_owned(request, request_id)?);
        }
        let upstream_canonical = cached_upstream_canonical
            .as_mut()
            .expect("cached canonical must exist");
        upstream_canonical.model.clear();
        upstream_canonical.model.push_str(route.actual_model);
        let candidate_upstream = prepare_candidate_upstream_request(state, route, false);
        let io_ctx = candidate_upstream.io_ctx(client_model);
        return anthropic_handle_non_streaming(io_ctx, upstream_canonical, false, &[]).await;
    }

    let start_idx = start_candidate_index(route_candidates, route);
    let mut last_err: Option<CanonicalError> = None;
    let mut passthrough_body_cache: Option<(&str, bytes::Bytes)> = None;
    for idx in start_idx..route_candidates.len() {
        let candidate_route = route_candidates[idx];
        let candidate_upstream = prepare_candidate_upstream_request(state, candidate_route, false);
        let candidate_provider =
            state.prepared_upstreams[candidate_route.upstream_index].provider_kind();
        let attempt_result = if is_protocol_passthrough(candidate_provider, IngressApi::Anthropic) {
            let candidate_passthrough_body = cached_passthrough_body_for_model(
                &mut passthrough_body_cache,
                body,
                candidate_route.actual_model,
                client_model,
                "Anthropic request",
                model_value_range,
            )?;
            let io_ctx = candidate_upstream.io_ctx(client_model);
            passthrough_non_streaming_io(io_ctx, candidate_passthrough_body).await
        } else {
            if cached_upstream_canonical.is_none() {
                let request: AnthropicRequest = serde_json::from_slice(body).map_err(|e| {
                    CanonicalError::InvalidRequest(format!("Invalid Anthropic request body: {e}"))
                })?;
                cached_upstream_canonical =
                    Some(decode_anthropic_request_owned(request, request_id)?);
            }
            let upstream_canonical = cached_upstream_canonical
                .as_mut()
                .expect("cached canonical must exist");
            upstream_canonical.model.clear();
            upstream_canonical
                .model
                .push_str(candidate_route.actual_model);

            let io_ctx = candidate_upstream.io_ctx(client_model);
            anthropic_handle_non_streaming(io_ctx, upstream_canonical, false, &[]).await
        };

        state.record_upstream_outcome(
            candidate_route.upstream_index,
            client_model,
            &attempt_result,
        );
        match attempt_result {
            Ok(response) => return Ok(response),
            Err(err) => {
                if idx + 1 < route_candidates.len() && state.should_try_alternate_upstream(&err) {
                    last_err = Some(err);
                    continue;
                }
                return Err(err);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| CanonicalError::Internal(EXHAUSTED_MSG.to_string())))
}

pub(crate) async fn run_gemini_no_tools_non_stream<'a>(
    state: &AppState,
    body: &bytes::Bytes,
    model_value_range: Option<&std::ops::Range<usize>>,
    route_candidates: &[RouteTarget<'a>],
    route: RouteTarget<'a>,
    request_id: uuid::Uuid,
    model: &str,
) -> Result<Response, CanonicalError> {
    if route_candidates.len() > 1
        && all_candidates_protocol_passthrough(state, route_candidates, IngressApi::Gemini)
    {
        return run_passthrough_only_no_tools_failover(
            state,
            body,
            model_value_range,
            route_candidates,
            route,
            model,
            model,
            "Gemini request",
        )
        .await;
    }

    let route_provider = state.prepared_upstreams[route.upstream_index].provider_kind();
    if is_protocol_passthrough(route_provider, IngressApi::Gemini) {
        let passthrough_body = passthrough_body_for_model(
            body,
            route.actual_model,
            model,
            "Gemini request",
            model_value_range,
        )?;
        let candidate_upstream = prepare_candidate_upstream_request(state, route, false);
        let io_ctx = candidate_upstream.io_ctx(model);
        return passthrough_non_streaming_io(io_ctx, passthrough_body).await;
    }

    let mut cached_upstream_canonical = None;
    if route_candidates.len() == 1 {
        if cached_upstream_canonical.is_none() {
            let request: GeminiRequest = serde_json::from_slice(body).map_err(|e| {
                CanonicalError::InvalidRequest(format!("Invalid Gemini request body: {e}"))
            })?;
            let mut decoded = decode_gemini_request_owned(request, model.to_owned(), request_id)?;
            decoded.stream = false;
            cached_upstream_canonical = Some(decoded);
        }
        let upstream_canonical = cached_upstream_canonical
            .as_mut()
            .expect("cached canonical must exist");
        upstream_canonical.model.clear();
        upstream_canonical.model.push_str(route.actual_model);
        let candidate_upstream = prepare_candidate_upstream_request(state, route, false);
        let io_ctx = candidate_upstream.io_ctx(model);
        return gemini_handle_non_streaming(io_ctx, upstream_canonical, false, &[]).await;
    }

    let start_idx = start_candidate_index(route_candidates, route);
    let mut last_err: Option<CanonicalError> = None;
    let mut passthrough_body_cache: Option<(&str, bytes::Bytes)> = None;
    for idx in start_idx..route_candidates.len() {
        let candidate_route = route_candidates[idx];
        let candidate_upstream = prepare_candidate_upstream_request(state, candidate_route, false);
        let candidate_provider =
            state.prepared_upstreams[candidate_route.upstream_index].provider_kind();
        let attempt_result = if is_protocol_passthrough(candidate_provider, IngressApi::Gemini) {
            let candidate_passthrough_body = cached_passthrough_body_for_model(
                &mut passthrough_body_cache,
                body,
                candidate_route.actual_model,
                model,
                "Gemini request",
                model_value_range,
            )?;
            let io_ctx = candidate_upstream.io_ctx(model);
            passthrough_non_streaming_io(io_ctx, candidate_passthrough_body).await
        } else {
            if cached_upstream_canonical.is_none() {
                let request: GeminiRequest = serde_json::from_slice(body).map_err(|e| {
                    CanonicalError::InvalidRequest(format!("Invalid Gemini request body: {e}"))
                })?;
                let mut decoded =
                    decode_gemini_request_owned(request, model.to_owned(), request_id)?;
                decoded.stream = false;
                cached_upstream_canonical = Some(decoded);
            }
            let upstream_canonical = cached_upstream_canonical
                .as_mut()
                .expect("cached canonical must exist");
            upstream_canonical.model.clear();
            upstream_canonical
                .model
                .push_str(candidate_route.actual_model);

            let io_ctx = candidate_upstream.io_ctx(model);
            gemini_handle_non_streaming(io_ctx, upstream_canonical, false, &[]).await
        };

        state.record_upstream_outcome(candidate_route.upstream_index, model, &attempt_result);
        match attempt_result {
            Ok(response) => return Ok(response),
            Err(err) => {
                if idx + 1 < route_candidates.len() && state.should_try_alternate_upstream(&err) {
                    last_err = Some(err);
                    continue;
                }
                return Err(err);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| CanonicalError::Internal(EXHAUSTED_MSG.to_string())))
}
