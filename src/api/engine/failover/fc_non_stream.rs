use axum::response::Response;

use crate::api::ingress::anthropic::io::handle_non_streaming as anthropic_handle_non_streaming;
use crate::api::ingress::gemini::io::handle_non_streaming as gemini_handle_non_streaming;
use crate::api::ingress::openai_chat::io::handle_non_streaming as openai_chat_handle_non_streaming;
use crate::api::ingress::openai_responses::io::handle_non_streaming as openai_responses_handle_non_streaming;
use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalRequest, CanonicalToolSpec};
use crate::routing::RouteTarget;
use crate::state::AppState;

use super::{prepare_candidate_upstream_request, start_candidate_index};

const EXHAUSTED_MSG: &str = "No upstream candidate available for fc non-stream failover";

pub(crate) async fn run_openai_chat_fc_non_stream<'a>(
    state: &AppState,
    route_candidates: &[RouteTarget<'a>],
    route: RouteTarget<'a>,
    upstream_canonical: &CanonicalRequest,
    saved_tools: &[CanonicalToolSpec],
    client_model: &str,
) -> Result<Response, CanonicalError> {
    if route_candidates.len() == 1 {
        let candidate_upstream = prepare_candidate_upstream_request(state, route, false);
        let io_ctx = candidate_upstream.io_ctx(client_model);
        return openai_chat_handle_non_streaming(
            io_ctx,
            upstream_canonical,
            true,
            saved_tools,
            route.actual_model == client_model,
        )
        .await;
    }

    let start_idx = start_candidate_index(route_candidates, route);
    let mut last_err: Option<CanonicalError> = None;
    let mut candidate_canonical = upstream_canonical.clone();
    for idx in start_idx..route_candidates.len() {
        let candidate_route = route_candidates[idx];
        candidate_canonical.model.clear();
        candidate_canonical
            .model
            .push_str(candidate_route.actual_model);
        let model_matches = candidate_route.actual_model == client_model;
        let candidate_upstream = prepare_candidate_upstream_request(state, candidate_route, false);
        let io_ctx = candidate_upstream.io_ctx(client_model);
        let attempt_result = openai_chat_handle_non_streaming(
            io_ctx,
            &candidate_canonical,
            true,
            saved_tools,
            model_matches,
        )
        .await;

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

pub(crate) async fn run_openai_responses_fc_non_stream<'a>(
    state: &AppState,
    route_candidates: &[RouteTarget<'a>],
    route: RouteTarget<'a>,
    upstream_canonical: &CanonicalRequest,
    saved_tools: &[CanonicalToolSpec],
    client_model: &str,
) -> Result<Response, CanonicalError> {
    if route_candidates.len() == 1 {
        let candidate_upstream = prepare_candidate_upstream_request(state, route, false);
        let io_ctx = candidate_upstream.io_ctx(client_model);
        return openai_responses_handle_non_streaming(
            io_ctx,
            upstream_canonical,
            true,
            saved_tools,
            route.actual_model == client_model,
        )
        .await;
    }

    let start_idx = start_candidate_index(route_candidates, route);
    let mut last_err: Option<CanonicalError> = None;
    let mut candidate_canonical = upstream_canonical.clone();
    for idx in start_idx..route_candidates.len() {
        let candidate_route = route_candidates[idx];
        candidate_canonical.model.clear();
        candidate_canonical
            .model
            .push_str(candidate_route.actual_model);
        let model_matches = candidate_route.actual_model == client_model;
        let candidate_upstream = prepare_candidate_upstream_request(state, candidate_route, false);
        let io_ctx = candidate_upstream.io_ctx(client_model);
        let attempt_result = openai_responses_handle_non_streaming(
            io_ctx,
            &candidate_canonical,
            true,
            saved_tools,
            model_matches,
        )
        .await;

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

pub(crate) async fn run_anthropic_fc_non_stream<'a>(
    state: &AppState,
    route_candidates: &[RouteTarget<'a>],
    route: RouteTarget<'a>,
    upstream_canonical: &CanonicalRequest,
    saved_tools: &[CanonicalToolSpec],
    client_model: &str,
) -> Result<Response, CanonicalError> {
    if route_candidates.len() == 1 {
        let candidate_upstream = prepare_candidate_upstream_request(state, route, false);
        let io_ctx = candidate_upstream.io_ctx(client_model);
        return anthropic_handle_non_streaming(io_ctx, upstream_canonical, true, saved_tools).await;
    }

    let start_idx = start_candidate_index(route_candidates, route);
    let mut last_err: Option<CanonicalError> = None;
    let mut candidate_canonical = upstream_canonical.clone();
    for idx in start_idx..route_candidates.len() {
        let candidate_route = route_candidates[idx];
        candidate_canonical.model.clear();
        candidate_canonical
            .model
            .push_str(candidate_route.actual_model);
        let candidate_upstream = prepare_candidate_upstream_request(state, candidate_route, false);
        let io_ctx = candidate_upstream.io_ctx(client_model);
        let attempt_result =
            anthropic_handle_non_streaming(io_ctx, &candidate_canonical, true, saved_tools).await;

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

pub(crate) async fn run_gemini_fc_non_stream<'a>(
    state: &AppState,
    route_candidates: &[RouteTarget<'a>],
    route: RouteTarget<'a>,
    upstream_canonical: &CanonicalRequest,
    saved_tools: &[CanonicalToolSpec],
    model: &str,
) -> Result<Response, CanonicalError> {
    if route_candidates.len() == 1 {
        let candidate_upstream = prepare_candidate_upstream_request(state, route, false);
        let io_ctx = candidate_upstream.io_ctx(model);
        return gemini_handle_non_streaming(io_ctx, upstream_canonical, true, saved_tools).await;
    }

    let start_idx = start_candidate_index(route_candidates, route);
    let mut last_err: Option<CanonicalError> = None;
    let mut candidate_canonical = upstream_canonical.clone();
    for idx in start_idx..route_candidates.len() {
        let candidate_route = route_candidates[idx];
        candidate_canonical.model.clear();
        candidate_canonical
            .model
            .push_str(candidate_route.actual_model);
        let candidate_upstream = prepare_candidate_upstream_request(state, candidate_route, false);
        let io_ctx = candidate_upstream.io_ctx(model);
        let attempt_result =
            gemini_handle_non_streaming(io_ctx, &candidate_canonical, true, saved_tools).await;

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
