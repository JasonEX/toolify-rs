use crate::api::engine::pipeline::{prepare_upstream_io_request, PreparedUpstreamIoRequest};
use crate::routing::RouteTarget;
use crate::state::AppState;

mod fc_non_stream;
mod no_tools;

pub(crate) use fc_non_stream::{
    run_anthropic_fc_non_stream, run_gemini_fc_non_stream, run_openai_chat_fc_non_stream,
    run_openai_responses_fc_non_stream,
};
pub(crate) use no_tools::{
    run_anthropic_no_tools_non_stream, run_gemini_no_tools_non_stream,
    run_openai_chat_no_tools_non_stream, run_openai_responses_no_tools_non_stream,
};

pub(super) type CandidateUpstreamRequest<'a> = PreparedUpstreamIoRequest<'a>;

#[inline]
pub(super) fn prepare_candidate_upstream_request<'a>(
    state: &'a AppState,
    candidate_route: RouteTarget<'a>,
    stream: bool,
) -> CandidateUpstreamRequest<'a> {
    let prepared_upstream = &state.prepared_upstreams[candidate_route.upstream_index];
    prepare_upstream_io_request(
        state,
        prepared_upstream,
        candidate_route.actual_model,
        stream,
    )
}

#[inline]
pub(super) fn start_candidate_index<'a>(
    route_candidates: &[RouteTarget<'a>],
    route: RouteTarget<'a>,
) -> usize {
    if let Some(first) = route_candidates.first() {
        if first.upstream_index == route.upstream_index && first.actual_model == route.actual_model
        {
            return 0;
        }
    }

    route_candidates
        .iter()
        .position(|candidate| {
            candidate.upstream_index == route.upstream_index
                && candidate.actual_model == route.actual_model
        })
        .unwrap_or(0)
}
