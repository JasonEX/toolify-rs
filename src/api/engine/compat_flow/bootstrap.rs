use std::ops::Range;

use crate::api::common::CommonProbeRanges;
use crate::error::CanonicalError;
use crate::routing::RouteTarget;
use crate::state::AppState;

#[inline]
pub(crate) fn probe_messages_range(ranges: Option<&CommonProbeRanges>) -> Option<&Range<usize>> {
    ranges.and_then(CommonProbeRanges::messages_range)
}

#[inline]
pub(crate) fn start_candidate_index<'a>(
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

#[inline]
pub(crate) fn should_continue_stream_failover(
    state: &AppState,
    err: &CanonicalError,
    idx: usize,
    total: usize,
) -> bool {
    idx + 1 < total && state.should_try_alternate_upstream(err)
}
