use axum::response::Response;
use std::future::Future;

use crate::api::engine::pipeline::{PreparedUpstreamIoRequest, UpstreamIoRequest};
use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::canonical::{CanonicalRequest, CanonicalToolSpec, ProviderKind};
use crate::routing::RouteTarget;
use crate::state::AppState;

pub(crate) fn begin_auto_inject_retry(
    state: &AppState,
    route: RouteTarget<'_>,
    requested_model: &str,
    request_seq: u64,
    err: &CanonicalError,
) -> bool {
    if !fc::should_auto_fallback_to_inject(err) {
        state.record_upstream_failure(route.upstream_index, requested_model, err);
        return false;
    }

    state.record_upstream_failure(route.upstream_index, requested_model, err);
    state.mark_auto_inject(&route);
    tracing::debug!(
        request_id = %request_seq,
        upstream = %state.upstream_name(route.upstream_index),
        "fc_mode=auto: canonical native tool path unsupported, retrying with inject mode"
    );
    true
}

pub(crate) fn record_and_return(
    state: &AppState,
    route: RouteTarget<'_>,
    requested_model: &str,
    result: Result<Response, CanonicalError>,
) -> Result<Response, CanonicalError> {
    state.record_upstream_outcome(route.upstream_index, requested_model, &result);
    result
}

pub(crate) async fn run_auto_inject_fallback<W, C, FW, FC>(
    state: &AppState,
    route: RouteTarget<'_>,
    requested_model: &str,
    request_seq: u64,
    err: CanonicalError,
    prefer_wire: bool,
    wire_retry: FW,
    canonical_retry: FC,
) -> Result<Response, CanonicalError>
where
    W: Future<Output = Result<Response, CanonicalError>>,
    C: Future<Output = Result<Response, CanonicalError>>,
    FW: FnOnce() -> W,
    FC: FnOnce() -> C,
{
    if !begin_auto_inject_retry(state, route, requested_model, request_seq, &err) {
        return Err(err);
    }

    let result = if prefer_wire {
        wire_retry().await
    } else {
        canonical_retry().await
    };
    record_and_return(state, route, requested_model, result)
}

#[inline]
pub(crate) async fn run_preencoded_retry<'a, FS, FN, HS, HN>(
    io_target: &'a PreparedUpstreamIoRequest<'a>,
    client_model: &'a str,
    request_seq: u64,
    upstream_body: bytes::Bytes,
    stream: bool,
    fc_active: bool,
    saved_tools: &'a [CanonicalToolSpec],
    stream_handler: HS,
    non_stream_handler: HN,
) -> Result<Response, CanonicalError>
where
    FS: Future<Output = Result<Response, CanonicalError>>,
    FN: Future<Output = Result<Response, CanonicalError>>,
    HS: FnOnce(UpstreamIoRequest<'a>, bytes::Bytes, u64, bool, &'a [CanonicalToolSpec]) -> FS,
    HN: FnOnce(UpstreamIoRequest<'a>, bytes::Bytes, bool, &'a [CanonicalToolSpec]) -> FN,
{
    let io_ctx = io_target.io_ctx(client_model);
    if stream {
        return stream_handler(io_ctx, upstream_body, request_seq, fc_active, saved_tools).await;
    }
    non_stream_handler(io_ctx, upstream_body, fc_active, saved_tools).await
}

#[inline]
pub(crate) async fn run_canonical_retry<'a, FS, FN, HS, HN>(
    io_target: &'a PreparedUpstreamIoRequest<'a>,
    client_model: &'a str,
    request_seq: u64,
    provider: ProviderKind,
    canonical_request: &'a CanonicalRequest,
    fc_active: bool,
    saved_tools: &'a [CanonicalToolSpec],
    stream_handler: HS,
    non_stream_handler: HN,
) -> Result<Response, CanonicalError>
where
    FS: Future<Output = Result<Response, CanonicalError>>,
    FN: Future<Output = Result<Response, CanonicalError>>,
    HS: FnOnce(UpstreamIoRequest<'a>, bytes::Bytes, u64, bool, &'a [CanonicalToolSpec]) -> FS,
    HN: FnOnce(UpstreamIoRequest<'a>, &'a CanonicalRequest, bool, &'a [CanonicalToolSpec]) -> FN,
{
    let io_ctx = io_target.io_ctx(client_model);
    if canonical_request.stream {
        let upstream_body =
            crate::api::engine::pipeline::encode_for_provider(provider, canonical_request)?;
        return stream_handler(io_ctx, upstream_body, request_seq, fc_active, saved_tools).await;
    }
    non_stream_handler(io_ctx, canonical_request, fc_active, saved_tools).await
}
