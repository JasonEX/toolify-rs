use std::sync::Arc;

use axum::http::HeaderMap;
use axum::response::Response;

use crate::api::engine::compat_flow::run_compat_handler;
use crate::error::CanonicalError;
use crate::state::AppState;

use super::spec::AnthropicSpec;

pub(super) async fn handler_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: bytes::Bytes,
) -> Result<Response, CanonicalError> {
    run_compat_handler::<AnthropicSpec>(state, headers, body).await
}
