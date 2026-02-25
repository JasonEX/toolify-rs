use std::sync::Arc;

use axum::http::HeaderMap;
use axum::response::Response;

use crate::api::engine::compat_flow::run_compat_handler_with_route;
use crate::error::CanonicalError;
use crate::state::AppState;

use super::spec::{parse_model_action, GeminiSpec};

pub(super) async fn handler_inner(
    state: Arc<AppState>,
    model_action: &str,
    headers: HeaderMap,
    body: bytes::Bytes,
) -> Result<Response, CanonicalError> {
    let action = parse_model_action(model_action);
    run_compat_handler_with_route::<GeminiSpec>(
        state,
        headers,
        body,
        Some(action.model),
        Some(action.is_stream),
    )
    .await
}
