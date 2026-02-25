use std::sync::Arc;

use axum::extract::State;
use axum::response::Json;
use serde_json::{json, Value};

use crate::state::AppState;

/// Health check handler.
/// Returns JSON with status and config summary.
pub fn health_handler(State(state): State<Arc<AppState>>) -> Json<Value> {
    let config = &state.config;
    Json(json!({
        "status": "toolify-rs is running",
        "config": {
            "upstream_services_count": config.upstream_services.len(),
            "client_keys_count": config.client_authentication.allowed_keys.len(),
            "features": {
                "enable_function_calling": config.features.enable_function_calling,
                "log_level": config.features.log_level,
                "convert_developer_to_system": config.features.convert_developer_to_system,
                "enable_fc_error_retry": config.features.enable_fc_error_retry,
                "fc_error_retry_max_attempts": config.features.fc_error_retry_max_attempts,
            }
        }
    }))
}
