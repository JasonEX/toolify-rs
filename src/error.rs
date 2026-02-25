use crate::protocol::canonical::IngressApi;
use crate::protocol::error_shapes::{
    anthropic_error_payload, gemini_error_payload, openai_error_payload,
};

/// Canonical error type used across all modules.
#[derive(Debug, thiserror::Error)]
pub enum CanonicalError {
    #[error("Config error: {0}")]
    Config(String),
    #[error("Auth error: {0}")]
    Auth(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Upstream error: status={status}, message={message}")]
    Upstream { status: u16, message: String },
    #[error("Transport error: {0}")]
    Transport(String),
    #[error("Protocol translation error: {0}")]
    Translation(String),
    #[error("FC parse error: {0}")]
    FcParse(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Broad error category for status code selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    InvalidRequest,
    Authentication,
    Permission,
    RateLimit,
    ServerError,
    Unknown,
}

/// Map an upstream HTTP status code to an error category (spec S7-I2).
#[must_use]
pub fn category_from_upstream_status(status: u16) -> ErrorCategory {
    match status {
        400 => ErrorCategory::InvalidRequest,
        401 => ErrorCategory::Authentication,
        403 => ErrorCategory::Permission,
        429 => ErrorCategory::RateLimit,
        500..=599 => ErrorCategory::ServerError,
        _ => ErrorCategory::Unknown,
    }
}

impl CanonicalError {
    #[must_use]
    pub fn category(&self) -> ErrorCategory {
        match self {
            CanonicalError::InvalidRequest(_) => ErrorCategory::InvalidRequest,
            CanonicalError::Auth(_) => ErrorCategory::Authentication,
            CanonicalError::Config(_)
            | CanonicalError::Transport(_)
            | CanonicalError::Translation(_)
            | CanonicalError::FcParse(_)
            | CanonicalError::Internal(_) => ErrorCategory::ServerError,
            CanonicalError::Upstream { status, .. } => category_from_upstream_status(*status),
        }
    }
}

// ---------------------------------------------------------------------------
// Category -> HTTP status code
// ---------------------------------------------------------------------------

fn http_status_for_category(cat: ErrorCategory) -> http::StatusCode {
    match cat {
        ErrorCategory::InvalidRequest => http::StatusCode::BAD_REQUEST,
        ErrorCategory::Authentication => http::StatusCode::UNAUTHORIZED,
        ErrorCategory::Permission => http::StatusCode::FORBIDDEN,
        ErrorCategory::RateLimit => http::StatusCode::TOO_MANY_REQUESTS,
        ErrorCategory::ServerError | ErrorCategory::Unknown => {
            http::StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

// ---------------------------------------------------------------------------
// Format an error for a given ingress API (spec S7-I1)
// ---------------------------------------------------------------------------

/// Format an error for a given ingress API, returning (`status_code`, JSON body).
#[must_use]
pub fn format_error(
    err: &CanonicalError,
    ingress: IngressApi,
) -> (http::StatusCode, serde_json::Value) {
    let cat = err.category();
    let status = http_status_for_category(cat);
    let message = err.to_string();

    let body = match ingress {
        IngressApi::OpenAiChat | IngressApi::OpenAiResponses => openai_error_payload(cat, &message),
        IngressApi::Anthropic => anthropic_error_payload(cat, &message),
        IngressApi::Gemini => gemini_error_payload(cat, status, &message),
    };

    (status, body)
}

// ---------------------------------------------------------------------------
// Axum integration
// ---------------------------------------------------------------------------

/// Convert a `CanonicalError` into an axum response for a specific ingress.
#[must_use]
pub fn into_axum_response(err: &CanonicalError, ingress: IngressApi) -> axum::response::Response {
    use axum::response::IntoResponse;
    let (status, body) = format_error(err, ingress);
    (status, axum::Json(body)).into_response()
}

/// Default `IntoResponse` implementation uses `OpenAiChat` as the fallback ingress.
/// Real handlers should call [`into_axum_response`] with the correct ingress instead.
impl axum::response::IntoResponse for CanonicalError {
    fn into_response(self) -> axum::response::Response {
        into_axum_response(&self, IngressApi::OpenAiChat)
    }
}
