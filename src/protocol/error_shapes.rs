use crate::error::ErrorCategory;

fn openai_error_type(cat: ErrorCategory) -> &'static str {
    match cat {
        ErrorCategory::InvalidRequest => "invalid_request_error",
        ErrorCategory::Authentication => "authentication_error",
        ErrorCategory::Permission => "permission_error",
        ErrorCategory::RateLimit => "rate_limit_error",
        ErrorCategory::ServerError | ErrorCategory::Unknown => "server_error",
    }
}

fn openai_error_code(cat: ErrorCategory) -> &'static str {
    match cat {
        ErrorCategory::InvalidRequest => "invalid_request",
        ErrorCategory::Authentication => "invalid_api_key",
        ErrorCategory::Permission => "permission_denied",
        ErrorCategory::RateLimit => "rate_limit_exceeded",
        ErrorCategory::ServerError | ErrorCategory::Unknown => "server_error",
    }
}

fn anthropic_error_type(cat: ErrorCategory) -> &'static str {
    match cat {
        ErrorCategory::InvalidRequest => "invalid_request_error",
        ErrorCategory::Authentication | ErrorCategory::Permission => "authentication_error",
        ErrorCategory::RateLimit => "rate_limit_error",
        ErrorCategory::ServerError | ErrorCategory::Unknown => "api_error",
    }
}

fn gemini_error_status(cat: ErrorCategory) -> &'static str {
    match cat {
        ErrorCategory::InvalidRequest => "INVALID_ARGUMENT",
        ErrorCategory::Authentication => "UNAUTHENTICATED",
        ErrorCategory::Permission => "PERMISSION_DENIED",
        ErrorCategory::RateLimit => "RESOURCE_EXHAUSTED",
        ErrorCategory::ServerError | ErrorCategory::Unknown => "INTERNAL",
    }
}

#[must_use]
pub(crate) fn openai_error_payload(cat: ErrorCategory, message: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message,
            "type": openai_error_type(cat),
            "code": openai_error_code(cat),
            "param": null,
        }
    })
}

#[must_use]
pub(crate) fn anthropic_error_payload(cat: ErrorCategory, message: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "error",
        "error": {
            "type": anthropic_error_type(cat),
            "message": message,
        }
    })
}

#[must_use]
pub(crate) fn gemini_error_payload(
    cat: ErrorCategory,
    status: http::StatusCode,
    message: &str,
) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "code": status.as_u16(),
            "message": message,
            "status": gemini_error_status(cat),
        }
    })
}
