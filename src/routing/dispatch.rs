use std::convert::Infallible;
use std::sync::Arc;

use axum::body::{self, Body};
use axum::extract::State;
use axum::http::{Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};

use crate::api::{anthropic, gemini, health, models, openai_chat, openai_responses};
use crate::state::AppState;

const DEFAULT_BODY_LIMIT_BYTES: usize = 2 * 1024 * 1024;

enum RouteMatch<'a> {
    Health,
    Models,
    OpenAiChat,
    OpenAiResponses,
    Anthropic,
    Gemini { model_action: &'a str },
    MethodNotAllowed,
    NotFound,
}

/// Dispatch a raw HTTP request to the matching ingress handler.
///
/// # Errors
///
/// This function currently never returns `Err` and uses `Infallible`.
pub async fn dispatch_request(
    state: Arc<AppState>,
    base_path: Arc<str>,
    request: Request<Body>,
) -> Result<Response, Infallible> {
    let (parts, body) = request.into_parts();
    let route = match_route(&parts.method, parts.uri.path(), base_path.as_ref());

    let response = match route {
        RouteMatch::Health => health::health_handler(State(state)).into_response(),
        RouteMatch::Models => models::handler(State(state), &parts.headers).await,
        RouteMatch::OpenAiChat => {
            let body_bytes = match read_request_body(body).await {
                Ok(bytes) => bytes,
                Err(response) => return Ok(response),
            };
            openai_chat::handler(State(state), parts.headers, body_bytes).await
        }
        RouteMatch::OpenAiResponses => {
            let body_bytes = match read_request_body(body).await {
                Ok(bytes) => bytes,
                Err(response) => return Ok(response),
            };
            openai_responses::handler(State(state), parts.headers, body_bytes).await
        }
        RouteMatch::Anthropic => {
            let body_bytes = match read_request_body(body).await {
                Ok(bytes) => bytes,
                Err(response) => return Ok(response),
            };
            anthropic::handler(State(state), parts.headers, body_bytes).await
        }
        RouteMatch::Gemini { model_action } => {
            let body_bytes = match read_request_body(body).await {
                Ok(bytes) => bytes,
                Err(response) => return Ok(response),
            };
            gemini::handler_from_action(state, model_action, parts.headers, body_bytes).await
        }
        RouteMatch::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED.into_response(),
        RouteMatch::NotFound => StatusCode::NOT_FOUND.into_response(),
    };

    Ok(response)
}

#[must_use]
pub fn normalize_base_path(base_path: &str) -> String {
    let trimmed = base_path.trim();
    if trimmed.is_empty() || trimmed == "/" {
        String::new()
    } else if trimmed.starts_with('/') {
        trimmed.trim_end_matches('/').to_string()
    } else {
        format!("/{}", trimmed.trim_end_matches('/'))
    }
}

async fn read_request_body(body: Body) -> Result<bytes::Bytes, Response> {
    body::to_bytes(body, DEFAULT_BODY_LIMIT_BYTES)
        .await
        .map_err(|_| {
            (
                StatusCode::PAYLOAD_TOO_LARGE,
                "Request body too large (max 2MiB)",
            )
                .into_response()
        })
}

fn match_route<'a>(method: &Method, path: &'a str, base_path: &str) -> RouteMatch<'a> {
    let Some(path) = strip_base_path(path, base_path) else {
        return RouteMatch::NotFound;
    };

    match path {
        "/" => {
            if method == Method::GET {
                RouteMatch::Health
            } else {
                RouteMatch::MethodNotAllowed
            }
        }
        "/v1/models" => {
            if method == Method::GET {
                RouteMatch::Models
            } else {
                RouteMatch::MethodNotAllowed
            }
        }
        "/v1/chat/completions" => {
            if method == Method::POST {
                RouteMatch::OpenAiChat
            } else {
                RouteMatch::MethodNotAllowed
            }
        }
        "/v1/responses" => {
            if method == Method::POST {
                RouteMatch::OpenAiResponses
            } else {
                RouteMatch::MethodNotAllowed
            }
        }
        "/v1/messages" => {
            if method == Method::POST {
                RouteMatch::Anthropic
            } else {
                RouteMatch::MethodNotAllowed
            }
        }
        _ => {
            if let Some(model_action) = path.strip_prefix("/v1beta/models/") {
                if method != Method::POST {
                    RouteMatch::MethodNotAllowed
                } else if model_action.is_empty() {
                    RouteMatch::NotFound
                } else {
                    RouteMatch::Gemini { model_action }
                }
            } else {
                RouteMatch::NotFound
            }
        }
    }
}

fn strip_base_path<'a>(path: &'a str, base_path: &str) -> Option<&'a str> {
    if base_path.is_empty() {
        return Some(path);
    }

    let remainder = path.strip_prefix(base_path)?;
    if remainder.is_empty() {
        Some("/")
    } else if remainder.starts_with('/') {
        Some(remainder)
    } else {
        None
    }
}
