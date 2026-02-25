use axum::response::Response;

use crate::error::CanonicalError;
use crate::fc::{self, FcResult};
use crate::protocol::canonical::{
    CanonicalPart, CanonicalRequest, CanonicalResponse, CanonicalStopReason, CanonicalToolSpec,
    IngressApi,
};

use super::{
    decode_response_from_provider, encode_for_provider, is_protocol_passthrough,
    rewrite_model_field_in_json_body_with_range, sanitize_upstream_error, send_non_streaming_bytes,
    UpstreamIoRequest,
};

#[inline]
fn ok_json_response(body_bytes: bytes::Bytes) -> Response {
    let mut response = Response::new(axum::body::Body::from(body_bytes));
    *response.status_mut() = http::StatusCode::OK;
    response.headers_mut().insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    response
}

#[inline]
fn maybe_rewrite_passthrough_response_model(
    body_bytes: &bytes::Bytes,
    client_model: &str,
    ingress: IngressApi,
) -> Option<bytes::Bytes> {
    if client_model.is_empty()
        || !matches!(
            ingress,
            IngressApi::OpenAiChat | IngressApi::OpenAiResponses
        )
    {
        return None;
    }
    rewrite_model_field_in_json_body_with_range(
        body_bytes,
        client_model,
        "upstream non-stream response",
        None,
    )
    .ok()
}

pub(crate) async fn handle_non_streaming_common<F>(
    ctx: UpstreamIoRequest<'_>,
    upstream_canonical: &CanonicalRequest,
    fc_active: bool,
    saved_tools: &[CanonicalToolSpec],
    passthrough_enabled: bool,
    ingress: IngressApi,
    encode_client_response: F,
) -> Result<Response, CanonicalError>
where
    F: Fn(&CanonicalResponse, &str) -> Result<Response, CanonicalError> + Copy,
{
    if !fc_active || !ctx.state.config.features.enable_fc_error_retry {
        let upstream_body = encode_for_provider(ctx.provider, upstream_canonical)?;
        return handle_non_streaming_preencoded_common(
            ctx,
            upstream_body,
            fc_active,
            saved_tools,
            passthrough_enabled,
            ingress,
            encode_client_response,
        )
        .await;
    }

    let mut retry_canonical: Option<CanonicalRequest> = None;
    let mut retry_ctx = fc::retry::RetryContext::new(&ctx.state.config.features);

    loop {
        let current_canonical = retry_canonical.as_ref().unwrap_or(upstream_canonical);
        let upstream_body = encode_for_provider(ctx.provider, current_canonical)?;
        let (status, body_bytes) = send_non_streaming_bytes(
            ctx.state,
            ctx.url,
            ctx.parsed_url,
            ctx.parsed_hyper_uri,
            ctx.proxy_url,
            ctx.preconfigured_proxy_client,
            ctx.upstream_headers,
            upstream_body,
        )
        .await?;

        if !status.is_success() {
            return Err(CanonicalError::Upstream {
                status: status.as_u16(),
                message: sanitize_upstream_error(&body_bytes),
            });
        }
        let maybe_fc_trigger = fc::response_text_contains_trigger(&body_bytes);

        if !maybe_fc_trigger && is_protocol_passthrough(ctx.provider, ingress) {
            if passthrough_enabled {
                return Ok(ok_json_response(body_bytes));
            }
            if let Some(rewritten) =
                maybe_rewrite_passthrough_response_model(&body_bytes, ctx.client_model, ingress)
            {
                return Ok(ok_json_response(rewritten));
            }
        }

        let mut upstream_response = decode_response_from_provider(ctx.provider, &body_bytes)?;

        // FC post-processing with optional retry for parse/validation failures.
        if maybe_fc_trigger {
            if let Some(response_text) =
                fc::extract_response_text_if_trigger(&upstream_response.content)
            {
                match fc::process_fc_response(response_text.as_ref(), saved_tools)? {
                    FcResult::ToolCalls {
                        tool_parts,
                        text_before,
                    } => {
                        let mut new_content: Vec<CanonicalPart> = Vec::with_capacity(
                            tool_parts.len() + usize::from(text_before.is_some()),
                        );
                        if let Some(text_before) = text_before {
                            new_content.push(CanonicalPart::Text(text_before));
                        }
                        new_content.extend(tool_parts);
                        upstream_response.content = new_content;
                        upstream_response.stop_reason = CanonicalStopReason::ToolCalls;
                    }
                    FcResult::NoToolCalls => {}
                    FcResult::ParseError {
                        trigger_found,
                        error,
                        original_text,
                    } => {
                        if retry_ctx.should_continue(trigger_found, true) {
                            let retry_prompt = fc::retry::build_retry_prompt(
                                &error,
                                &original_text,
                                retry_ctx.retry_template.as_deref(),
                            );
                            let retry_target =
                                retry_canonical.get_or_insert_with(|| upstream_canonical.clone());
                            retry_target.messages = fc::retry::build_retry_messages(
                                &retry_target.messages,
                                &original_text,
                                &retry_prompt,
                            );
                            retry_ctx.increment();
                            continue;
                        }
                        // Retry disabled/exhausted; pass through upstream response.
                    }
                }
            }
        }

        return encode_client_response(&upstream_response, ctx.client_model);
    }
}

pub(crate) async fn handle_non_streaming_preencoded_common<F>(
    ctx: UpstreamIoRequest<'_>,
    upstream_body: bytes::Bytes,
    fc_active: bool,
    saved_tools: &[CanonicalToolSpec],
    passthrough_enabled: bool,
    ingress: IngressApi,
    encode_client_response: F,
) -> Result<Response, CanonicalError>
where
    F: Fn(&CanonicalResponse, &str) -> Result<Response, CanonicalError> + Copy,
{
    let (status, body_bytes) = send_non_streaming_bytes(
        ctx.state,
        ctx.url,
        ctx.parsed_url,
        ctx.parsed_hyper_uri,
        ctx.proxy_url,
        ctx.preconfigured_proxy_client,
        ctx.upstream_headers,
        upstream_body,
    )
    .await?;

    if !status.is_success() {
        return Err(CanonicalError::Upstream {
            status: status.as_u16(),
            message: sanitize_upstream_error(&body_bytes),
        });
    }
    let maybe_fc_trigger = if fc_active {
        fc::response_text_contains_trigger(&body_bytes)
    } else {
        false
    };

    if is_protocol_passthrough(ctx.provider, ingress) {
        let should_passthrough = if fc_active { !maybe_fc_trigger } else { true };
        if should_passthrough {
            if passthrough_enabled {
                return Ok(ok_json_response(body_bytes));
            }
            if let Some(rewritten) =
                maybe_rewrite_passthrough_response_model(&body_bytes, ctx.client_model, ingress)
            {
                return Ok(ok_json_response(rewritten));
            }
        }
    }

    let mut upstream_response = decode_response_from_provider(ctx.provider, &body_bytes)?;
    if fc_active && maybe_fc_trigger {
        fc::apply_fc_postprocess_once(&mut upstream_response, saved_tools)?;
    }
    encode_client_response(&upstream_response, ctx.client_model)
}

#[cfg(test)]
mod tests {
    use super::maybe_rewrite_passthrough_response_model;
    use crate::protocol::canonical::IngressApi;

    #[test]
    fn test_rewrite_passthrough_response_model_openai_chat() {
        let body = bytes::Bytes::from_static(br#"{"id":"x","model":"m1","choices":[]}"#);
        let rewritten =
            maybe_rewrite_passthrough_response_model(&body, "m2", IngressApi::OpenAiChat)
                .expect("rewritten response body");
        let json: serde_json::Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(
            json.get("model").and_then(serde_json::Value::as_str),
            Some("m2")
        );
    }

    #[test]
    fn test_rewrite_passthrough_response_model_openai_responses() {
        let body = bytes::Bytes::from_static(br#"{"id":"x","object":"response","model":"m1"}"#);
        let rewritten =
            maybe_rewrite_passthrough_response_model(&body, "m2", IngressApi::OpenAiResponses)
                .expect("rewritten response body");
        let json: serde_json::Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(
            json.get("model").and_then(serde_json::Value::as_str),
            Some("m2")
        );
    }

    #[test]
    fn test_rewrite_passthrough_response_model_non_passthrough_ingress() {
        let body = bytes::Bytes::from_static(br#"{"id":"x","model":"m1"}"#);
        assert!(
            maybe_rewrite_passthrough_response_model(&body, "m2", IngressApi::Anthropic).is_none()
        );
    }
}
