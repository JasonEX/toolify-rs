use crate::error::CanonicalError;
use crate::protocol::anthropic::{AnthropicContentBlock, AnthropicResponse};
use crate::protocol::canonical::{CanonicalPart, CanonicalResponse, CanonicalUsage};
use crate::protocol::mapping::anthropic_stop_to_canonical;

/// Decode an Anthropic Messages API response into canonical form.
///
/// # Errors
///
/// Returns [`CanonicalError`] when tool payloads cannot be serialized into
/// canonical raw JSON arguments.
pub fn decode_anthropic_response(
    response: &AnthropicResponse,
) -> Result<CanonicalResponse, CanonicalError> {
    // --- content blocks ---
    let mut content = Vec::new();
    for block in &response.content {
        match block {
            AnthropicContentBlock::Text { text } => {
                content.push(CanonicalPart::Text(text.clone()));
            }
            AnthropicContentBlock::Thinking { thinking } => {
                content.push(CanonicalPart::ReasoningText(thinking.clone()));
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                let raw = serde_json::value::to_raw_value(input).map_err(|e| {
                    CanonicalError::Translation(format!(
                        "Failed to convert Anthropic tool_use input arguments to RawValue: {e}"
                    ))
                })?;
                content.push(CanonicalPart::ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: raw,
                });
            }
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content: result_content,
            } => {
                let text = match result_content {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                };
                content.push(CanonicalPart::ToolResult {
                    tool_call_id: tool_use_id.clone(),
                    content: text,
                });
            }
        }
    }

    // --- stop reason ---
    let stop_reason = response.stop_reason.as_deref().map_or(
        crate::protocol::canonical::CanonicalStopReason::EndOfTurn,
        anthropic_stop_to_canonical,
    );

    // --- usage ---
    let input_tokens = response.usage.input_tokens;
    let output_tokens = response.usage.output_tokens;
    let usage = CanonicalUsage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
        total_tokens: Some(input_tokens + output_tokens),
    };

    Ok(CanonicalResponse {
        id: response.id.clone(),
        model: response.model.clone(),
        content,
        stop_reason,
        usage,
        provider_extensions: serde_json::Map::new(),
    })
}

/// Decode an Anthropic Messages API response by consuming ownership.
///
/// # Errors
///
/// Returns [`CanonicalError`] when tool payloads cannot be serialized into
/// canonical raw JSON arguments.
pub fn decode_anthropic_response_owned(
    response: AnthropicResponse,
) -> Result<CanonicalResponse, CanonicalError> {
    let AnthropicResponse {
        id,
        type_: _,
        role: _,
        model,
        content: blocks,
        stop_reason,
        stop_sequence: _,
        usage: usage_wire,
    } = response;

    let mut content = Vec::with_capacity(blocks.len());
    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text } => {
                content.push(CanonicalPart::Text(text));
            }
            AnthropicContentBlock::Thinking { thinking } => {
                content.push(CanonicalPart::ReasoningText(thinking));
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                let raw = serde_json::value::to_raw_value(&input).map_err(|e| {
                    CanonicalError::Translation(format!(
                        "Failed to convert Anthropic tool_use input arguments to RawValue: {e}"
                    ))
                })?;
                content.push(CanonicalPart::ToolCall {
                    id,
                    name,
                    arguments: raw,
                });
            }
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content: result_content,
            } => {
                let text = match result_content {
                    serde_json::Value::String(text) => text,
                    other => serde_json::to_string(&other).unwrap_or_default(),
                };
                content.push(CanonicalPart::ToolResult {
                    tool_call_id: tool_use_id,
                    content: text,
                });
            }
        }
    }

    let stop_reason = stop_reason.as_deref().map_or(
        crate::protocol::canonical::CanonicalStopReason::EndOfTurn,
        anthropic_stop_to_canonical,
    );

    let usage = CanonicalUsage {
        input_tokens: Some(usage_wire.input_tokens),
        output_tokens: Some(usage_wire.output_tokens),
        total_tokens: Some(usage_wire.input_tokens + usage_wire.output_tokens),
    };

    Ok(CanonicalResponse {
        id,
        model,
        content,
        stop_reason,
        usage,
        provider_extensions: serde_json::Map::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::anthropic::{AnthropicResponse, AnthropicUsage};

    #[test]
    fn test_decode_owned_matches_borrowed() {
        let response = AnthropicResponse {
            id: "msg_123".to_string(),
            type_: "message".to_string(),
            role: "assistant".to_string(),
            model: "claude-3-7-sonnet".to_string(),
            content: vec![
                AnthropicContentBlock::Text {
                    text: "Hello".to_string(),
                },
                AnthropicContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "lookup".to_string(),
                    input: serde_json::json!({"q":"rust"}),
                },
            ],
            stop_reason: Some("tool_use".to_string()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let borrowed = decode_anthropic_response(&response).unwrap();
        let owned = decode_anthropic_response_owned(response).unwrap();
        assert_eq!(borrowed.id, owned.id);
        assert_eq!(borrowed.model, owned.model);
        assert_eq!(borrowed.content.len(), owned.content.len());
        assert!(matches!(
            (borrowed.content.first(), owned.content.first()),
            (Some(CanonicalPart::Text(left)), Some(CanonicalPart::Text(right))) if left == right
        ));
        assert!(matches!(
            (borrowed.content.get(1), owned.content.get(1)),
            (
                Some(CanonicalPart::ToolCall {
                    id: left_id,
                    name: left_name,
                    ..
                }),
                Some(CanonicalPart::ToolCall {
                    id: right_id,
                    name: right_name,
                    ..
                })
            ) if left_id == right_id && left_name == right_name
        ));
        assert_eq!(borrowed.stop_reason, owned.stop_reason);
        assert_eq!(borrowed.usage.total_tokens, owned.usage.total_tokens);
    }
}
