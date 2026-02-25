use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalPart, CanonicalResponse};
use crate::util::next_generated_id;
use std::sync::atomic::AtomicU64;

use super::{ResponsesContentPart, ResponsesOutput, ResponsesOutputItem, ResponsesUsage};

static GENERATED_RESP_MSG_ID_SEQ: AtomicU64 = AtomicU64::new(1);

fn next_generated_message_item_id() -> String {
    next_generated_id("msg", &GENERATED_RESP_MSG_ID_SEQ)
}

/// Encode a canonical response into an `OpenAI` Responses API output.
///
/// # Errors
///
/// Returns [`CanonicalError`] when canonical tool payloads cannot be mapped to
/// Responses output items.
pub fn encode_responses_output(
    canonical: &CanonicalResponse,
    model: &str,
) -> Result<ResponsesOutput, CanonicalError> {
    let mut output_items: Vec<ResponsesOutputItem> = Vec::new();

    // Collect text and refusal parts into a message output item
    let mut content_parts: Vec<ResponsesContentPart> = Vec::new();
    let mut function_call_index: usize = 0;
    let mut function_result_index: usize = 0;

    for part in &canonical.content {
        match part {
            CanonicalPart::Text(text) => {
                content_parts.push(ResponsesContentPart::OutputText { text: text.clone() });
            }
            CanonicalPart::Refusal(refusal) => {
                content_parts.push(ResponsesContentPart::Refusal {
                    refusal: refusal.clone(),
                });
            }
            CanonicalPart::ToolCall {
                id,
                name,
                arguments,
            } => {
                let fc_id = format!("fc_{function_call_index}");
                function_call_index += 1;

                output_items.push(ResponsesOutputItem::FunctionCall {
                    id: fc_id,
                    call_id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.get().to_string(),
                });
            }
            CanonicalPart::ToolResult {
                tool_call_id,
                content,
            } => {
                output_items.push(ResponsesOutputItem::FunctionCallOutput {
                    id: format!("fco_{function_result_index}"),
                    call_id: tool_call_id.clone(),
                    output: content.clone(),
                });
                function_result_index += 1;
            }
            _ => {}
        }
    }

    // If we have text/refusal content, add a message item first
    if !content_parts.is_empty() {
        let msg_item = ResponsesOutputItem::Message {
            id: next_generated_message_item_id(),
            role: "assistant".into(),
            content: content_parts,
        };
        // Insert message before function calls
        output_items.insert(0, msg_item);
    }

    let usage = if canonical.usage.input_tokens.is_some() || canonical.usage.output_tokens.is_some()
    {
        let input = canonical.usage.input_tokens.unwrap_or(0);
        let output = canonical.usage.output_tokens.unwrap_or(0);
        Some(ResponsesUsage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: canonical.usage.total_tokens.or(Some(input + output)),
        })
    } else {
        None
    };

    Ok(ResponsesOutput {
        id: canonical.id.clone(),
        object: "response".into(),
        model: model.to_string(),
        output: output_items,
        usage,
        status: Some("completed".into()),
        extra: canonical.provider_extensions.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::{CanonicalStopReason, CanonicalUsage};

    #[test]
    fn test_encode_text_response() {
        let canonical = CanonicalResponse {
            id: "resp_123".into(),
            model: "gpt-4o".into(),
            content: vec![CanonicalPart::Text("Hello!".into())],
            stop_reason: CanonicalStopReason::EndOfTurn,
            usage: CanonicalUsage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                total_tokens: Some(15),
            },
            provider_extensions: serde_json::Map::new(),
        };

        let result = encode_responses_output(&canonical, "gpt-4o").unwrap();
        assert_eq!(result.id, "resp_123");
        assert_eq!(result.object, "response");
        assert_eq!(result.output.len(), 1);
        assert!(result.usage.is_some());
    }

    #[test]
    fn test_encode_tool_call_response() {
        let raw = serde_json::value::RawValue::from_string(r#"{"city":"SF"}"#.into()).unwrap();
        let canonical = CanonicalResponse {
            id: "resp_456".into(),
            model: "gpt-4o".into(),
            content: vec![
                CanonicalPart::Text("Let me check.".into()),
                CanonicalPart::ToolCall {
                    id: "call_abc".into(),
                    name: "get_weather".into(),
                    arguments: raw,
                },
            ],
            stop_reason: CanonicalStopReason::ToolCalls,
            usage: CanonicalUsage::default(),
            provider_extensions: serde_json::Map::new(),
        };

        let result = encode_responses_output(&canonical, "gpt-4o").unwrap();
        // Message first, then function call
        assert_eq!(result.output.len(), 2);
    }
}
