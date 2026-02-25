use crate::error::CanonicalError;
use crate::protocol::canonical::{
    CanonicalPart, CanonicalResponse, CanonicalStopReason, CanonicalUsage,
};
use crate::util::raw_value_from_string;

use super::{ResponsesContentPart, ResponsesOutput, ResponsesOutputItem};

/// Decode an `OpenAI` Responses API output into a canonical response.
///
/// # Errors
///
/// Returns [`CanonicalError`] when function-call argument payloads cannot be
/// converted into canonical raw JSON.
pub fn decode_responses_output(
    output: &ResponsesOutput,
) -> Result<CanonicalResponse, CanonicalError> {
    let mut parts = Vec::new();
    let mut has_function_call = false;
    let mut has_tool_result = false;

    for item in &output.output {
        match item {
            ResponsesOutputItem::Message { content, .. } => {
                for cp in content {
                    match cp {
                        ResponsesContentPart::OutputText { text } => {
                            parts.push(CanonicalPart::Text(text.clone()));
                        }
                        ResponsesContentPart::Refusal { refusal } => {
                            parts.push(CanonicalPart::Refusal(refusal.clone()));
                        }
                    }
                }
            }
            ResponsesOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                has_function_call = true;
                let raw = raw_value_from_string(arguments.clone(), "Responses function_call")
                    .or_else(|_| {
                        raw_value_from_string("{}".to_string(), "Responses function_call")
                    })?;
                parts.push(CanonicalPart::ToolCall {
                    id: call_id.clone(),
                    name: name.clone(),
                    arguments: raw,
                });
            }
            ResponsesOutputItem::FunctionCallOutput {
                call_id, output, ..
            } => {
                has_tool_result = true;
                parts.push(CanonicalPart::ToolResult {
                    tool_call_id: call_id.clone(),
                    content: output.clone(),
                });
            }
        }
    }

    let stop_reason = if has_function_call || has_tool_result {
        CanonicalStopReason::ToolCalls
    } else {
        CanonicalStopReason::EndOfTurn
    };

    let usage = output
        .usage
        .as_ref()
        .map(|u| {
            let total = u.total_tokens.unwrap_or(u.input_tokens + u.output_tokens);
            CanonicalUsage {
                input_tokens: Some(u.input_tokens),
                output_tokens: Some(u.output_tokens),
                total_tokens: Some(total),
            }
        })
        .unwrap_or_default();

    Ok(CanonicalResponse {
        id: output.id.clone(),
        model: output.model.clone(),
        content: parts,
        stop_reason,
        usage,
        provider_extensions: output.extra.clone(),
    })
}

/// Decode an `OpenAI` Responses API output by consuming ownership.
///
/// # Errors
///
/// Returns [`CanonicalError`] when function-call argument payloads cannot be
/// converted into canonical raw JSON.
pub fn decode_responses_output_owned(
    output: ResponsesOutput,
) -> Result<CanonicalResponse, CanonicalError> {
    let ResponsesOutput {
        id,
        object: _,
        model,
        output: output_items,
        usage,
        status: _,
        extra,
    } = output;

    let mut content = Vec::new();
    let mut has_function_call = false;
    let mut has_tool_result = false;

    for output_item in output_items {
        match output_item {
            ResponsesOutputItem::Message { content: parts, .. } => {
                for part in parts {
                    match part {
                        ResponsesContentPart::OutputText { text } => {
                            content.push(CanonicalPart::Text(text));
                        }
                        ResponsesContentPart::Refusal { refusal } => {
                            content.push(CanonicalPart::Refusal(refusal));
                        }
                    }
                }
            }
            ResponsesOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                has_function_call = true;
                let args =
                    raw_value_from_string(arguments, "Responses function_call").or_else(|_| {
                        raw_value_from_string("{}".to_string(), "Responses function_call")
                    })?;
                content.push(CanonicalPart::ToolCall {
                    id: call_id,
                    name,
                    arguments: args,
                });
            }
            ResponsesOutputItem::FunctionCallOutput {
                call_id,
                output: value,
                ..
            } => {
                has_tool_result = true;
                content.push(CanonicalPart::ToolResult {
                    tool_call_id: call_id,
                    content: value,
                });
            }
        }
    }

    let stop_reason = if has_function_call || has_tool_result {
        CanonicalStopReason::ToolCalls
    } else {
        CanonicalStopReason::EndOfTurn
    };

    let usage = usage.map_or_else(CanonicalUsage::default, |usage| {
        let total_tokens = usage
            .total_tokens
            .unwrap_or(usage.input_tokens + usage.output_tokens);
        CanonicalUsage {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            total_tokens: Some(total_tokens),
        }
    });

    Ok(CanonicalResponse {
        id,
        model,
        content,
        stop_reason,
        usage,
        provider_extensions: extra,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_text_response() {
        let output = ResponsesOutput {
            id: "resp_123".into(),
            object: "response".into(),
            model: "gpt-4o".into(),
            output: vec![ResponsesOutputItem::Message {
                id: "msg_1".into(),
                role: "assistant".into(),
                content: vec![ResponsesContentPart::OutputText {
                    text: "Hello!".into(),
                }],
            }],
            usage: Some(super::super::ResponsesUsage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: Some(15),
            }),
            status: Some("completed".into()),
            extra: serde_json::Map::new(),
        };

        let result = decode_responses_output(&output).unwrap();
        assert_eq!(result.id, "resp_123");
        assert_eq!(result.stop_reason, CanonicalStopReason::EndOfTurn);
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.usage.input_tokens, Some(10));
    }

    #[test]
    fn test_decode_function_call_response() {
        let output = ResponsesOutput {
            id: "resp_456".into(),
            object: "response".into(),
            model: "gpt-4o".into(),
            output: vec![
                ResponsesOutputItem::Message {
                    id: "msg_1".into(),
                    role: "assistant".into(),
                    content: vec![],
                },
                ResponsesOutputItem::FunctionCall {
                    id: "fc_1".into(),
                    call_id: "call_abc".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"city":"SF"}"#.into(),
                },
            ],
            usage: None,
            status: None,
            extra: serde_json::Map::new(),
        };

        let result = decode_responses_output(&output).unwrap();
        assert_eq!(result.stop_reason, CanonicalStopReason::ToolCalls);
        assert_eq!(result.content.len(), 1); // only the function call
    }

    #[test]
    fn test_decode_function_call_output_response() {
        let output = ResponsesOutput {
            id: "resp_789".into(),
            object: "response".into(),
            model: "gpt-4o".into(),
            output: vec![
                ResponsesOutputItem::FunctionCallOutput {
                    id: "fco_0".into(),
                    call_id: "call_abc".into(),
                    output: r#"{"temp":72}"#.into(),
                },
                ResponsesOutputItem::FunctionCall {
                    id: "fc_0".into(),
                    call_id: "call_abc".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"city":"SF"}"#.into(),
                },
            ],
            usage: None,
            status: None,
            extra: serde_json::Map::new(),
        };

        let result = decode_responses_output(&output).unwrap();
        assert_eq!(result.stop_reason, CanonicalStopReason::ToolCalls);
        assert_eq!(result.content.len(), 2);
        assert!(matches!(
            &result.content[0],
            CanonicalPart::ToolResult { .. }
        ));
        assert!(matches!(&result.content[1], CanonicalPart::ToolCall { .. }));
    }

    #[test]
    fn test_decode_owned_matches_borrowed() {
        let output = ResponsesOutput {
            id: "resp_123".into(),
            object: "response".into(),
            model: "gpt-4o".into(),
            output: vec![ResponsesOutputItem::Message {
                id: "msg_1".into(),
                role: "assistant".into(),
                content: vec![ResponsesContentPart::OutputText {
                    text: "Hello!".into(),
                }],
            }],
            usage: Some(super::super::ResponsesUsage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: Some(15),
            }),
            status: Some("completed".into()),
            extra: serde_json::Map::new(),
        };

        let borrowed = decode_responses_output(&output).unwrap();
        let owned = decode_responses_output_owned(output).unwrap();
        assert_eq!(borrowed.id, owned.id);
        assert_eq!(borrowed.model, owned.model);
        assert_eq!(borrowed.content.len(), owned.content.len());
        assert!(matches!(
            (borrowed.content.first(), owned.content.first()),
            (Some(CanonicalPart::Text(left)), Some(CanonicalPart::Text(right))) if left == right
        ));
        assert_eq!(borrowed.stop_reason, owned.stop_reason);
        assert_eq!(borrowed.usage.total_tokens, owned.usage.total_tokens);
    }
}
