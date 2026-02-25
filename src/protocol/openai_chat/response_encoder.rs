use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalPart, CanonicalResponse};
use crate::protocol::mapping::canonical_stop_to_openai;

use super::{
    OpenAiChatResponse, OpenAiChoice, OpenAiMessage, OpenAiToolCall, OpenAiToolCallFunction,
    OpenAiUsage,
};

/// Encode a canonical response into the `OpenAI` Chat Completions wire format.
///
/// # Errors
///
/// Returns [`CanonicalError`] when required timestamps cannot be computed.
pub fn encode_openai_chat_response(
    canonical: &CanonicalResponse,
    model: &str,
) -> Result<OpenAiChatResponse, CanonicalError> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<OpenAiToolCall> = Vec::new();
    let mut refusal: Option<String> = None;

    for part in &canonical.content {
        match part {
            CanonicalPart::Text(t) => text_parts.push(t.clone()),
            CanonicalPart::ToolCall {
                id,
                name,
                arguments,
            } => {
                tool_calls.push(OpenAiToolCall {
                    id: id.clone(),
                    type_: "function".to_string(),
                    function: OpenAiToolCallFunction {
                        name: name.clone(),
                        arguments: arguments.get().to_string(),
                    },
                });
            }
            CanonicalPart::Refusal(r) => {
                refusal = Some(r.clone());
            }
            _ => {}
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(serde_json::Value::String(text_parts.join("")))
    };

    let tool_calls_field = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    let finish_reason = canonical_stop_to_openai(canonical.stop_reason).to_string();

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let usage = OpenAiUsage {
        prompt_tokens: canonical.usage.input_tokens.unwrap_or(0),
        completion_tokens: canonical.usage.output_tokens.unwrap_or(0),
        total_tokens: canonical.usage.total_tokens.unwrap_or(0),
    };

    Ok(OpenAiChatResponse {
        id: canonical.id.clone(),
        object: "chat.completion".to_string(),
        created: Some(created),
        model: model.to_string(),
        choices: vec![OpenAiChoice {
            index: 0,
            message: OpenAiMessage {
                role: "assistant".to_string(),
                content,
                name: None,
                tool_calls: tool_calls_field,
                tool_call_id: None,
                refusal,
            },
            finish_reason: Some(finish_reason),
        }],
        usage: Some(usage),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::{CanonicalStopReason, CanonicalUsage};

    #[test]
    fn test_encode_text_response() {
        let canonical = CanonicalResponse {
            id: "chatcmpl-123".to_string(),
            model: "gpt-4".to_string(),
            content: vec![CanonicalPart::Text("Hello world".to_string())],
            stop_reason: CanonicalStopReason::EndOfTurn,
            usage: CanonicalUsage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                total_tokens: Some(15),
            },
            provider_extensions: serde_json::Map::new(),
        };
        let wire = encode_openai_chat_response(&canonical, "gpt-4").unwrap();
        assert_eq!(wire.id, "chatcmpl-123");
        assert_eq!(wire.object, "chat.completion");
        assert_eq!(wire.choices[0].finish_reason, Some("stop".to_string()));
        assert_eq!(
            wire.choices[0].message.content,
            Some(serde_json::Value::String("Hello world".to_string()))
        );
        assert_eq!(wire.usage.as_ref().unwrap().prompt_tokens, 10);
    }

    #[test]
    fn test_encode_tool_call_response() {
        let args =
            serde_json::value::RawValue::from_string("{\"city\":\"LA\"}".to_string()).unwrap();
        let canonical = CanonicalResponse {
            id: "chatcmpl-456".to_string(),
            model: "gpt-4".to_string(),
            content: vec![CanonicalPart::ToolCall {
                id: "call_xyz".to_string(),
                name: "get_weather".to_string(),
                arguments: args,
            }],
            stop_reason: CanonicalStopReason::ToolCalls,
            usage: CanonicalUsage::default(),
            provider_extensions: serde_json::Map::new(),
        };
        let wire = encode_openai_chat_response(&canonical, "gpt-4").unwrap();
        let tc = wire.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].id, "call_xyz");
        assert_eq!(tc[0].function.arguments, "{\"city\":\"LA\"}");
        assert_eq!(
            wire.choices[0].finish_reason,
            Some("tool_calls".to_string())
        );
    }
}
