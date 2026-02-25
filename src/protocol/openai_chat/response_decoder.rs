use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalPart, CanonicalResponse, CanonicalUsage};
use crate::protocol::mapping::openai_stop_to_canonical;
use crate::util::raw_value_from_string;
use serde::Deserialize;

use super::OpenAiChatResponse;

#[derive(Debug, Deserialize)]
struct OpenAiTextOnlyFastResponse<'a> {
    #[serde(borrow)]
    id: &'a str,
    #[serde(borrow)]
    model: &'a str,
    #[serde(borrow)]
    choices: Vec<OpenAiTextOnlyFastChoice<'a>>,
    usage: Option<OpenAiTextOnlyFastUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiTextOnlyFastChoice<'a> {
    #[serde(borrow)]
    message: OpenAiTextOnlyFastMessage<'a>,
    #[serde(default, borrow)]
    finish_reason: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAiTextOnlyFastContent<'a> {
    Text(&'a str),
    Parts(Vec<OpenAiTextOnlyFastContentPart<'a>>),
}

#[derive(Debug, Deserialize)]
struct OpenAiTextOnlyFastContentPart<'a> {
    #[serde(default, borrow, rename = "type")]
    part_type: Option<&'a str>,
    #[serde(default, borrow)]
    text: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct OpenAiTextOnlyFastMessage<'a> {
    #[serde(default, borrow)]
    content: Option<OpenAiTextOnlyFastContent<'a>>,
    #[serde(default, borrow)]
    refusal: Option<&'a str>,
    #[serde(default)]
    tool_calls: Option<serde::de::IgnoredAny>,
}

#[derive(Debug, Deserialize)]
struct OpenAiTextOnlyFastUsage {
    #[serde(rename = "prompt_tokens")]
    prompt: u64,
    #[serde(rename = "completion_tokens")]
    completion: u64,
    #[serde(rename = "total_tokens")]
    total: u64,
}

/// Attempt to decode a text-only `OpenAI` Chat response directly from bytes.
///
/// Returns `None` when payload is not in the expected fast-path shape.
#[must_use]
pub fn try_decode_openai_chat_text_response_bytes(body: &[u8]) -> Option<CanonicalResponse> {
    let parsed: OpenAiTextOnlyFastResponse<'_> = serde_json::from_slice(body).ok()?;
    let choice = parsed.choices.first()?;
    if choice.message.tool_calls.is_some() {
        return None;
    }

    let mut content: Vec<CanonicalPart> = Vec::new();
    if let Some(refusal) = choice.message.refusal {
        content.push(CanonicalPart::Refusal(refusal.to_owned()));
    }

    match &choice.message.content {
        Some(OpenAiTextOnlyFastContent::Text(text)) => {
            if !text.is_empty() {
                content.push(CanonicalPart::Text((*text).to_owned()));
            }
        }
        Some(OpenAiTextOnlyFastContent::Parts(parts)) => {
            for part in parts {
                let Some(text) = part.text else {
                    continue;
                };
                if text.is_empty() {
                    continue;
                }
                if let Some(part_type) = part.part_type {
                    if part_type != "text" {
                        continue;
                    }
                }
                content.push(CanonicalPart::Text(text.to_owned()));
            }
        }
        None => {}
    }

    let usage = parsed
        .usage
        .map_or_else(CanonicalUsage::default, |usage| CanonicalUsage {
            input_tokens: Some(usage.prompt),
            output_tokens: Some(usage.completion),
            total_tokens: Some(usage.total),
        });

    Some(CanonicalResponse {
        id: parsed.id.to_owned(),
        model: parsed.model.to_owned(),
        content,
        stop_reason: choice.finish_reason.map_or(
            crate::protocol::canonical::CanonicalStopReason::EndOfTurn,
            openai_stop_to_canonical,
        ),
        usage,
        provider_extensions: serde_json::Map::new(),
    })
}

/// Decode an `OpenAI` Chat Completions response into a canonical response.
///
/// # Errors
///
/// Returns [`CanonicalError`] when the response has no choices or a tool-call
/// argument payload cannot be converted into canonical raw JSON.
pub fn decode_openai_chat_response(
    response: &OpenAiChatResponse,
) -> Result<CanonicalResponse, CanonicalError> {
    let choice = response
        .choices
        .first()
        .ok_or_else(|| CanonicalError::Translation("OpenAI response has no choices".to_string()))?;

    let mut content: Vec<CanonicalPart> = Vec::new();

    if let Some(ref refusal) = choice.message.refusal {
        content.push(CanonicalPart::Refusal(refusal.clone()));
    }

    match &choice.message.content {
        Some(serde_json::Value::String(s)) => {
            if !s.is_empty() {
                content.push(CanonicalPart::Text(s.clone()));
            }
        }
        Some(serde_json::Value::Array(arr)) => {
            for part in arr {
                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                    content.push(CanonicalPart::Text(t.to_string()));
                }
            }
        }
        _ => {}
    }

    if let Some(ref tool_calls) = choice.message.tool_calls {
        for tc in tool_calls {
            let arguments =
                raw_value_from_string(tc.function.arguments.clone(), "OpenAI response tool call")?;
            content.push(CanonicalPart::ToolCall {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                arguments,
            });
        }
    }

    let stop_reason = choice.finish_reason.as_deref().map_or(
        crate::protocol::canonical::CanonicalStopReason::EndOfTurn,
        openai_stop_to_canonical,
    );

    let usage = match &response.usage {
        Some(u) => CanonicalUsage {
            input_tokens: Some(u.prompt_tokens),
            output_tokens: Some(u.completion_tokens),
            total_tokens: Some(u.total_tokens),
        },
        None => CanonicalUsage::default(),
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

/// Decode an `OpenAI` Chat Completions response by consuming ownership.
///
/// # Errors
///
/// Returns [`CanonicalError`] when the response has no choices or a tool-call
/// argument payload cannot be converted into canonical raw JSON.
pub fn decode_openai_chat_response_owned(
    response: OpenAiChatResponse,
) -> Result<CanonicalResponse, CanonicalError> {
    let OpenAiChatResponse {
        id,
        object: _,
        created: _,
        model,
        choices,
        usage,
    } = response;
    let choice = choices
        .into_iter()
        .next()
        .ok_or_else(|| CanonicalError::Translation("OpenAI response has no choices".to_string()))?;

    let mut content: Vec<CanonicalPart> = Vec::new();

    if let Some(refusal) = choice.message.refusal {
        content.push(CanonicalPart::Refusal(refusal));
    }

    match choice.message.content {
        Some(serde_json::Value::String(text)) => {
            if !text.is_empty() {
                content.push(CanonicalPart::Text(text));
            }
        }
        Some(serde_json::Value::Array(arr)) => {
            for part in arr {
                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                    content.push(CanonicalPart::Text(text.to_owned()));
                }
            }
        }
        None | Some(_) => {}
    }

    if let Some(tool_calls) = choice.message.tool_calls {
        for tc in tool_calls {
            let arguments =
                raw_value_from_string(tc.function.arguments, "OpenAI response tool call")?;
            content.push(CanonicalPart::ToolCall {
                id: tc.id,
                name: tc.function.name,
                arguments,
            });
        }
    }

    let stop_reason = choice.finish_reason.as_deref().map_or(
        crate::protocol::canonical::CanonicalStopReason::EndOfTurn,
        openai_stop_to_canonical,
    );

    let usage = usage.map_or_else(CanonicalUsage::default, |usage| CanonicalUsage {
        input_tokens: Some(usage.prompt_tokens),
        output_tokens: Some(usage.completion_tokens),
        total_tokens: Some(usage.total_tokens),
    });

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
    use serde_json::json;

    #[test]
    fn test_decode_text_response() {
        let resp: OpenAiChatResponse = serde_json::from_value(json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .unwrap();
        let canon = decode_openai_chat_response(&resp).unwrap();
        assert_eq!(canon.id, "chatcmpl-123");
        assert!(matches!(&canon.content[0], CanonicalPart::Text(t) if t == "Hello!"));
        assert_eq!(
            canon.stop_reason,
            crate::protocol::canonical::CanonicalStopReason::EndOfTurn
        );
        assert_eq!(canon.usage.input_tokens, Some(10));
    }

    #[test]
    fn test_decode_tool_call_response() {
        let resp: OpenAiChatResponse = serde_json::from_value(json!({
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"SF\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .unwrap();
        let canon = decode_openai_chat_response(&resp).unwrap();
        assert!(matches!(
            &canon.content[0],
            CanonicalPart::ToolCall { id, name, .. }
                if id == "call_abc" && name == "get_weather"
        ));
        assert_eq!(
            canon.stop_reason,
            crate::protocol::canonical::CanonicalStopReason::ToolCalls
        );
    }

    #[test]
    fn test_decode_empty_choices() {
        let resp: OpenAiChatResponse = serde_json::from_value(json!({
            "id": "chatcmpl-789",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": []
        }))
        .unwrap();
        let result = decode_openai_chat_response(&resp);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_owned_matches_borrowed() {
        let resp: OpenAiChatResponse = serde_json::from_value(json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .unwrap();
        let borrowed = decode_openai_chat_response(&resp).unwrap();
        let owned = decode_openai_chat_response_owned(resp).unwrap();
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

    #[test]
    fn test_try_decode_text_response_bytes() {
        let body = br#"{
            "id":"chatcmpl-fast",
            "object":"chat.completion",
            "model":"gpt-4o-mini",
            "choices":[{"index":0,"message":{"role":"assistant","content":"fast path"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}
        }"#;
        let decoded = try_decode_openai_chat_text_response_bytes(body).unwrap();
        assert_eq!(decoded.id, "chatcmpl-fast");
        assert_eq!(decoded.model, "gpt-4o-mini");
        assert!(matches!(
            decoded.content.first(),
            Some(CanonicalPart::Text(text)) if text == "fast path"
        ));
    }

    #[test]
    fn test_try_decode_text_response_bytes_rejects_tool_calls() {
        let body = br#"{
            "id":"chatcmpl-tool",
            "object":"chat.completion",
            "model":"gpt-4o-mini",
            "choices":[{"index":0,"message":{"role":"assistant","content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"foo","arguments":"{}"}}]},"finish_reason":"tool_calls"}]
        }"#;
        assert!(try_decode_openai_chat_text_response_bytes(body).is_none());
    }

    #[test]
    fn test_try_decode_text_response_bytes_array_content() {
        let body = br#"{
            "id":"chatcmpl-array",
            "object":"chat.completion",
            "model":"gpt-4o-mini",
            "choices":[{"index":0,"message":{"role":"assistant","content":[{"type":"text","text":"hello"},{"type":"text","text":" world"}]},"finish_reason":"stop"}]
        }"#;
        let decoded = try_decode_openai_chat_text_response_bytes(body).unwrap();
        assert_eq!(decoded.id, "chatcmpl-array");
        assert_eq!(decoded.content.len(), 2);
        assert!(matches!(
            decoded.content.first(),
            Some(CanonicalPart::Text(text)) if text == "hello"
        ));
        assert!(matches!(
            decoded.content.get(1),
            Some(CanonicalPart::Text(text)) if text == " world"
        ));
    }
}
