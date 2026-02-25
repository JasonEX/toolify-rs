use serde_json::Value;

use crate::error::CanonicalError;
use crate::protocol::canonical::{
    provider_extensions_to_map, CanonicalMessage, CanonicalPart, CanonicalRequest, CanonicalRole,
    CanonicalToolChoice, CanonicalToolSpec,
};
use crate::protocol::mapping::canonical_role_to_openai;

use super::{
    OpenAiChatRequest, OpenAiMessage, OpenAiStop, OpenAiTool, OpenAiToolCall,
    OpenAiToolCallFunction, OpenAiToolChoice, OpenAiToolChoiceFunction,
    OpenAiToolChoiceFunctionCall, OpenAiToolFunction,
};

/// Encode a canonical request into the `OpenAI` Chat Completions wire format.
///
/// # Errors
///
/// Returns [`CanonicalError`] when any canonical message cannot be encoded.
pub fn encode_openai_chat_request(
    canonical: &CanonicalRequest,
) -> Result<OpenAiChatRequest, CanonicalError> {
    let mut messages: Vec<OpenAiMessage> = Vec::with_capacity(
        canonical.messages.len() + usize::from(canonical.system_prompt.is_some()),
    );

    if let Some(ref system) = canonical.system_prompt {
        messages.push(OpenAiMessage {
            role: "system".to_string(),
            content: Some(Value::String(system.clone())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
        });
    }

    for msg in &canonical.messages {
        messages.push(encode_message(msg));
    }

    let tools = if canonical.tools.is_empty() {
        None
    } else {
        Some(canonical.tools.iter().map(encode_tool).collect())
    };

    let tool_choice = encode_tool_choice(&canonical.tool_choice, &canonical.tools);

    let stop = canonical.generation.stop.as_ref().map(|stops| {
        if stops.len() == 1 {
            OpenAiStop::Single(stops[0].clone())
        } else {
            OpenAiStop::Multi(stops.clone())
        }
    });

    Ok(OpenAiChatRequest {
        model: canonical.model.clone(),
        messages,
        tools,
        tool_choice,
        stream: if canonical.stream { Some(true) } else { None },
        stream_options: None,
        temperature: canonical.generation.temperature,
        max_tokens: canonical.generation.max_tokens,
        max_completion_tokens: None,
        top_p: canonical.generation.top_p,
        frequency_penalty: canonical.generation.frequency_penalty,
        presence_penalty: canonical.generation.presence_penalty,
        n: canonical.generation.n,
        stop,
        extra: provider_extensions_to_map(&canonical.provider_extensions),
    })
}

fn encode_message(msg: &CanonicalMessage) -> OpenAiMessage {
    let role = canonical_role_to_openai(msg.role).to_string();

    if msg.role == CanonicalRole::Tool {
        let (tool_call_id, content) = extract_tool_result(msg);
        return OpenAiMessage {
            role,
            content: Some(Value::String(content)),
            name: msg.name.clone(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
            refusal: None,
        };
    }

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<OpenAiToolCall> = Vec::new();
    let mut refusal: Option<String> = None;
    let mut has_image = false;
    let mut image_parts: Vec<Value> = Vec::new();

    for part in &msg.parts {
        match part {
            CanonicalPart::Text(t) => {
                text_parts.push(t.clone());
            }
            CanonicalPart::ImageUrl { url, detail } => {
                has_image = true;
                let mut img_obj = serde_json::json!({"url": url});
                if let Some(d) = detail {
                    img_obj["detail"] = Value::String(d.clone());
                }
                image_parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": img_obj,
                }));
            }
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
            CanonicalPart::ReasoningText(_) | CanonicalPart::ToolResult { .. } => {}
        }
    }

    let content = if has_image {
        let mut arr: Vec<Value> = text_parts
            .iter()
            .map(|t| serde_json::json!({"type": "text", "text": t}))
            .collect();
        arr.extend(image_parts);
        if arr.is_empty() {
            None
        } else {
            Some(Value::Array(arr))
        }
    } else if text_parts.is_empty() {
        None
    } else {
        Some(Value::String(text_parts.join("")))
    };

    let tool_calls_field = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    OpenAiMessage {
        role,
        content,
        name: msg.name.clone(),
        tool_calls: tool_calls_field,
        tool_call_id: msg.tool_call_id.clone(),
        refusal,
    }
}

fn extract_tool_result(msg: &CanonicalMessage) -> (String, String) {
    for part in &msg.parts {
        if let CanonicalPart::ToolResult {
            tool_call_id,
            content,
        } = part
        {
            return (tool_call_id.clone(), content.clone());
        }
    }
    (msg.tool_call_id.clone().unwrap_or_default(), String::new())
}

fn encode_tool(spec: &CanonicalToolSpec) -> OpenAiTool {
    OpenAiTool {
        type_: "function".to_string(),
        function: OpenAiToolFunction {
            name: spec.function.name.clone(),
            description: spec.function.description.clone(),
            parameters: Some(spec.function.parameters.clone()),
        },
    }
}

fn encode_tool_choice(
    choice: &CanonicalToolChoice,
    tools: &[CanonicalToolSpec],
) -> Option<OpenAiToolChoice> {
    if tools.is_empty() {
        return None;
    }
    match choice {
        CanonicalToolChoice::Auto => Some(OpenAiToolChoice::Mode("auto".to_string())),
        CanonicalToolChoice::None => Some(OpenAiToolChoice::Mode("none".to_string())),
        CanonicalToolChoice::Required => Some(OpenAiToolChoice::Mode("required".to_string())),
        CanonicalToolChoice::Specific(name) => {
            Some(OpenAiToolChoice::Function(OpenAiToolChoiceFunctionCall {
                type_: "function".to_string(),
                function: OpenAiToolChoiceFunction { name: name.clone() },
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::{GenerationParams, IngressApi};
    use crate::protocol::openai_chat::decoder::decode_openai_chat_request;

    fn make_canonical_request(messages: Vec<CanonicalMessage>) -> CanonicalRequest {
        CanonicalRequest {
            request_id: uuid::Uuid::nil(),
            ingress_api: IngressApi::OpenAiChat,
            model: "gpt-4".to_string(),
            stream: false,
            system_prompt: None,
            messages,
            tools: Vec::new().into(),
            tool_choice: CanonicalToolChoice::Auto,
            generation: GenerationParams::default(),
            provider_extensions: None,
        }
    }

    #[test]
    fn test_roundtrip_simple() {
        let wire: super::super::OpenAiChatRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        }))
        .unwrap();
        let canonical = decode_openai_chat_request(&wire, uuid::Uuid::nil()).unwrap();
        let re_encoded = encode_openai_chat_request(&canonical).unwrap();

        assert_eq!(re_encoded.model, "gpt-4");
        assert_eq!(re_encoded.messages.len(), 2);
        assert_eq!(re_encoded.messages[0].role, "system");
        assert_eq!(
            re_encoded.messages[0].content,
            Some(Value::String("Be concise.".to_string()))
        );
        assert_eq!(re_encoded.messages[1].role, "user");
    }

    #[test]
    fn test_encode_tool_calls() {
        let args = serde_json::value::RawValue::from_string("{\"x\":1}".to_string()).unwrap();
        let msg = CanonicalMessage {
            role: CanonicalRole::Assistant,
            parts: vec![CanonicalPart::ToolCall {
                id: "call_1".to_string(),
                name: "my_func".to_string(),
                arguments: args,
            }]
            .into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        };
        let req = make_canonical_request(vec![msg]);
        let wire = encode_openai_chat_request(&req).unwrap();
        let tc = wire.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].id, "call_1");
        assert_eq!(tc[0].function.name, "my_func");
        assert_eq!(tc[0].function.arguments, "{\"x\":1}");
    }

    #[test]
    fn test_encode_tool_result() {
        let msg = CanonicalMessage {
            role: CanonicalRole::Tool,
            parts: vec![CanonicalPart::ToolResult {
                tool_call_id: "call_1".to_string(),
                content: "42".to_string(),
            }]
            .into(),
            name: None,
            tool_call_id: Some("call_1".to_string()),
            provider_extensions: None,
        };
        let req = make_canonical_request(vec![msg]);
        let wire = encode_openai_chat_request(&req).unwrap();
        assert_eq!(wire.messages[0].role, "tool");
        assert_eq!(wire.messages[0].tool_call_id, Some("call_1".to_string()));
        assert_eq!(
            wire.messages[0].content,
            Some(Value::String("42".to_string()))
        );
    }
}
