use crate::error::CanonicalError;
use crate::protocol::canonical::{
    provider_extensions_to_map, CanonicalMessage, CanonicalPart, CanonicalRequest, CanonicalRole,
    CanonicalToolChoice,
};

use super::{ResponsesRequest, ResponsesTool};

/// Encode a canonical request into an `OpenAI` Responses API request.
///
/// # Errors
///
/// Returns [`CanonicalError`] when canonical tool configuration cannot be
/// represented in Responses wire format.
pub fn encode_responses_request(
    canonical: &CanonicalRequest,
) -> Result<ResponsesRequest, CanonicalError> {
    let input = encode_messages(&canonical.messages);

    let mut tools: Vec<ResponsesTool> = canonical
        .tools
        .iter()
        .map(|t| ResponsesTool::Function {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            parameters: Some(t.function.parameters.clone()),
        })
        .collect();

    // Restore builtin tools from provider_extensions
    if let Some(serde_json::Value::Array(builtins)) = canonical
        .provider_extensions_ref()
        .get("responses_builtin_tools")
    {
        for builtin in builtins {
            if let Some(tool_type) = builtin.get("type").and_then(|v| v.as_str()) {
                let mut extra = serde_json::Map::new();
                if let Some(obj) = builtin.as_object() {
                    for (k, v) in obj {
                        if k != "type" {
                            extra.insert(k.clone(), v.clone());
                        }
                    }
                }
                match tool_type {
                    "web_search" => tools.push(ResponsesTool::WebSearch { extra }),
                    "file_search" => tools.push(ResponsesTool::FileSearch { extra }),
                    _ => {}
                }
            }
        }
    }

    let previous_response_id = canonical
        .provider_extensions_ref()
        .get("previous_response_id")
        .and_then(|v| v.as_str())
        .map(std::string::ToString::to_string);

    let store = canonical
        .provider_extensions_ref()
        .get("store")
        .and_then(serde_json::Value::as_bool);
    let tool_choice = if tools.is_empty() {
        None
    } else {
        Some(encode_responses_tool_choice(&canonical.tool_choice))
    };

    let mut extra = provider_extensions_to_map(&canonical.provider_extensions);
    extra.remove("responses_builtin_tools");
    extra.remove("previous_response_id");
    extra.remove("store");

    Ok(ResponsesRequest {
        model: canonical.model.clone(),
        input,
        instructions: canonical.system_prompt.clone(),
        tools: if tools.is_empty() { None } else { Some(tools) },
        tool_choice,
        previous_response_id,
        store,
        stream: if canonical.stream { Some(true) } else { None },
        temperature: canonical.generation.temperature,
        max_output_tokens: canonical.generation.max_tokens,
        top_p: canonical.generation.top_p,
        extra,
    })
}

fn encode_responses_tool_choice(choice: &CanonicalToolChoice) -> serde_json::Value {
    match choice {
        CanonicalToolChoice::Auto => serde_json::Value::String("auto".to_string()),
        CanonicalToolChoice::None => serde_json::Value::String("none".to_string()),
        CanonicalToolChoice::Required => serde_json::Value::String("required".to_string()),
        CanonicalToolChoice::Specific(name) => {
            serde_json::json!({"type":"function","name":name})
        }
    }
}

/// Encode canonical messages into the Responses API `input` JSON value.
fn encode_messages(messages: &[CanonicalMessage]) -> serde_json::Value {
    let items: Vec<serde_json::Value> = messages.iter().flat_map(encode_message).collect();
    serde_json::Value::Array(items)
}

/// Encode a single canonical message into one or more Responses API input items.
fn encode_message(msg: &CanonicalMessage) -> Vec<serde_json::Value> {
    let mut items = Vec::new();

    match msg.role {
        CanonicalRole::User | CanonicalRole::System => {
            let role = match msg.role {
                CanonicalRole::System => "developer",
                _ => "user",
            };
            let text_parts: Vec<&str> = msg
                .parts
                .iter()
                .filter_map(|p| match p {
                    CanonicalPart::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect();

            if !text_parts.is_empty() {
                let content: Vec<serde_json::Value> = text_parts
                    .into_iter()
                    .map(|t| {
                        serde_json::json!({
                            "type": "input_text",
                            "text": t
                        })
                    })
                    .collect();

                items.push(serde_json::json!({
                    "type": "message",
                    "role": role,
                    "content": content,
                }));
            }
        }
        CanonicalRole::Assistant => {
            // Assistant messages: text parts become message items,
            // tool call parts become function_call items.
            let text_parts: Vec<&str> = msg
                .parts
                .iter()
                .filter_map(|p| match p {
                    CanonicalPart::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect();

            if !text_parts.is_empty() {
                let content: Vec<serde_json::Value> = text_parts
                    .into_iter()
                    .map(|t| {
                        serde_json::json!({
                            "type": "output_text",
                            "text": t
                        })
                    })
                    .collect();

                items.push(serde_json::json!({
                    "type": "message",
                    "role": "assistant",
                    "content": content,
                }));
            }

            for part in &msg.parts {
                if let CanonicalPart::ToolCall {
                    id,
                    name,
                    arguments,
                } = part
                {
                    items.push(serde_json::json!({
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        "arguments": arguments.get(),
                    }));
                }
            }
        }
        CanonicalRole::Tool => {
            for part in &msg.parts {
                if let CanonicalPart::ToolResult {
                    tool_call_id,
                    content,
                } = part
                {
                    items.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": content,
                    }));
                }
            }
        }
    }

    items
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::{
        CanonicalToolChoice, CanonicalToolFunction, CanonicalToolSpec, GenerationParams, IngressApi,
    };

    fn make_canonical() -> CanonicalRequest {
        CanonicalRequest {
            request_id: uuid::Uuid::from_u128(1),
            ingress_api: IngressApi::OpenAiResponses,
            model: "gpt-4o".into(),
            stream: false,
            system_prompt: Some("Be helpful.".into()),
            messages: vec![CanonicalMessage {
                role: CanonicalRole::User,
                parts: vec![CanonicalPart::Text("Hello".into())].into(),
                name: None,
                tool_call_id: None,
                provider_extensions: None,
            }],
            tools: vec![CanonicalToolSpec {
                function: CanonicalToolFunction {
                    name: "get_weather".into(),
                    description: Some("Get weather".into()),
                    parameters: serde_json::json!({"type": "object"}),
                },
            }]
            .into(),
            tool_choice: CanonicalToolChoice::Auto,
            generation: GenerationParams::default(),
            provider_extensions: None,
        }
    }

    #[test]
    fn test_encode_basic_request() {
        let canonical = make_canonical();
        let encoded = encode_responses_request(&canonical).unwrap();
        assert_eq!(encoded.model, "gpt-4o");
        assert_eq!(encoded.instructions, Some("Be helpful.".into()));
        assert!(encoded.input.is_array());
        assert_eq!(encoded.tools.as_ref().unwrap().len(), 1);
        assert_eq!(encoded.tool_choice, Some(serde_json::json!("auto")));
    }

    #[test]
    fn test_previous_response_id_roundtrip() {
        let mut canonical = make_canonical();
        canonical.provider_extensions_mut().insert(
            "previous_response_id".into(),
            serde_json::Value::String("resp_abc".into()),
        );

        let encoded = encode_responses_request(&canonical).unwrap();
        assert_eq!(encoded.previous_response_id, Some("resp_abc".into()));
        assert!(!encoded.extra.contains_key("previous_response_id"));
    }

    #[test]
    fn test_encode_specific_tool_choice() {
        let mut canonical = make_canonical();
        canonical.tool_choice = CanonicalToolChoice::Specific("get_weather".into());
        let encoded = encode_responses_request(&canonical).unwrap();
        assert_eq!(
            encoded.tool_choice,
            Some(serde_json::json!({"type":"function","name":"get_weather"}))
        );
    }
}
