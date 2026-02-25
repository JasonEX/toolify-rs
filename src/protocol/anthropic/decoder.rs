use crate::error::CanonicalError;
use crate::protocol::anthropic::AnthropicRequest;
use crate::protocol::canonical::{
    provider_extensions_from_map, CanonicalMessage, CanonicalPart, CanonicalRequest, CanonicalRole,
    CanonicalToolChoice, CanonicalToolFunction, CanonicalToolSpec, GenerationParams, IngressApi,
};
use crate::protocol::mapping::anthropic_role_to_canonical;
use crate::util::raw_value_from_string;

/// Parse an Anthropic Messages API request into canonical form.
///
/// # Errors
///
/// Returns [`CanonicalError`] when content blocks cannot be translated into
/// canonical parts.
pub fn decode_anthropic_request(
    request: &AnthropicRequest,
    request_id: uuid::Uuid,
) -> Result<CanonicalRequest, CanonicalError> {
    let system_prompt = decode_system_prompt(request.system.as_ref());

    // --- messages ---
    let mut messages = Vec::new();
    for msg in &request.messages {
        let mut role = anthropic_role_to_canonical(&msg.role);
        let parts = decode_content_value(&msg.content, role)?;
        if role == CanonicalRole::User
            && !parts.is_empty()
            && parts
                .iter()
                .all(|part| matches!(part, CanonicalPart::ToolResult { .. }))
        {
            role = CanonicalRole::Tool;
        }
        messages.push(CanonicalMessage {
            role,
            parts: parts.into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        });
    }

    let tools = decode_anthropic_tools(request.tools.as_ref());

    // --- tool_choice ---
    let tool_choice = request
        .tool_choice
        .as_ref()
        .map_or(CanonicalToolChoice::Auto, decode_tool_choice);

    // --- generation params ---
    let generation = GenerationParams {
        max_tokens: Some(request.max_tokens),
        temperature: request.temperature,
        top_p: request.top_p,
        stop: request.stop_sequences.clone(),
        ..Default::default()
    };

    Ok(build_anthropic_request(
        request_id,
        request.model.clone(),
        request.stream.unwrap_or(false),
        system_prompt,
        messages,
        tools,
        tool_choice,
        generation,
        request.extra.clone(),
    ))
}

/// Parse an Anthropic Messages API request into canonical form by consuming
/// ownership to reduce clone cost on hot paths.
///
/// # Errors
///
/// Returns [`CanonicalError`] when content blocks cannot be translated into
/// canonical parts.
pub fn decode_anthropic_request_owned(
    request: AnthropicRequest,
    request_id: uuid::Uuid,
) -> Result<CanonicalRequest, CanonicalError> {
    let AnthropicRequest {
        model,
        max_tokens,
        system,
        messages: wire_messages,
        tools: wire_tools,
        tool_choice: wire_tool_choice,
        stream,
        temperature,
        top_p,
        stop_sequences,
        extra,
    } = request;

    let system_prompt = decode_system_prompt_owned(system);

    let mut messages = Vec::with_capacity(wire_messages.len());
    for msg in wire_messages {
        let mut role = anthropic_role_to_canonical(&msg.role);
        let parts = decode_content_value_owned(msg.content, role)?;
        if role == CanonicalRole::User
            && !parts.is_empty()
            && parts
                .iter()
                .all(|part| matches!(part, CanonicalPart::ToolResult { .. }))
        {
            role = CanonicalRole::Tool;
        }
        messages.push(CanonicalMessage {
            role,
            parts: parts.into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        });
    }

    let tools = decode_anthropic_tools_owned(wire_tools);

    let tool_choice = wire_tool_choice
        .as_ref()
        .map_or(CanonicalToolChoice::Auto, decode_tool_choice);

    let generation = GenerationParams {
        max_tokens: Some(max_tokens),
        temperature,
        top_p,
        stop: stop_sequences,
        ..Default::default()
    };

    Ok(build_anthropic_request(
        request_id,
        model,
        stream.unwrap_or(false),
        system_prompt,
        messages,
        tools,
        tool_choice,
        generation,
        extra,
    ))
}

fn build_anthropic_request(
    request_id: uuid::Uuid,
    model: String,
    stream: bool,
    system_prompt: Option<String>,
    messages: Vec<CanonicalMessage>,
    tools: Vec<CanonicalToolSpec>,
    tool_choice: CanonicalToolChoice,
    generation: GenerationParams,
    provider_extra: serde_json::Map<String, serde_json::Value>,
) -> CanonicalRequest {
    CanonicalRequest {
        request_id,
        ingress_api: IngressApi::Anthropic,
        model,
        stream,
        system_prompt,
        messages,
        tools: tools.into(),
        tool_choice,
        generation,
        provider_extensions: provider_extensions_from_map(provider_extra),
    }
}

/// Decode a content JSON value (string or array of blocks) into canonical parts.
fn decode_content_value(
    content: &serde_json::Value,
    _role: CanonicalRole,
) -> Result<Vec<CanonicalPart>, CanonicalError> {
    match content {
        serde_json::Value::String(s) => Ok(vec![CanonicalPart::Text(s.clone())]),
        serde_json::Value::Array(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");
                match block_type {
                    "text" => {
                        let text = block
                            .get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string();
                        parts.push(CanonicalPart::Text(text));
                    }
                    "tool_use" => {
                        let id = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = block.get("input").cloned().unwrap_or(serde_json::json!({}));
                        let arguments_str = serde_json::to_string(&input).unwrap_or_default();
                        let raw = raw_value_from_string(arguments_str, "Anthropic tool_use input")?;
                        parts.push(CanonicalPart::ToolCall {
                            id,
                            name,
                            arguments: raw,
                        });
                    }
                    "tool_result" => {
                        let tool_use_id = block
                            .get("tool_use_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let content_val = block
                            .get("content")
                            .cloned()
                            .unwrap_or(serde_json::json!(""));
                        let content_str = match &content_val {
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Array(arr) => {
                                // Concatenate text blocks from tool_result content array
                                arr.iter()
                                    .filter_map(|b| {
                                        if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                                            b.get("text").and_then(|t| t.as_str()).map(String::from)
                                        } else {
                                            None
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            }
                            other => serde_json::to_string(other).unwrap_or_default(),
                        };
                        parts.push(CanonicalPart::ToolResult {
                            tool_call_id: tool_use_id,
                            content: content_str,
                        });
                    }
                    "thinking" => {
                        let thinking = block
                            .get("thinking")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string();
                        parts.push(CanonicalPart::ReasoningText(thinking));
                    }
                    _ => {
                        // Unknown block type — skip
                    }
                }
            }
            Ok(parts)
        }
        _ => Ok(vec![]),
    }
}

fn decode_system_prompt_owned(system: Option<serde_json::Value>) -> Option<String> {
    let system = system?;
    match system {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Array(blocks) => {
            let mut texts: Vec<String> = Vec::new();
            for block in blocks {
                let serde_json::Value::Object(mut obj) = block else {
                    continue;
                };
                if obj.get("type").and_then(|t| t.as_str()) != Some("text") {
                    continue;
                }
                if let Some(serde_json::Value::String(text)) = obj.remove("text") {
                    texts.push(text);
                }
            }
            if texts.is_empty() {
                None
            } else {
                Some(texts.join("\n"))
            }
        }
        _ => None,
    }
}

fn decode_system_prompt(system: Option<&serde_json::Value>) -> Option<String> {
    let system = system?;
    match system {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(blocks) => {
            let texts: Vec<String> = blocks
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                        block
                            .get("text")
                            .and_then(|t| t.as_str())
                            .map(std::string::ToString::to_string)
                    } else {
                        None
                    }
                })
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join("\n"))
            }
        }
        _ => None,
    }
}

fn decode_anthropic_tools(
    tools: Option<&Vec<crate::protocol::anthropic::AnthropicTool>>,
) -> Vec<CanonicalToolSpec> {
    tools
        .map(|items| {
            items
                .iter()
                .map(|tool| CanonicalToolSpec {
                    function: CanonicalToolFunction {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: tool.input_schema.clone(),
                    },
                })
                .collect()
        })
        .unwrap_or_default()
}

fn decode_anthropic_tools_owned(
    tools: Option<Vec<crate::protocol::anthropic::AnthropicTool>>,
) -> Vec<CanonicalToolSpec> {
    tools
        .map(|items| {
            items
                .into_iter()
                .map(|tool| CanonicalToolSpec {
                    function: CanonicalToolFunction {
                        name: tool.name,
                        description: tool.description,
                        parameters: tool.input_schema,
                    },
                })
                .collect()
        })
        .unwrap_or_default()
}

fn decode_content_value_owned(
    content: serde_json::Value,
    _role: CanonicalRole,
) -> Result<Vec<CanonicalPart>, CanonicalError> {
    match content {
        serde_json::Value::String(s) => Ok(vec![CanonicalPart::Text(s)]),
        serde_json::Value::Array(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                let serde_json::Value::Object(mut obj) = block else {
                    continue;
                };
                let block_type = obj
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("text")
                    .to_string();
                match block_type.as_str() {
                    "text" => {
                        let text = match obj.remove("text") {
                            Some(serde_json::Value::String(s)) => s,
                            _ => String::new(),
                        };
                        parts.push(CanonicalPart::Text(text));
                    }
                    "tool_use" => {
                        let id = match obj.remove("id") {
                            Some(serde_json::Value::String(s)) => s,
                            _ => String::new(),
                        };
                        let name = match obj.remove("name") {
                            Some(serde_json::Value::String(s)) => s,
                            _ => String::new(),
                        };
                        let input = obj
                            .remove("input")
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        let arguments_str = serde_json::to_string(&input).unwrap_or_default();
                        let raw = raw_value_from_string(arguments_str, "Anthropic tool_use input")?;
                        parts.push(CanonicalPart::ToolCall {
                            id,
                            name,
                            arguments: raw,
                        });
                    }
                    "tool_result" => {
                        let tool_use_id = match obj.remove("tool_use_id") {
                            Some(serde_json::Value::String(s)) => s,
                            _ => String::new(),
                        };
                        let content_val = obj
                            .remove("content")
                            .unwrap_or(serde_json::Value::String(String::new()));
                        let content_str = match content_val {
                            serde_json::Value::String(s) => s,
                            serde_json::Value::Array(items) => {
                                let mut texts = Vec::new();
                                for item in items {
                                    let serde_json::Value::Object(mut item_obj) = item else {
                                        continue;
                                    };
                                    if item_obj.get("type").and_then(|t| t.as_str()) == Some("text")
                                    {
                                        if let Some(serde_json::Value::String(text)) =
                                            item_obj.remove("text")
                                        {
                                            texts.push(text);
                                        }
                                    }
                                }
                                texts.join("\n")
                            }
                            other => serde_json::to_string(&other).unwrap_or_default(),
                        };
                        parts.push(CanonicalPart::ToolResult {
                            tool_call_id: tool_use_id,
                            content: content_str,
                        });
                    }
                    "thinking" => {
                        let thinking = match obj.remove("thinking") {
                            Some(serde_json::Value::String(s)) => s,
                            _ => String::new(),
                        };
                        parts.push(CanonicalPart::ReasoningText(thinking));
                    }
                    _ => {}
                }
            }
            Ok(parts)
        }
        _ => Ok(Vec::new()),
    }
}

/// Decode Anthropic `tool_choice` JSON to canonical form.
fn decode_tool_choice(v: &serde_json::Value) -> CanonicalToolChoice {
    match v.get("type").and_then(|t| t.as_str()) {
        Some("none") => CanonicalToolChoice::None,
        Some("any") => CanonicalToolChoice::Required,
        Some("tool") => {
            let name = v
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            CanonicalToolChoice::Specific(name)
        }
        _ => CanonicalToolChoice::Auto,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_owned_basic_request() {
        let req = AnthropicRequest {
            model: "claude-sonnet-4-5".to_string(),
            max_tokens: 256,
            system: Some(serde_json::json!("You are helpful")),
            messages: vec![crate::protocol::anthropic::AnthropicMessage {
                role: "user".to_string(),
                content: serde_json::json!("Hello"),
            }],
            tools: Some(vec![crate::protocol::anthropic::AnthropicTool {
                name: "get_weather".to_string(),
                description: Some("Get weather by city".to_string()),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                }),
            }]),
            tool_choice: Some(serde_json::json!({"type": "auto"})),
            stream: Some(true),
            temperature: Some(0.3),
            top_p: Some(0.9),
            stop_sequences: Some(vec!["stop".to_string()]),
            extra: serde_json::Map::new(),
        };

        let canonical = decode_anthropic_request_owned(req, uuid::Uuid::from_u128(1)).unwrap();
        assert_eq!(canonical.model, "claude-sonnet-4-5");
        assert!(canonical.stream);
        assert_eq!(canonical.system_prompt.as_deref(), Some("You are helpful"));
        assert_eq!(canonical.messages.len(), 1);
        assert_eq!(canonical.tools.len(), 1);
        assert_eq!(canonical.tools[0].function.name, "get_weather");
        assert_eq!(canonical.generation.max_tokens, Some(256));
        assert_eq!(canonical.generation.temperature, Some(0.3));
        assert_eq!(canonical.generation.top_p, Some(0.9));
        assert_eq!(canonical.generation.stop, Some(vec!["stop".to_string()]));
    }

    #[test]
    fn test_decode_user_tool_result_maps_to_tool_role() {
        let req = AnthropicRequest {
            model: "claude-sonnet-4-5".to_string(),
            max_tokens: 128,
            system: None,
            messages: vec![crate::protocol::anthropic::AnthropicMessage {
                role: "user".to_string(),
                content: serde_json::json!([{
                    "type":"tool_result",
                    "tool_use_id":"call_1",
                    "content":"{\"ok\":true}"
                }]),
            }],
            tools: None,
            tool_choice: None,
            stream: None,
            temperature: None,
            top_p: None,
            stop_sequences: None,
            extra: serde_json::Map::new(),
        };

        let canonical = decode_anthropic_request(&req, uuid::Uuid::from_u128(1)).unwrap();
        assert_eq!(canonical.messages.len(), 1);
        assert_eq!(canonical.messages[0].role, CanonicalRole::Tool);
        assert!(matches!(
            canonical.messages[0].parts.first(),
            Some(CanonicalPart::ToolResult { tool_call_id, .. }) if tool_call_id == "call_1"
        ));
    }
}
