use crate::error::CanonicalError;
use crate::protocol::anthropic::{AnthropicMessage, AnthropicRequest, AnthropicTool};
use crate::protocol::canonical::{
    provider_extensions_to_map, CanonicalPart, CanonicalRequest, CanonicalRole, CanonicalToolChoice,
};
use crate::protocol::mapping::canonical_role_to_anthropic;

/// Encode a canonical request into the Anthropic Messages API wire format.
///
/// # Errors
///
/// Returns [`CanonicalError`] when a canonical part cannot be encoded.
pub fn encode_anthropic_request(
    canonical: &CanonicalRequest,
) -> Result<AnthropicRequest, CanonicalError> {
    // --- system ---
    let system = canonical
        .system_prompt
        .as_ref()
        .map(|s| serde_json::Value::String(s.clone()));

    // --- messages ---
    let mut messages = Vec::new();
    for msg in &canonical.messages {
        // System messages should not appear in Anthropic messages array.
        if msg.role == CanonicalRole::System {
            continue;
        }

        let role = canonical_role_to_anthropic(msg.role).to_string();
        let content = encode_parts(msg.role, &msg.parts);

        messages.push(AnthropicMessage { role, content });
    }

    // --- tools ---
    let tools = if canonical.tools.is_empty() {
        None
    } else {
        Some(
            canonical
                .tools
                .iter()
                .map(|t| AnthropicTool {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    input_schema: t.function.parameters.clone(),
                })
                .collect(),
        )
    };

    // --- tool_choice ---
    let tool_choice = encode_tool_choice(&canonical.tool_choice, &canonical.tools);

    // --- max_tokens (required for Anthropic) ---
    let max_tokens = canonical.generation.max_tokens.unwrap_or(4096);

    // --- stream ---
    let stream = if canonical.stream { Some(true) } else { None };

    Ok(AnthropicRequest {
        model: canonical.model.clone(),
        max_tokens,
        system,
        messages,
        tools,
        tool_choice,
        stream,
        temperature: canonical.generation.temperature,
        top_p: canonical.generation.top_p,
        stop_sequences: canonical.generation.stop.clone(),
        extra: provider_extensions_to_map(&canonical.provider_extensions),
    })
}

/// Encode canonical parts into an Anthropic content JSON value (always an array).
fn encode_parts(role: CanonicalRole, parts: &[CanonicalPart]) -> serde_json::Value {
    let mut blocks = Vec::with_capacity(parts.len());
    // Anthropic requires `tool_result` blocks to come first in a user message.
    if matches!(role, CanonicalRole::User | CanonicalRole::Tool) {
        for part in parts {
            if matches!(part, CanonicalPart::ToolResult { .. }) {
                encode_part(part, &mut blocks);
            }
        }
        for part in parts {
            if !matches!(part, CanonicalPart::ToolResult { .. }) {
                encode_part(part, &mut blocks);
            }
        }
    } else {
        for part in parts {
            encode_part(part, &mut blocks);
        }
    }
    serde_json::Value::Array(blocks)
}

fn encode_part(part: &CanonicalPart, blocks: &mut Vec<serde_json::Value>) {
    match part {
        CanonicalPart::Text(text) | CanonicalPart::Refusal(text) => {
            blocks.push(serde_json::json!({
                "type": "text",
                "text": text,
            }));
        }
        CanonicalPart::ReasoningText(text) => {
            blocks.push(serde_json::json!({
                "type": "thinking",
                "thinking": text,
            }));
        }
        CanonicalPart::ToolCall {
            id,
            name,
            arguments,
        } => {
            let input: serde_json::Value =
                serde_json::from_str(arguments.get()).unwrap_or(serde_json::json!({}));
            blocks.push(serde_json::json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }));
        }
        CanonicalPart::ToolResult {
            tool_call_id,
            content,
        } => {
            blocks.push(serde_json::json!({
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content,
            }));
        }
        CanonicalPart::ImageUrl { url, .. } => {
            blocks.push(serde_json::json!({
                "type": "image",
                "source": {
                    "type": "url",
                    "url": url,
                },
            }));
        }
    }
}

/// Encode canonical tool choice to Anthropic `tool_choice` JSON.
fn encode_tool_choice(
    choice: &CanonicalToolChoice,
    tools: &[crate::protocol::canonical::CanonicalToolSpec],
) -> Option<serde_json::Value> {
    if tools.is_empty() {
        return None;
    }
    Some(match choice {
        CanonicalToolChoice::Auto => serde_json::json!({"type": "auto"}),
        CanonicalToolChoice::None => serde_json::json!({"type": "none"}),
        CanonicalToolChoice::Required => serde_json::json!({"type": "any"}),
        CanonicalToolChoice::Specific(name) => {
            serde_json::json!({"type": "tool", "name": name})
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::{
        CanonicalMessage, CanonicalPart, CanonicalRequest, CanonicalToolFunction,
        CanonicalToolSpec, GenerationParams, IngressApi,
    };

    #[test]
    fn test_user_tool_result_is_encoded_before_text() {
        let req = CanonicalRequest {
            request_id: uuid::Uuid::from_u128(1),
            ingress_api: IngressApi::Anthropic,
            model: "claude-sonnet-4-5".into(),
            stream: false,
            system_prompt: None,
            messages: vec![CanonicalMessage {
                role: CanonicalRole::User,
                parts: vec![
                    CanonicalPart::Text("follow-up text".into()),
                    CanonicalPart::ToolResult {
                        tool_call_id: "call_1".into(),
                        content: "{\"ok\":true}".into(),
                    },
                ]
                .into(),
                name: None,
                tool_call_id: None,
                provider_extensions: None,
            }],
            tools: vec![CanonicalToolSpec {
                function: CanonicalToolFunction {
                    name: "get_weather".into(),
                    description: None,
                    parameters: serde_json::json!({"type":"object"}),
                },
            }]
            .into(),
            tool_choice: CanonicalToolChoice::Auto,
            generation: GenerationParams::default(),
            provider_extensions: None,
        };

        let wire = encode_anthropic_request(&req).unwrap();
        let blocks = wire.messages[0].content.as_array().unwrap();
        assert_eq!(
            blocks[0].get("type").and_then(serde_json::Value::as_str),
            Some("tool_result")
        );
        assert_eq!(
            blocks[1].get("type").and_then(serde_json::Value::as_str),
            Some("text")
        );
    }
}
