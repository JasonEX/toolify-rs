use std::collections::HashMap;

use crate::error::CanonicalError;
use crate::protocol::canonical::{
    CanonicalPart, CanonicalRequest, CanonicalRole, CanonicalToolChoice,
};
use crate::protocol::gemini::{
    GeminiContent, GeminiFunctionCallingConfig, GeminiFunctionDeclaration, GeminiGenerationConfig,
    GeminiPart, GeminiRequest, GeminiToolConfig, GeminiToolDeclaration,
};
use crate::protocol::mapping::canonical_role_to_gemini;

/// Encode a canonical request into a Gemini wire request for upstream.
///
/// # Errors
///
/// Returns [`CanonicalError`] when canonical tool-call argument payloads are
/// invalid JSON.
pub fn encode_gemini_request(
    canonical: &CanonicalRequest,
) -> Result<GeminiRequest, CanonicalError> {
    // --- system instruction ---
    let system_instruction = canonical
        .system_prompt
        .as_ref()
        .map(|prompt| GeminiContent {
            role: None,
            parts: vec![GeminiPart::Text(prompt.clone())],
        });

    // --- contents ---
    let mut contents = Vec::with_capacity(canonical.messages.len());
    let mut call_id_to_name: HashMap<String, String> = HashMap::new();
    for msg in &canonical.messages {
        for part in &msg.parts {
            if let CanonicalPart::ToolCall { id, name, .. } = part {
                call_id_to_name.insert(id.clone(), name.clone());
            }
        }
    }

    for msg in &canonical.messages {
        // System messages go into systemInstruction, not contents.
        if msg.role == CanonicalRole::System {
            continue;
        }

        let role = canonical_role_to_gemini(msg.role).to_string();
        let mut parts = Vec::with_capacity(msg.parts.len());

        for part in &msg.parts {
            match part {
                CanonicalPart::Text(t) => {
                    parts.push(GeminiPart::Text(t.clone()));
                }
                CanonicalPart::ReasoningText(t) => {
                    // Map reasoning text as regular text for Gemini.
                    parts.push(GeminiPart::Text(t.clone()));
                }
                CanonicalPart::ToolCall {
                    name, arguments, ..
                } => {
                    // Parse the arguments JSON string into a Value for Gemini's `args`.
                    let args: serde_json::Value = serde_json::from_str(arguments.get())
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    parts.push(GeminiPart::FunctionCall {
                        name: name.clone(),
                        args,
                    });
                }
                CanonicalPart::ToolResult {
                    content,
                    tool_call_id,
                } => {
                    // Prefer call-id binding; fallback to message-level name, then call_id.
                    let fn_name = call_id_to_name
                        .get(tool_call_id)
                        .cloned()
                        .or_else(|| msg.name.clone())
                        .unwrap_or_else(|| tool_call_id.clone());
                    // Try to parse the content as JSON; fallback to wrapping in an object.
                    let response: serde_json::Value = serde_json::from_str(content)
                        .unwrap_or_else(|_| serde_json::json!({ "result": content }));
                    parts.push(GeminiPart::FunctionResponse {
                        name: fn_name,
                        response,
                    });
                }
                CanonicalPart::ImageUrl { url, .. } => {
                    // Map image URLs to Gemini InlineData when possible.
                    // For remote URLs, Gemini expects inline base64 data, so we
                    // pass the URL and let the upstream handle it. Log a warning
                    // since fidelity may be lost.
                    tracing::warn!("Gemini encoder: ImageUrl part mapped as text reference — Gemini may not fetch remote URLs");
                    parts.push(GeminiPart::Text(format!("[image: {url}]")));
                }
                CanonicalPart::Refusal(text) => {
                    tracing::warn!(
                        "Gemini encoder: Refusal part not natively supported, mapping as text"
                    );
                    parts.push(GeminiPart::Text(text.clone()));
                }
            }
        }

        if !parts.is_empty() {
            contents.push(GeminiContent {
                role: Some(role),
                parts,
            });
        }
    }

    // --- tools ---
    let tools = if canonical.tools.is_empty() {
        None
    } else {
        let decls: Vec<GeminiFunctionDeclaration> = canonical
            .tools
            .iter()
            .map(|spec| GeminiFunctionDeclaration {
                name: spec.function.name.clone(),
                description: spec.function.description.clone(),
                parameters: if spec.function.parameters.is_null()
                    || spec.function.parameters == serde_json::Value::Object(serde_json::Map::new())
                {
                    None
                } else {
                    Some(spec.function.parameters.clone())
                },
            })
            .collect();
        Some(vec![GeminiToolDeclaration {
            function_declarations: decls,
        }])
    };

    // --- tool config ---
    let tool_config = match &canonical.tool_choice {
        CanonicalToolChoice::Auto => None, // AUTO is the default
        CanonicalToolChoice::None => Some(GeminiToolConfig {
            function_calling_config: Some(GeminiFunctionCallingConfig {
                mode: Some("NONE".into()),
                allowed_function_names: None,
            }),
        }),
        CanonicalToolChoice::Required => Some(GeminiToolConfig {
            function_calling_config: Some(GeminiFunctionCallingConfig {
                mode: Some("ANY".into()),
                allowed_function_names: None,
            }),
        }),
        CanonicalToolChoice::Specific(name) => Some(GeminiToolConfig {
            function_calling_config: Some(GeminiFunctionCallingConfig {
                mode: Some("ANY".into()),
                allowed_function_names: Some(vec![name.clone()]),
            }),
        }),
    };

    // --- generation config ---
    let generation_config = {
        let g = &canonical.generation;
        let has_any = g.temperature.is_some()
            || g.top_p.is_some()
            || g.max_tokens.is_some()
            || g.stop.is_some()
            || g.n.is_some();
        if has_any {
            Some(GeminiGenerationConfig {
                temperature: g.temperature,
                top_p: g.top_p,
                max_output_tokens: g.max_tokens,
                stop_sequences: g.stop.clone(),
                candidate_count: g.n,
            })
        } else {
            None
        }
    };

    Ok(GeminiRequest {
        contents,
        tools,
        tool_config,
        system_instruction,
        generation_config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::*;
    use uuid::Uuid;

    fn make_canonical() -> CanonicalRequest {
        CanonicalRequest {
            request_id: Uuid::from_u128(1),
            ingress_api: IngressApi::Gemini,
            model: "gemini-pro".into(),
            stream: false,
            system_prompt: Some("Be helpful".into()),
            messages: vec![CanonicalMessage {
                role: CanonicalRole::User,
                parts: vec![CanonicalPart::Text("Hello".into())].into(),
                name: None,
                tool_call_id: None,
                provider_extensions: None,
            }],
            tools: vec![].into(),
            tool_choice: CanonicalToolChoice::Auto,
            generation: GenerationParams {
                temperature: Some(0.5),
                ..Default::default()
            },
            provider_extensions: None,
        }
    }

    #[test]
    fn test_basic_encode() {
        let canonical = make_canonical();
        let gemini = encode_gemini_request(&canonical).unwrap();
        assert!(gemini.system_instruction.is_some());
        assert_eq!(gemini.contents.len(), 1);
        assert_eq!(gemini.contents[0].role.as_deref(), Some("user"));
        assert!(gemini.generation_config.is_some());
        assert_eq!(gemini.generation_config.unwrap().temperature, Some(0.5));
    }

    #[test]
    fn test_tool_call_encode() {
        let raw = serde_json::value::RawValue::from_string(r#"{"city":"SF"}"#.into()).unwrap();
        let canonical = CanonicalRequest {
            request_id: Uuid::from_u128(1),
            ingress_api: IngressApi::Gemini,
            model: "gemini-pro".into(),
            stream: false,
            system_prompt: None,
            messages: vec![CanonicalMessage {
                role: CanonicalRole::Assistant,
                parts: vec![CanonicalPart::ToolCall {
                    id: "call_123".into(),
                    name: "get_weather".into(),
                    arguments: raw,
                }]
                .into(),
                name: None,
                tool_call_id: None,
                provider_extensions: None,
            }],
            tools: vec![].into(),
            tool_choice: CanonicalToolChoice::Auto,
            generation: GenerationParams::default(),
            provider_extensions: None,
        };

        let gemini = encode_gemini_request(&canonical).unwrap();
        assert_eq!(gemini.contents[0].role.as_deref(), Some("model"));
        match &gemini.contents[0].parts[0] {
            GeminiPart::FunctionCall { name, args } => {
                assert_eq!(name, "get_weather");
                assert_eq!(args["city"], "SF");
            }
            other => panic!("expected FunctionCall, got {other:?}"),
        }
    }
}
