use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

use crate::error::CanonicalError;
use crate::protocol::canonical::{
    CanonicalMessage, CanonicalPart, CanonicalRequest, CanonicalRole, CanonicalToolChoice,
    CanonicalToolFunction, CanonicalToolSpec, GenerationParams, IngressApi,
};
use crate::protocol::gemini::{GeminiPart, GeminiRequest};
use crate::protocol::mapping::gemini_role_to_canonical;
use crate::util::raw_value_from_string;

/// Decode a Gemini wire request into the canonical IR.
///
/// `model` comes from the URL path parameter, not the request body.
///
/// # Errors
///
/// Returns [`CanonicalError`] when request content cannot be decoded into
/// canonical message parts.
pub fn decode_gemini_request(
    request: &GeminiRequest,
    model: &str,
    request_id: Uuid,
) -> Result<CanonicalRequest, CanonicalError> {
    // --- system prompt ---
    let system_prompt = request
        .system_instruction
        .as_ref()
        .and_then(|si| si.parts.first())
        .and_then(|p| match p {
            GeminiPart::Text(t) => Some(t.clone()),
            _ => None,
        });

    // --- messages ---
    let mut messages = Vec::with_capacity(request.contents.len());
    let mut call_counter: usize = 0;
    let mut pending_calls_by_name: HashMap<String, VecDeque<String>> = HashMap::new();

    for content in &request.contents {
        let role = content
            .role
            .as_deref()
            .map_or(CanonicalRole::User, gemini_role_to_canonical);

        let mut parts = Vec::with_capacity(content.parts.len());
        for part in &content.parts {
            match part {
                GeminiPart::Text(t) => {
                    parts.push(CanonicalPart::Text(t.clone()));
                }
                GeminiPart::FunctionCall { name, args } => {
                    let call_id = format!("call_{call_counter}");
                    call_counter += 1;
                    pending_calls_by_name
                        .entry(name.clone())
                        .or_default()
                        .push_back(call_id.clone());
                    let args_str = serde_json::to_string(args).unwrap_or_else(|_| "{}".into());
                    let raw = raw_value_from_string(args_str, "Gemini function_call")?;
                    parts.push(CanonicalPart::ToolCall {
                        id: call_id,
                        name: name.clone(),
                        arguments: raw,
                    });
                }
                GeminiPart::FunctionResponse { name, response } => {
                    let call_id = pending_calls_by_name
                        .get_mut(name)
                        .and_then(VecDeque::pop_front)
                        .unwrap_or_else(|| {
                            let id = format!("call_{call_counter}");
                            call_counter += 1;
                            id
                        });
                    let content_str =
                        serde_json::to_string(response).unwrap_or_else(|_| "{}".into());
                    parts.push(CanonicalPart::ToolResult {
                        tool_call_id: call_id,
                        content: content_str,
                    });
                    // Attach the function name to the message via `name` field below.
                    // We push the name on the message level.
                    // (handled after loop via the first FunctionResponse name)
                }
                GeminiPart::InlineData { .. } => {
                    // Inline data not mapped to canonical yet; skip.
                }
            }
        }

        // For function-role messages, try to extract the name from the first FunctionResponse.
        let name = if role == CanonicalRole::Tool {
            content.parts.iter().find_map(|p| match p {
                GeminiPart::FunctionResponse { name, .. } => Some(name.clone()),
                _ => None,
            })
        } else {
            None
        };

        messages.push(CanonicalMessage {
            role,
            parts: parts.into(),
            name,
            tool_call_id: None,
            provider_extensions: None,
        });
    }

    // --- tools ---
    let tools = request
        .tools
        .as_ref()
        .map(|tool_decls| {
            tool_decls
                .iter()
                .flat_map(|td| {
                    td.function_declarations.iter().map(|fd| CanonicalToolSpec {
                        function: CanonicalToolFunction {
                            name: fd.name.clone(),
                            description: fd.description.clone(),
                            parameters: fd
                                .parameters
                                .clone()
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                        },
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let tool_choice = decode_gemini_tool_choice_config(
        request
            .tool_config
            .as_ref()
            .and_then(|tc| tc.function_calling_config.as_ref()),
    );

    // --- generation config ---
    let generation = request
        .generation_config
        .as_ref()
        .map(|gc| GenerationParams {
            temperature: gc.temperature,
            max_tokens: gc.max_output_tokens,
            top_p: gc.top_p,
            stop: gc.stop_sequences.clone(),
            n: gc.candidate_count,
            ..Default::default()
        })
        .unwrap_or_default();

    // --- stream detection ---
    // Streaming is determined by the endpoint, not the body. Default to false.
    let stream = false;

    Ok(build_gemini_request(
        request_id,
        model.to_string(),
        stream,
        system_prompt,
        messages,
        tools,
        tool_choice,
        generation,
    ))
}

/// Decode a Gemini wire request into the canonical IR by consuming ownership.
///
/// # Errors
///
/// Returns [`CanonicalError`] when request content cannot be decoded into
/// canonical message parts.
pub fn decode_gemini_request_owned(
    request: GeminiRequest,
    model: String,
    request_id: Uuid,
) -> Result<CanonicalRequest, CanonicalError> {
    let GeminiRequest {
        contents,
        tools,
        tool_config,
        system_instruction,
        generation_config,
    } = request;

    let system_prompt = system_instruction
        .and_then(|si| si.parts.into_iter().next())
        .and_then(|p| match p {
            GeminiPart::Text(t) => Some(t),
            _ => None,
        });

    let mut messages = Vec::with_capacity(contents.len());
    let mut call_counter: usize = 0;
    let mut pending_calls_by_name: HashMap<String, VecDeque<String>> = HashMap::new();

    for content in contents {
        let role = content
            .role
            .as_deref()
            .map_or(CanonicalRole::User, gemini_role_to_canonical);

        let mut parts = Vec::with_capacity(content.parts.len());
        let mut first_function_response_name: Option<String> = None;
        for part in content.parts {
            match part {
                GeminiPart::Text(t) => {
                    parts.push(CanonicalPart::Text(t));
                }
                GeminiPart::FunctionCall { name, args } => {
                    let call_id = format!("call_{call_counter}");
                    call_counter += 1;
                    pending_calls_by_name
                        .entry(name.clone())
                        .or_default()
                        .push_back(call_id.clone());
                    let args_str = serde_json::to_string(&args).unwrap_or_else(|_| "{}".into());
                    let raw = raw_value_from_string(args_str, "Gemini function_call")?;
                    parts.push(CanonicalPart::ToolCall {
                        id: call_id,
                        name,
                        arguments: raw,
                    });
                }
                GeminiPart::FunctionResponse { name, response } => {
                    let call_id = pending_calls_by_name
                        .get_mut(name.as_str())
                        .and_then(VecDeque::pop_front)
                        .unwrap_or_else(|| {
                            let id = format!("call_{call_counter}");
                            call_counter += 1;
                            id
                        });
                    if first_function_response_name.is_none() {
                        first_function_response_name = Some(name);
                    }
                    let content_str =
                        serde_json::to_string(&response).unwrap_or_else(|_| "{}".into());
                    parts.push(CanonicalPart::ToolResult {
                        tool_call_id: call_id,
                        content: content_str,
                    });
                }
                GeminiPart::InlineData { .. } => {}
            }
        }

        let name = if role == CanonicalRole::Tool {
            first_function_response_name
        } else {
            None
        };

        messages.push(CanonicalMessage {
            role,
            parts: parts.into(),
            name,
            tool_call_id: None,
            provider_extensions: None,
        });
    }

    let tools = tools
        .map(|tool_decls| {
            tool_decls
                .into_iter()
                .flat_map(|td| {
                    td.function_declarations
                        .into_iter()
                        .map(|fd| CanonicalToolSpec {
                            function: CanonicalToolFunction {
                                name: fd.name,
                                description: fd.description,
                                parameters: fd
                                    .parameters
                                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                            },
                        })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let tool_choice = decode_gemini_tool_choice_config(
        tool_config
            .as_ref()
            .and_then(|tc| tc.function_calling_config.as_ref()),
    );

    let generation = generation_config
        .map(|gc| GenerationParams {
            temperature: gc.temperature,
            max_tokens: gc.max_output_tokens,
            top_p: gc.top_p,
            stop: gc.stop_sequences,
            n: gc.candidate_count,
            ..Default::default()
        })
        .unwrap_or_default();

    Ok(build_gemini_request(
        request_id,
        model,
        false,
        system_prompt,
        messages,
        tools,
        tool_choice,
        generation,
    ))
}

fn build_gemini_request(
    request_id: Uuid,
    model: String,
    stream: bool,
    system_prompt: Option<String>,
    messages: Vec<CanonicalMessage>,
    tools: Vec<CanonicalToolSpec>,
    tool_choice: CanonicalToolChoice,
    generation: GenerationParams,
) -> CanonicalRequest {
    CanonicalRequest {
        request_id,
        ingress_api: IngressApi::Gemini,
        model,
        stream,
        system_prompt,
        messages,
        tools: tools.into(),
        tool_choice,
        generation,
        provider_extensions: None,
    }
}

fn decode_gemini_tool_choice_config(
    cfg: Option<&crate::protocol::gemini::GeminiFunctionCallingConfig>,
) -> CanonicalToolChoice {
    let Some(cfg) = cfg else {
        return CanonicalToolChoice::Auto;
    };
    match cfg.mode.as_deref().unwrap_or("AUTO") {
        "NONE" => CanonicalToolChoice::None,
        "ANY" => {
            if let Some(allowed) = cfg.allowed_function_names.as_ref() {
                if allowed.len() == 1 {
                    return CanonicalToolChoice::Specific(allowed[0].clone());
                }
            }
            CanonicalToolChoice::Required
        }
        _ => CanonicalToolChoice::Auto,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::gemini::*;

    #[test]
    fn test_basic_decode() {
        let req = GeminiRequest {
            contents: vec![GeminiContent {
                role: Some("user".into()),
                parts: vec![GeminiPart::Text("Hello".into())],
            }],
            tools: None,
            tool_config: None,
            system_instruction: Some(GeminiContent {
                role: None,
                parts: vec![GeminiPart::Text("You are helpful".into())],
            }),
            generation_config: Some(GeminiGenerationConfig {
                temperature: Some(0.7),
                top_p: None,
                max_output_tokens: Some(1024),
                stop_sequences: None,
                candidate_count: None,
            }),
        };

        let canonical = decode_gemini_request(&req, "gemini-pro", Uuid::from_u128(1)).unwrap();
        assert_eq!(canonical.system_prompt.as_deref(), Some("You are helpful"));
        assert_eq!(canonical.model, "gemini-pro");
        assert_eq!(canonical.messages.len(), 1);
        assert_eq!(canonical.messages[0].role, CanonicalRole::User);
        assert_eq!(canonical.generation.temperature, Some(0.7));
        assert_eq!(canonical.generation.max_tokens, Some(1024));
    }

    #[test]
    fn test_function_call_decode() {
        let req = GeminiRequest {
            contents: vec![GeminiContent {
                role: Some("model".into()),
                parts: vec![GeminiPart::FunctionCall {
                    name: "get_weather".into(),
                    args: serde_json::json!({"city": "SF"}),
                }],
            }],
            tools: None,
            tool_config: None,
            system_instruction: None,
            generation_config: None,
        };

        let canonical = decode_gemini_request(&req, "gemini-pro", Uuid::from_u128(1)).unwrap();
        assert_eq!(canonical.messages[0].role, CanonicalRole::Assistant);
        match &canonical.messages[0].parts[0] {
            CanonicalPart::ToolCall { id, name, .. } => {
                assert!(id.starts_with("call_"));
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn test_tool_choice_mapping() {
        let req = GeminiRequest {
            contents: vec![GeminiContent {
                role: Some("user".into()),
                parts: vec![GeminiPart::Text("Hi".into())],
            }],
            tools: None,
            tool_config: Some(GeminiToolConfig {
                function_calling_config: Some(GeminiFunctionCallingConfig {
                    mode: Some("ANY".into()),
                    allowed_function_names: None,
                }),
            }),
            system_instruction: None,
            generation_config: None,
        };

        let canonical = decode_gemini_request(&req, "gemini-pro", Uuid::from_u128(1)).unwrap();
        assert_eq!(canonical.tool_choice, CanonicalToolChoice::Required);
    }

    #[test]
    fn test_tool_choice_any_single_allowed_maps_specific() {
        let req = GeminiRequest {
            contents: vec![GeminiContent {
                role: Some("user".into()),
                parts: vec![GeminiPart::Text("Hi".into())],
            }],
            tools: None,
            tool_config: Some(GeminiToolConfig {
                function_calling_config: Some(GeminiFunctionCallingConfig {
                    mode: Some("ANY".into()),
                    allowed_function_names: Some(vec!["get_weather".into()]),
                }),
            }),
            system_instruction: None,
            generation_config: None,
        };

        let canonical = decode_gemini_request(&req, "gemini-pro", Uuid::from_u128(1)).unwrap();
        assert_eq!(
            canonical.tool_choice,
            CanonicalToolChoice::Specific("get_weather".to_string())
        );
    }

    #[test]
    fn test_tool_choice_validated_maps_auto() {
        let req = GeminiRequest {
            contents: vec![GeminiContent {
                role: Some("user".into()),
                parts: vec![GeminiPart::Text("Hi".into())],
            }],
            tools: None,
            tool_config: Some(GeminiToolConfig {
                function_calling_config: Some(GeminiFunctionCallingConfig {
                    mode: Some("VALIDATED".into()),
                    allowed_function_names: Some(vec!["get_weather".into()]),
                }),
            }),
            system_instruction: None,
            generation_config: None,
        };

        let canonical = decode_gemini_request(&req, "gemini-pro", Uuid::from_u128(1)).unwrap();
        assert_eq!(canonical.tool_choice, CanonicalToolChoice::Auto);
    }

    #[test]
    fn test_function_call_response_id_binding() {
        let req = GeminiRequest {
            contents: vec![GeminiContent {
                role: Some("model".into()),
                parts: vec![
                    GeminiPart::FunctionCall {
                        name: "get_weather".into(),
                        args: serde_json::json!({"city": "SF"}),
                    },
                    GeminiPart::FunctionResponse {
                        name: "get_weather".into(),
                        response: serde_json::json!({"temp": 72}),
                    },
                    GeminiPart::FunctionResponse {
                        name: "get_weather".into(),
                        response: serde_json::json!({"temp": 73}),
                    },
                ],
            }],
            tools: None,
            tool_config: None,
            system_instruction: None,
            generation_config: None,
        };

        let canonical = decode_gemini_request(&req, "gemini-pro", Uuid::from_u128(1)).unwrap();
        assert_eq!(canonical.messages.len(), 1);
        assert_eq!(canonical.messages[0].parts.len(), 3);

        match &canonical.messages[0].parts[0] {
            CanonicalPart::ToolCall { id, name, .. } => {
                assert_eq!(id, "call_0");
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }

        match &canonical.messages[0].parts[1] {
            CanonicalPart::ToolResult {
                tool_call_id,
                content,
            } => {
                assert_eq!(tool_call_id, "call_0");
                assert!(content.contains("72"));
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }

        match &canonical.messages[0].parts[2] {
            CanonicalPart::ToolResult { tool_call_id, .. } => {
                assert_eq!(tool_call_id, "call_1");
            }
            other => panic!("expected fallback ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_owned_basic() {
        let req = GeminiRequest {
            contents: vec![GeminiContent {
                role: Some("user".into()),
                parts: vec![GeminiPart::Text("Hello".into())],
            }],
            tools: None,
            tool_config: None,
            system_instruction: Some(GeminiContent {
                role: None,
                parts: vec![GeminiPart::Text("You are helpful".into())],
            }),
            generation_config: Some(GeminiGenerationConfig {
                temperature: Some(0.4),
                top_p: Some(0.9),
                max_output_tokens: Some(512),
                stop_sequences: Some(vec!["stop".into()]),
                candidate_count: Some(1),
            }),
        };

        let canonical =
            decode_gemini_request_owned(req, "gemini-2.5-pro".to_string(), Uuid::from_u128(1))
                .unwrap();
        assert_eq!(canonical.model, "gemini-2.5-pro");
        assert_eq!(canonical.system_prompt.as_deref(), Some("You are helpful"));
        assert_eq!(canonical.messages.len(), 1);
        assert_eq!(canonical.generation.temperature, Some(0.4));
        assert_eq!(canonical.generation.top_p, Some(0.9));
        assert_eq!(canonical.generation.max_tokens, Some(512));
        assert_eq!(canonical.generation.stop, Some(vec!["stop".to_string()]));
        assert_eq!(canonical.generation.n, Some(1));
    }
}
