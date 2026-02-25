use std::sync::Arc;

use axum::http::HeaderMap;
use axum::response::Response;

use crate::error::into_axum_response;
use crate::protocol::canonical::IngressApi;
#[cfg(test)]
use crate::protocol::gemini::GeminiRequest;

pub(crate) mod auto_fallback;
pub(crate) mod channel_b;
pub(crate) mod fc;
pub(crate) mod flow;
pub(crate) mod io;
pub(crate) mod parse;
pub(crate) mod spec;

#[cfg(test)]
use self::fc::apply_fc_inject_gemini_wire;
use self::flow::handler_inner;
#[cfg(test)]
use crate::config::FeaturesConfig;
#[cfg(test)]
use crate::protocol::gemini::{GeminiContent, GeminiPart, GeminiToolConfig, GeminiToolDeclaration};
use crate::state::AppState;

const INGRESS: IngressApi = IngressApi::Gemini;

pub async fn handler_from_action(
    state: Arc<AppState>,
    model_action: &str,
    headers: HeaderMap,
    body: bytes::Bytes,
) -> Response {
    match handler_inner(state, model_action, headers, body).await {
        Ok(response) => response,
        Err(err) => into_axum_response(&err, INGRESS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::gemini::GeminiFunctionCallingConfig;

    #[test]
    fn test_apply_fc_inject_gemini_wire_transforms_messages() {
        let mut req = GeminiRequest {
            contents: vec![
                GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![
                        GeminiPart::Text("preface".to_string()),
                        GeminiPart::FunctionCall {
                            name: "get_weather".to_string(),
                            args: serde_json::json!({"city":"SF"}),
                        },
                    ],
                },
                GeminiContent {
                    role: Some("function".to_string()),
                    parts: vec![GeminiPart::FunctionResponse {
                        name: "get_weather".to_string(),
                        response: serde_json::json!({"temp":72}),
                    }],
                },
            ],
            tools: Some(vec![GeminiToolDeclaration {
                function_declarations: vec![crate::protocol::gemini::GeminiFunctionDeclaration {
                    name: "get_weather".to_string(),
                    description: Some("weather".to_string()),
                    parameters: Some(serde_json::json!({
                        "type":"object",
                        "properties":{"city":{"type":"string"}}
                    })),
                }],
            }]),
            tool_config: Some(GeminiToolConfig {
                function_calling_config: Some(GeminiFunctionCallingConfig {
                    mode: Some("ANY".to_string()),
                    allowed_function_names: None,
                }),
            }),
            system_instruction: Some(GeminiContent {
                role: None,
                parts: vec![GeminiPart::Text("sys".to_string())],
            }),
            generation_config: None,
        };

        let saved_tools =
            apply_fc_inject_gemini_wire(&mut req, &FeaturesConfig::default()).unwrap();
        assert_eq!(saved_tools.len(), 1);
        assert!(req.tools.is_none());
        assert!(req.tool_config.is_none());

        let system_text = req
            .system_instruction
            .as_ref()
            .and_then(|c| c.parts.first())
            .and_then(|p| match p {
                GeminiPart::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .unwrap();
        assert!(system_text.contains("sys"));
        assert!(system_text.contains(crate::fc::prompt::get_trigger_signal()));

        let model_text = req.contents[0]
            .parts
            .first()
            .and_then(|p| match p {
                GeminiPart::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .unwrap();
        assert!(model_text.contains(crate::fc::prompt::get_trigger_signal()));
        assert!(model_text.contains("<function_call>"));

        assert_eq!(req.contents[1].role.as_deref(), Some("user"));
        let user_text = req.contents[1]
            .parts
            .first()
            .and_then(|p| match p {
                GeminiPart::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .unwrap();
        assert!(user_text.contains("Tool execution result:"));
        assert!(user_text.contains("get_weather"));
    }

    #[test]
    fn test_apply_fc_inject_gemini_wire_skips_when_tool_choice_none() {
        let mut req = GeminiRequest {
            contents: vec![GeminiContent {
                role: Some("user".to_string()),
                parts: vec![GeminiPart::Text("hi".to_string())],
            }],
            tools: Some(vec![GeminiToolDeclaration {
                function_declarations: vec![crate::protocol::gemini::GeminiFunctionDeclaration {
                    name: "get_weather".to_string(),
                    description: Some("weather".to_string()),
                    parameters: Some(serde_json::json!({
                        "type":"object",
                        "properties":{"city":{"type":"string"}}
                    })),
                }],
            }]),
            tool_config: Some(GeminiToolConfig {
                function_calling_config: Some(GeminiFunctionCallingConfig {
                    mode: Some("NONE".to_string()),
                    allowed_function_names: None,
                }),
            }),
            system_instruction: Some(GeminiContent {
                role: None,
                parts: vec![GeminiPart::Text("sys".to_string())],
            }),
            generation_config: None,
        };

        let saved_tools =
            apply_fc_inject_gemini_wire(&mut req, &FeaturesConfig::default()).unwrap();
        assert!(saved_tools.is_empty());
        assert!(req.tools.is_some());
        assert!(req.tool_config.is_some());
        let system_text = req
            .system_instruction
            .as_ref()
            .and_then(|content| content.parts.first())
            .and_then(|part| match part {
                GeminiPart::Text(text) => Some(text.as_str()),
                _ => None,
            });
        assert_eq!(system_text, Some("sys"));
        assert_eq!(req.contents.len(), 1);
    }
}
