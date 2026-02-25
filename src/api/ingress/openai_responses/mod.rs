use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;

use crate::error::into_axum_response;
#[cfg(test)]
use crate::error::CanonicalError;
use crate::protocol::canonical::IngressApi;
#[cfg(test)]
use crate::protocol::openai_responses::ResponsesRequest;

pub(crate) mod auto_fallback;
pub(crate) mod channel_b;
pub(crate) mod fc;
pub(crate) mod flow;
pub(crate) mod io;
pub(crate) mod parse;
pub(crate) mod spec;

#[cfg(test)]
use self::fc::apply_fc_inject_responses_wire;
#[cfg(test)]
use self::fc::preprocess_responses_wire_input;
use self::flow::handler_inner;
#[cfg(test)]
use self::parse::parse_openai_responses_probe;
#[cfg(test)]
use crate::config::FeaturesConfig;
#[cfg(test)]
use crate::protocol::openai_responses::ResponsesTool;
use crate::state::AppState;

const INGRESS: IngressApi = IngressApi::OpenAiResponses;

pub async fn handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: bytes::Bytes,
) -> Response {
    match handler_inner(state, headers, body).await {
        Ok(response) => response,
        Err(err) => into_axum_response(&err, INGRESS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_fc_inject_responses_wire_transforms() {
        let mut req = ResponsesRequest {
            model: "m1".to_string(),
            input: serde_json::json!([
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type":"input_text","text":"hi"}]
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "{\"ok\":true}"
                }
            ]),
            instructions: Some("sys".to_string()),
            tools: Some(vec![
                ResponsesTool::Function {
                    name: "get_weather".to_string(),
                    description: Some("weather".to_string()),
                    parameters: Some(serde_json::json!({
                        "type":"object",
                        "properties":{"city":{"type":"string"}}
                    })),
                },
                ResponsesTool::WebSearch {
                    extra: serde_json::Map::new(),
                },
            ]),
            tool_choice: Some(serde_json::json!("required")),
            previous_response_id: None,
            store: None,
            stream: None,
            temperature: None,
            max_output_tokens: None,
            top_p: None,
            extra: serde_json::Map::new(),
        };

        let saved_tools =
            apply_fc_inject_responses_wire(&mut req, &FeaturesConfig::default()).unwrap();
        assert_eq!(saved_tools.len(), 1);
        assert!(req
            .instructions
            .as_ref()
            .is_some_and(|s| s.contains(crate::fc::prompt::get_trigger_signal())));

        let tools = req.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert!(matches!(tools[0], ResponsesTool::WebSearch { .. }));
        assert!(req.tool_choice.is_none());

        let output_text = req
            .input
            .as_array()
            .and_then(|arr| arr.get(1))
            .and_then(|item| item.get("content"))
            .and_then(serde_json::Value::as_array)
            .and_then(|parts| parts.first())
            .and_then(|part| part.get("text"))
            .and_then(serde_json::Value::as_str)
            .unwrap();
        assert!(output_text.contains("Tool execution result:"));
    }

    #[test]
    fn test_preprocess_responses_wire_input_unknown_item_type() {
        let input = serde_json::json!([{"type":"unknown"}]);
        let err = preprocess_responses_wire_input(input).unwrap_err();
        assert!(matches!(err, CanonicalError::InvalidRequest(_)));
    }

    #[test]
    fn test_parse_openai_responses_probe_builtin_tools_not_fc_tools() {
        let body = bytes::Bytes::from_static(
            br#"{"model":"gpt-4o","input":"hi","tools":[{"type":"web_search"}]}"#,
        );
        let probe = parse_openai_responses_probe(&body).unwrap();
        assert!(!probe.has_tools);
    }

    #[test]
    fn test_parse_openai_responses_probe_function_tools_enable_fc() {
        let body = bytes::Bytes::from_static(
            br#"{"model":"gpt-4o","input":"hi","tools":[{"type":"function","name":"x"}]}"#,
        );
        let probe = parse_openai_responses_probe(&body).unwrap();
        assert!(probe.has_tools);
    }

    #[test]
    fn test_apply_fc_inject_responses_wire_skips_when_only_builtin_tools() {
        let mut req = ResponsesRequest {
            model: "m1".to_string(),
            input: serde_json::json!([{
                "type": "message",
                "role": "user",
                "content": [{"type":"input_text","text":"hi"}]
            }]),
            instructions: Some("sys".to_string()),
            tools: Some(vec![ResponsesTool::WebSearch {
                extra: serde_json::Map::new(),
            }]),
            tool_choice: Some(serde_json::json!("required")),
            previous_response_id: None,
            store: None,
            stream: None,
            temperature: None,
            max_output_tokens: None,
            top_p: None,
            extra: serde_json::Map::new(),
        };

        let saved_tools =
            apply_fc_inject_responses_wire(&mut req, &FeaturesConfig::default()).unwrap();
        assert!(saved_tools.is_empty());
        assert_eq!(req.instructions.as_deref(), Some("sys"));
        assert_eq!(req.tool_choice, Some(serde_json::json!("required")));
        assert!(req
            .tools
            .as_ref()
            .is_some_and(|tools| matches!(tools.first(), Some(ResponsesTool::WebSearch { .. }))));
    }

    #[test]
    fn test_apply_fc_inject_responses_wire_skips_when_tool_choice_none() {
        let mut req = ResponsesRequest {
            model: "m1".to_string(),
            input: serde_json::json!([{
                "type": "message",
                "role": "user",
                "content": [{"type":"input_text","text":"hi"}]
            }]),
            instructions: Some("sys".to_string()),
            tools: Some(vec![ResponsesTool::Function {
                name: "get_weather".to_string(),
                description: Some("weather".to_string()),
                parameters: Some(serde_json::json!({
                    "type":"object",
                    "properties":{"city":{"type":"string"}}
                })),
            }]),
            tool_choice: Some(serde_json::json!("none")),
            previous_response_id: None,
            store: None,
            stream: None,
            temperature: None,
            max_output_tokens: None,
            top_p: None,
            extra: serde_json::Map::new(),
        };

        let saved_tools =
            apply_fc_inject_responses_wire(&mut req, &FeaturesConfig::default()).unwrap();
        assert!(saved_tools.is_empty());
        assert_eq!(req.instructions.as_deref(), Some("sys"));
        assert_eq!(req.tool_choice, Some(serde_json::json!("none")));
        assert!(req
            .tools
            .as_ref()
            .is_some_and(|tools| matches!(tools.first(), Some(ResponsesTool::Function { .. }))));
    }

    #[test]
    fn test_apply_fc_inject_responses_wire_skips_when_structured_output_requested() {
        let mut req = ResponsesRequest {
            model: "m1".to_string(),
            input: serde_json::json!([{
                "type": "message",
                "role": "user",
                "content": [{"type":"input_text","text":"hi"}]
            }]),
            instructions: Some("sys".to_string()),
            tools: Some(vec![ResponsesTool::Function {
                name: "get_weather".to_string(),
                description: Some("weather".to_string()),
                parameters: Some(serde_json::json!({
                    "type":"object",
                    "properties":{"city":{"type":"string"}}
                })),
            }]),
            tool_choice: Some(serde_json::json!("auto")),
            previous_response_id: None,
            store: None,
            stream: None,
            temperature: None,
            max_output_tokens: None,
            top_p: None,
            extra: serde_json::Map::new(),
        };
        req.extra.insert(
            "text".to_string(),
            serde_json::json!({"format":{"type":"json_schema","schema":{"type":"object"}}}),
        );

        let saved_tools =
            apply_fc_inject_responses_wire(&mut req, &FeaturesConfig::default()).unwrap();
        assert!(saved_tools.is_empty());
        assert_eq!(req.instructions.as_deref(), Some("sys"));
        assert_eq!(req.tool_choice, Some(serde_json::json!("auto")));
        assert!(req
            .tools
            .as_ref()
            .is_some_and(|tools| matches!(tools.first(), Some(ResponsesTool::Function { .. }))));
    }

    #[test]
    fn test_preprocess_responses_wire_input_uses_function_call_index() {
        let input = serde_json::json!([
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{\"city\":\"London\"}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "{\"ok\":true}"
            }
        ]);
        let transformed = preprocess_responses_wire_input(input).unwrap();
        let items = transformed.as_array().unwrap();
        assert_eq!(items.len(), 2);

        let assistant_text = items[0]
            .get("content")
            .and_then(serde_json::Value::as_array)
            .and_then(|parts| parts.first())
            .and_then(|part| part.get("text"))
            .and_then(serde_json::Value::as_str)
            .unwrap();
        assert!(assistant_text.contains(crate::fc::prompt::get_trigger_signal()));
        assert!(assistant_text.contains("<id>call_1</id>"));
        assert!(assistant_text.contains("<tool>get_weather</tool>"));

        let tool_result_text = items[1]
            .get("content")
            .and_then(serde_json::Value::as_array)
            .and_then(|parts| parts.first())
            .and_then(|part| part.get("text"))
            .and_then(serde_json::Value::as_str)
            .unwrap();
        assert!(tool_result_text.contains("Tool name: get_weather"));
        assert!(tool_result_text.contains("Tool arguments: {\"city\":\"London\"}"));
    }
}
