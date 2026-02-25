use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;

use crate::error::into_axum_response;
#[cfg(test)]
use crate::protocol::anthropic::AnthropicMessage;
#[cfg(test)]
use crate::protocol::anthropic::AnthropicRequest;
use crate::protocol::canonical::IngressApi;

pub(crate) mod auto_fallback;
pub(crate) mod channel_b;
pub(crate) mod fc;
pub(crate) mod flow;
pub(crate) mod io;
pub(crate) mod parse;
pub(crate) mod spec;

#[cfg(test)]
use self::fc::apply_fc_inject_anthropic_wire;
use self::flow::handler_inner;
#[cfg(test)]
use crate::config::FeaturesConfig;
#[cfg(test)]
use crate::protocol::anthropic::AnthropicTool;
use crate::state::AppState;

const INGRESS: IngressApi = IngressApi::Anthropic;

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
    fn test_apply_fc_inject_anthropic_wire_transforms_messages() {
        let mut req = AnthropicRequest {
            model: "claude-sonnet-4-5".to_string(),
            max_tokens: 256,
            system: Some(serde_json::Value::String("sys".to_string())),
            messages: vec![
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: serde_json::json!([
                        {"type":"text","text":"preface"},
                        {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"SF"}}
                    ]),
                },
                AnthropicMessage {
                    role: "user".to_string(),
                    content: serde_json::json!([
                        {"type":"tool_result","tool_use_id":"toolu_1","content":"{\"temp\":72}"}
                    ]),
                },
            ],
            tools: Some(vec![AnthropicTool {
                name: "get_weather".to_string(),
                description: Some("weather".to_string()),
                input_schema: serde_json::json!({
                    "type":"object",
                    "properties":{"city":{"type":"string"}},
                    "required":["city"]
                }),
            }]),
            tool_choice: Some(serde_json::json!({"type":"auto"})),
            stream: Some(false),
            temperature: None,
            top_p: None,
            stop_sequences: None,
            extra: serde_json::Map::new(),
        };

        let saved_tools =
            apply_fc_inject_anthropic_wire(&mut req, &FeaturesConfig::default()).unwrap();
        assert_eq!(saved_tools.len(), 1);
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());

        let system = req
            .system
            .as_ref()
            .and_then(serde_json::Value::as_str)
            .unwrap();
        assert!(system.contains("sys"));
        assert!(system.contains(crate::fc::prompt::get_trigger_signal()));

        let assistant_text = req.messages[0]
            .content
            .as_array()
            .and_then(|items| items.first())
            .and_then(|v| v.get("text"))
            .and_then(serde_json::Value::as_str)
            .unwrap();
        assert!(assistant_text.contains(crate::fc::prompt::get_trigger_signal()));
        assert!(assistant_text.contains("<function_call>"));

        let user_text = req.messages[1].content.as_str().unwrap();
        assert!(user_text.contains("Tool execution result:"));
        assert!(user_text.contains("get_weather"));
    }

    #[test]
    fn test_apply_fc_inject_anthropic_wire_skips_when_tool_choice_none() {
        let mut req = AnthropicRequest {
            model: "claude-sonnet-4-5".to_string(),
            max_tokens: 256,
            system: Some(serde_json::Value::String("sys".to_string())),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: serde_json::Value::String("hi".to_string()),
            }],
            tools: Some(vec![AnthropicTool {
                name: "get_weather".to_string(),
                description: Some("weather".to_string()),
                input_schema: serde_json::json!({
                    "type":"object",
                    "properties":{"city":{"type":"string"}}
                }),
            }]),
            tool_choice: Some(serde_json::json!({"type":"none"})),
            stream: Some(false),
            temperature: None,
            top_p: None,
            stop_sequences: None,
            extra: serde_json::Map::new(),
        };

        let saved_tools =
            apply_fc_inject_anthropic_wire(&mut req, &FeaturesConfig::default()).unwrap();
        assert!(saved_tools.is_empty());
        assert_eq!(
            req.system,
            Some(serde_json::Value::String("sys".to_string()))
        );
        assert_eq!(req.tool_choice, Some(serde_json::json!({"type":"none"})));
        assert!(req.tools.is_some());
        assert_eq!(req.messages.len(), 1);
    }
}
