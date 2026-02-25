use std::sync::Arc;

use crate::error::into_axum_response;
use crate::protocol::canonical::IngressApi;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;

pub(crate) mod auto_fallback;
pub(crate) mod channel_b;
pub(crate) mod fc;
pub(crate) mod flow;
pub(crate) mod io;
pub(crate) mod parse;
pub(crate) mod spec;

#[cfg(test)]
use self::fc::{
    apply_fc_inject_openai_wire, build_openai_simple_inject_json_body,
    messages_inner_bounds_if_simple, try_build_openai_simple_fc_inject_body_from_raw,
};
use self::flow::handler_inner;
#[cfg(test)]
use crate::api::common::{
    find_top_level_field_value_range, rewrite_model_field_in_json_body_with_range,
};
#[cfg(test)]
use crate::routing::session::route_prompt_prefix_bytes;

use crate::state::AppState;

const INGRESS: IngressApi = IngressApi::OpenAiChat;
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
    use super::{
        apply_fc_inject_openai_wire, build_openai_simple_inject_json_body,
        find_top_level_field_value_range, messages_inner_bounds_if_simple,
        rewrite_model_field_in_json_body_with_range, route_prompt_prefix_bytes,
        try_build_openai_simple_fc_inject_body_from_raw,
    };
    use crate::config::FeaturesConfig;
    use crate::protocol::openai_chat::{
        OpenAiChatRequest, OpenAiMessage, OpenAiTool, OpenAiToolChoice, OpenAiToolChoiceFunction,
        OpenAiToolChoiceFunctionCall, OpenAiToolFunction,
    };

    #[test]
    fn test_find_top_level_field_value_range_last_match() {
        let body = br#"{"model":"a","x":{"model":"nested"},"model":{"v":1}}"#;
        let range = find_top_level_field_value_range(body, b"model")
            .expect("parse should succeed")
            .expect("model should exist");
        assert_eq!(&body[range], br#"{"v":1}"#);
    }

    #[test]
    fn test_rewrite_model_fast_path_string() {
        let body = bytes::Bytes::from_static(br#"{"model":"m1","messages":[{"role":"user"}]}"#);
        let out =
            rewrite_model_field_in_json_body_with_range(&body, "m2", "OpenAI Chat request", None)
                .unwrap();
        assert_eq!(
            out,
            br#"{"model":"m2","messages":[{"role":"user"}]}"#.as_slice()
        );
    }

    #[test]
    fn test_rewrite_model_fast_path_non_string() {
        let body = bytes::Bytes::from_static(br#"{"model":{"name":"m1"},"x":1}"#);
        let out =
            rewrite_model_field_in_json_body_with_range(&body, "m2", "OpenAI Chat request", None)
                .unwrap();
        assert_eq!(out, br#"{"model":"m2","x":1}"#.as_slice());
    }

    #[test]
    fn test_rewrite_model_fallback_insert_when_missing() {
        let body = bytes::Bytes::from_static(br#"{"messages":[{"role":"user"}]}"#);
        let out =
            rewrite_model_field_in_json_body_with_range(&body, "m2", "OpenAI Chat request", None)
                .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            json.get("model").and_then(serde_json::Value::as_str),
            Some("m2")
        );
    }

    #[test]
    fn test_rewrite_model_fast_path_utf8_no_escape() {
        let body = bytes::Bytes::from_static(br#"{"model":"m1","messages":[{"role":"user"}]}"#);
        let out = rewrite_model_field_in_json_body_with_range(
            &body,
            "模型α",
            "OpenAI Chat request",
            None,
        )
        .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            json.get("model").and_then(serde_json::Value::as_str),
            Some("模型α")
        );
    }

    #[test]
    fn test_rewrite_model_escape_path_quote_and_backslash() {
        let body = bytes::Bytes::from_static(br#"{"model":"m1","messages":[{"role":"user"}]}"#);
        let out = rewrite_model_field_in_json_body_with_range(
            &body,
            r#"m"2\beta"#,
            "OpenAI Chat request",
            None,
        )
        .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            json.get("model").and_then(serde_json::Value::as_str),
            Some(r#"m"2\beta"#)
        );
    }

    #[test]
    fn test_apply_fc_inject_openai_wire_transforms_tool_and_assistant_messages() {
        let mut req = OpenAiChatRequest {
            model: "m1".to_string(),
            messages: vec![
                OpenAiMessage {
                    role: "system".to_string(),
                    content: Some(serde_json::Value::String("base system".to_string())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    refusal: None,
                },
                OpenAiMessage {
                    role: "assistant".to_string(),
                    content: Some(serde_json::Value::String("Let me check.".to_string())),
                    name: None,
                    tool_calls: Some(vec![crate::protocol::openai_chat::OpenAiToolCall {
                        id: "call_1".to_string(),
                        type_: "function".to_string(),
                        function: crate::protocol::openai_chat::OpenAiToolCallFunction {
                            name: "get_weather".to_string(),
                            arguments: "{\"city\":\"London\"}".to_string(),
                        },
                    }]),
                    tool_call_id: None,
                    refusal: None,
                },
                OpenAiMessage {
                    role: "tool".to_string(),
                    content: Some(serde_json::Value::String("Sunny".to_string())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: Some("call_1".to_string()),
                    refusal: None,
                },
            ],
            tools: Some(vec![OpenAiTool {
                type_: "function".to_string(),
                function: OpenAiToolFunction {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}}
                    })),
                },
            }]),
            tool_choice: Some(OpenAiToolChoice::Function(OpenAiToolChoiceFunctionCall {
                type_: "function".to_string(),
                function: OpenAiToolChoiceFunction {
                    name: "get_weather".to_string(),
                },
            })),
            stream: Some(false),
            stream_options: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            n: None,
            stop: None,
            extra: serde_json::Map::new(),
        };

        let saved_tools =
            apply_fc_inject_openai_wire(&mut req, &FeaturesConfig::default()).unwrap();
        assert_eq!(saved_tools.len(), 1);
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());
        assert_eq!(req.messages[0].role, "system");

        let system_text = req.messages[0]
            .content
            .as_ref()
            .and_then(serde_json::Value::as_str)
            .unwrap();
        assert!(system_text.contains("base system"));
        assert!(system_text.contains(crate::fc::prompt::get_trigger_signal()));

        let assistant = req.messages.iter().find(|m| m.role == "assistant").unwrap();
        let assistant_text = assistant
            .content
            .as_ref()
            .and_then(serde_json::Value::as_str)
            .unwrap();
        assert!(assistant_text.contains("<function_calls>"));
        assert!(assistant_text.contains("<tool>get_weather</tool>"));

        let tool_as_user = req.messages.iter().find(|m| m.role == "user").unwrap();
        let tool_user_text = tool_as_user
            .content
            .as_ref()
            .and_then(serde_json::Value::as_str)
            .unwrap();
        assert!(tool_user_text.contains("Tool execution result"));
        assert!(tool_user_text.contains("get_weather"));
        assert!(tool_user_text.contains("{\"city\":\"London\"}"));
    }

    #[test]
    fn test_apply_fc_inject_openai_wire_prepends_system_when_missing() {
        let mut req = OpenAiChatRequest {
            model: "m1".to_string(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: Some(serde_json::Value::String("hi".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
            }],
            tools: Some(vec![OpenAiTool {
                type_: "function".to_string(),
                function: OpenAiToolFunction {
                    name: "noop".to_string(),
                    description: Some("noop".to_string()),
                    parameters: Some(serde_json::json!({"type": "object", "properties": {}})),
                },
            }]),
            tool_choice: Some(OpenAiToolChoice::Mode("auto".to_string())),
            stream: Some(false),
            stream_options: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            n: None,
            stop: None,
            extra: serde_json::Map::new(),
        };

        let saved_tools =
            apply_fc_inject_openai_wire(&mut req, &FeaturesConfig::default()).unwrap();
        assert_eq!(saved_tools.len(), 1);
        assert_eq!(req.messages[0].role, "system");
        let system_text = req.messages[0]
            .content
            .as_ref()
            .and_then(serde_json::Value::as_str)
            .unwrap();
        assert!(system_text.contains(crate::fc::prompt::get_trigger_signal()));
    }

    #[test]
    fn test_apply_fc_inject_openai_wire_skips_when_tool_choice_none() {
        let mut req = OpenAiChatRequest {
            model: "m1".to_string(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: Some(serde_json::Value::String("hi".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
            }],
            tools: Some(vec![OpenAiTool {
                type_: "function".to_string(),
                function: OpenAiToolFunction {
                    name: "noop".to_string(),
                    description: Some("noop".to_string()),
                    parameters: Some(serde_json::json!({"type": "object", "properties": {}})),
                },
            }]),
            tool_choice: Some(OpenAiToolChoice::Mode("none".to_string())),
            stream: Some(false),
            stream_options: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            n: None,
            stop: None,
            extra: serde_json::Map::new(),
        };

        let saved_tools =
            apply_fc_inject_openai_wire(&mut req, &FeaturesConfig::default()).unwrap();
        assert!(saved_tools.is_empty());
        assert!(req.tools.as_ref().is_some_and(|tools| tools.len() == 1));
        assert!(matches!(
            req.tool_choice,
            Some(OpenAiToolChoice::Mode(ref mode)) if mode == "none"
        ));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
    }

    #[test]
    fn test_raw_fc_inject_fast_path_skips_when_tool_choice_none() {
        let body = bytes::Bytes::from_static(
            br#"{
                "model":"m1",
                "messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"noop","description":"noop","parameters":{"type":"object","properties":{}}}}],
                "tool_choice":"none",
                "stream":true
            }"#,
        );

        let built = try_build_openai_simple_fc_inject_body_from_raw(
            &body,
            "m1",
            &FeaturesConfig::default(),
            None,
        )
        .unwrap();
        assert!(built.is_none());
    }

    #[test]
    fn test_apply_fc_inject_openai_wire_skips_when_response_format_json_mode() {
        let mut req = OpenAiChatRequest {
            model: "m1".to_string(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: Some(serde_json::Value::String("hi".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
            }],
            tools: Some(vec![OpenAiTool {
                type_: "function".to_string(),
                function: OpenAiToolFunction {
                    name: "noop".to_string(),
                    description: Some("noop".to_string()),
                    parameters: Some(serde_json::json!({"type": "object", "properties": {}})),
                },
            }]),
            tool_choice: Some(OpenAiToolChoice::Mode("auto".to_string())),
            stream: Some(false),
            stream_options: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            n: None,
            stop: None,
            extra: serde_json::Map::new(),
        };
        req.extra.insert(
            "response_format".to_string(),
            serde_json::json!({"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}),
        );

        let saved_tools =
            apply_fc_inject_openai_wire(&mut req, &FeaturesConfig::default()).unwrap();
        assert!(saved_tools.is_empty());
        assert!(req.tools.as_ref().is_some_and(|tools| tools.len() == 1));
        assert!(matches!(
            req.tool_choice,
            Some(OpenAiToolChoice::Mode(ref mode)) if mode == "auto"
        ));
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_raw_fc_inject_fast_path_skips_when_response_format_json_mode() {
        let body = bytes::Bytes::from_static(
            br#"{
                "model":"m1",
                "messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"noop","description":"noop","parameters":{"type":"object","properties":{}}}}],
                "tool_choice":"auto",
                "response_format":{"type":"json_object"},
                "stream":true
            }"#,
        );

        let built = try_build_openai_simple_fc_inject_body_from_raw(
            &body,
            "m1",
            &FeaturesConfig::default(),
            None,
        )
        .unwrap();
        assert!(built.is_none());
    }

    #[test]
    fn test_build_openai_simple_inject_json_body_transforms_top_level_and_messages() {
        let body = br#"{
            "model":"m1",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"noop","description":"noop","parameters":{"type":"object","properties":{}}}}],
            "tool_choice":"auto",
            "temperature":0.1
        }"#;
        let system_msg = serde_json::to_vec(&serde_json::json!({
            "role": "system",
            "content": "fc prompt",
        }))
        .unwrap();
        let out = build_openai_simple_inject_json_body(body, "m2", &system_msg, None)
            .unwrap()
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&out).unwrap();

        assert_eq!(
            json.get("model").and_then(serde_json::Value::as_str),
            Some("m2")
        );
        assert!(json.get("tools").is_none());
        assert!(json.get("tool_choice").is_none());
        assert_eq!(
            json.get("temperature").and_then(serde_json::Value::as_f64),
            Some(0.1)
        );

        let messages = json
            .get("messages")
            .and_then(serde_json::Value::as_array)
            .unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0].get("role").and_then(serde_json::Value::as_str),
            Some("system")
        );
        assert_eq!(
            messages[0]
                .get("content")
                .and_then(serde_json::Value::as_str),
            Some("fc prompt")
        );
        assert_eq!(
            messages[1].get("role").and_then(serde_json::Value::as_str),
            Some("user")
        );
    }

    #[test]
    fn test_build_openai_simple_inject_json_body_returns_none_on_non_array_messages() {
        let body = br#"{"model":"m1","messages":{"role":"user"},"tools":[]}"#;
        let system_msg = serde_json::to_vec(&serde_json::json!({
            "role": "system",
            "content": "fc prompt",
        }))
        .unwrap();
        let out = build_openai_simple_inject_json_body(body, "m2", &system_msg, None).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_build_openai_simple_inject_json_body_escapes_inserted_model() {
        let body = br#"{"messages":[{"role":"user","content":"hi"}],"tools":[]}"#;
        let system_msg = serde_json::to_vec(&serde_json::json!({
            "role": "system",
            "content": "fc prompt",
        }))
        .unwrap();
        let model = "m\"x\\\nz";

        let out = build_openai_simple_inject_json_body(body, model, &system_msg, None)
            .unwrap()
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            json.get("model").and_then(serde_json::Value::as_str),
            Some(model)
        );
    }

    #[test]
    fn test_messages_inner_bounds_if_simple_true_for_user_assistant() {
        let messages = br#"[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"}]"#;
        assert!(messages_inner_bounds_if_simple(messages).is_some());
    }

    #[test]
    fn test_messages_inner_bounds_if_simple_false_for_system_role() {
        let messages = br#"[{"role":"system","content":"x"},{"role":"user","content":"hi"}]"#;
        assert!(messages_inner_bounds_if_simple(messages).is_none());
    }

    #[test]
    fn test_messages_inner_bounds_if_simple_false_for_non_empty_tool_calls() {
        let messages = br#"[{"role":"assistant","tool_calls":[{"id":"c1"}]}]"#;
        assert!(messages_inner_bounds_if_simple(messages).is_none());
    }

    #[test]
    fn test_route_prompt_prefix_prefers_messages_field() {
        let body = br#"{
            "model":"smart",
            "messages":[{"role":"user","content":"hello world"}],
            "tools":[{"type":"function","function":{"name":"noop","parameters":{"type":"object"}}}]
        }"#;
        let prefix = route_prompt_prefix_bytes(body, None);
        let prefix_str = std::str::from_utf8(prefix).unwrap();
        assert!(prefix_str.starts_with("[{\"role\":\"user\""));
    }

    #[test]
    fn test_route_prompt_prefix_caps_to_256_bytes() {
        let long_input = "a".repeat(1024);
        let body = format!(r#"{{"model":"m1","input":"{long_input}"}}"#);
        let prefix = route_prompt_prefix_bytes(body.as_bytes(), None);
        assert!(prefix.len() <= 256);
    }
}
