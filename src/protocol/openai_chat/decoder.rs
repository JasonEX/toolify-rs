use serde_json::Value;

use crate::error::CanonicalError;
use crate::protocol::canonical::{
    provider_extensions_from_map, CanonicalMessage, CanonicalPart, CanonicalRequest,
    CanonicalToolChoice, CanonicalToolFunction, CanonicalToolSpec, GenerationParams, IngressApi,
};
use crate::protocol::mapping::openai_role_to_canonical;
use crate::util::raw_value_from_string;

use super::{OpenAiChatRequest, OpenAiStop, OpenAiTool, OpenAiToolChoice};

struct DecodedChatFields {
    model: String,
    stream: bool,
    system_prompt: Option<String>,
    messages: Vec<CanonicalMessage>,
    tools: Vec<CanonicalToolSpec>,
    tool_choice: CanonicalToolChoice,
    generation: GenerationParams,
    extra: serde_json::Map<String, Value>,
}

/// Decode an `OpenAI` Chat Completions wire request into the canonical IR.
///
/// # Errors
///
/// Returns [`CanonicalError`] when any message payload cannot be decoded.
pub fn decode_openai_chat_request(
    request: &OpenAiChatRequest,
    request_id: uuid::Uuid,
) -> Result<CanonicalRequest, CanonicalError> {
    let (system_prompt, messages) = collect_messages_borrowed(&request.messages)?;
    let decoded = DecodedChatFields {
        model: request.model.clone(),
        stream: request.stream.unwrap_or(false),
        system_prompt,
        messages,
        tools: decode_tools(request.tools.as_deref()),
        tool_choice: decode_tool_choice(request.tool_choice.as_ref()),
        generation: GenerationParams {
            temperature: request.temperature,
            max_tokens: request.max_tokens.or(request.max_completion_tokens),
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            n: request.n,
            stop: decode_stop(request.stop.as_ref()),
        },
        extra: request.extra.clone(),
    };

    Ok(build_openai_chat_request(request_id, decoded))
}

/// Decode an `OpenAI` Chat request by consuming ownership to avoid extra clones
/// on hot paths.
///
/// # Errors
///
/// Returns [`CanonicalError`] when any message payload cannot be decoded.
pub fn decode_openai_chat_request_owned(
    request: OpenAiChatRequest,
    request_id: uuid::Uuid,
) -> Result<CanonicalRequest, CanonicalError> {
    let OpenAiChatRequest {
        model,
        messages,
        tools,
        tool_choice,
        stream,
        stream_options: _,
        temperature,
        max_tokens,
        max_completion_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
        n,
        stop,
        extra,
    } = request;

    let (system_prompt, messages) = collect_messages_owned(messages)?;
    let decoded = DecodedChatFields {
        model,
        stream: stream.unwrap_or(false),
        system_prompt,
        messages,
        tools: decode_tools_owned(tools),
        tool_choice: decode_tool_choice_owned(tool_choice),
        generation: GenerationParams {
            temperature,
            max_tokens: max_tokens.or(max_completion_tokens),
            top_p,
            frequency_penalty,
            presence_penalty,
            n,
            stop: decode_stop_owned(stop),
        },
        extra,
    };

    Ok(build_openai_chat_request(request_id, decoded))
}

fn build_openai_chat_request(
    request_id: uuid::Uuid,
    decoded: DecodedChatFields,
) -> CanonicalRequest {
    CanonicalRequest {
        request_id,
        ingress_api: IngressApi::OpenAiChat,
        model: decoded.model,
        stream: decoded.stream,
        system_prompt: decoded.system_prompt,
        messages: decoded.messages,
        tools: decoded.tools.into(),
        tool_choice: decoded.tool_choice,
        generation: decoded.generation,
        provider_extensions: provider_extensions_from_map(decoded.extra),
    }
}

fn collect_messages_borrowed(
    messages: &[super::OpenAiMessage],
) -> Result<(Option<String>, Vec<CanonicalMessage>), CanonicalError> {
    let mut system_parts: Vec<String> = Vec::new();
    let mut canonical_messages: Vec<CanonicalMessage> = Vec::with_capacity(messages.len());

    for msg in messages {
        let role = openai_role_to_canonical(&msg.role);
        if role == crate::protocol::canonical::CanonicalRole::System {
            if let Some(text) = extract_text_content(msg.content.as_ref()) {
                system_parts.push(text);
            }
            continue;
        }
        canonical_messages.push(decode_message(msg)?);
    }

    Ok((build_system_prompt(&system_parts), canonical_messages))
}

fn collect_messages_owned(
    messages: Vec<super::OpenAiMessage>,
) -> Result<(Option<String>, Vec<CanonicalMessage>), CanonicalError> {
    let mut system_parts: Vec<String> = Vec::new();
    let mut canonical_messages: Vec<CanonicalMessage> = Vec::with_capacity(messages.len());

    for msg in messages {
        let role = openai_role_to_canonical(&msg.role);
        if role == crate::protocol::canonical::CanonicalRole::System {
            if let Some(text) = extract_text_content_owned(msg.content) {
                system_parts.push(text);
            }
            continue;
        }
        canonical_messages.push(decode_message_owned(msg)?);
    }

    Ok((build_system_prompt(&system_parts), canonical_messages))
}

#[inline]
fn build_system_prompt(system_parts: &[String]) -> Option<String> {
    if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n"))
    }
}

/// Extract plain text from an `OpenAI` message content field.
fn extract_text_content(content: Option<&Value>) -> Option<String> {
    match content {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(arr)) => {
            let mut text = String::new();
            for part in arr {
                if part.get("type").and_then(|t| t.as_str()) != Some("text") {
                    continue;
                }
                if let Some(content) = part.get("text").and_then(|t| t.as_str()) {
                    text.push_str(content);
                }
            }
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        None | Some(_) => None,
    }
}

fn extract_text_content_owned(content: Option<Value>) -> Option<String> {
    match content {
        Some(Value::String(s)) => Some(s),
        Some(Value::Array(arr)) => {
            let mut text = String::new();
            for part in arr {
                if part.get("type").and_then(|t| t.as_str()) != Some("text") {
                    continue;
                }
                if let Some(content) = part.get("text").and_then(|t| t.as_str()) {
                    text.push_str(content);
                }
            }
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        None | Some(_) => None,
    }
}

fn decode_message(msg: &super::OpenAiMessage) -> Result<CanonicalMessage, CanonicalError> {
    let role = openai_role_to_canonical(&msg.role);
    let mut parts: Vec<CanonicalPart> = Vec::with_capacity(
        usize::from(msg.refusal.is_some())
            + msg.tool_calls.as_ref().map_or(0, std::vec::Vec::len)
            + match &msg.content {
                Some(Value::String(_)) => 1,
                Some(Value::Array(arr)) => arr.len(),
                _ => 0,
            },
    );

    if role == crate::protocol::canonical::CanonicalRole::Tool {
        let text = extract_text_content(msg.content.as_ref()).unwrap_or_default();
        let tool_call_id = msg.tool_call_id.clone().unwrap_or_default();
        parts.push(CanonicalPart::ToolResult {
            tool_call_id: tool_call_id.clone(),
            content: text,
        });
        return Ok(CanonicalMessage {
            role,
            parts: parts.into(),
            name: msg.name.clone(),
            tool_call_id: msg.tool_call_id.clone(),
            provider_extensions: None,
        });
    }

    if let Some(ref refusal) = msg.refusal {
        parts.push(CanonicalPart::Refusal(refusal.clone()));
    }

    match &msg.content {
        Some(Value::String(s)) => {
            if !s.is_empty() {
                parts.push(CanonicalPart::Text(s.clone()));
            }
        }
        Some(Value::Array(arr)) => {
            for part in arr {
                match part.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                            parts.push(CanonicalPart::Text(t.to_string()));
                        }
                    }
                    Some("image_url") => {
                        if let Some(img) = part.get("image_url") {
                            let url = img
                                .get("url")
                                .and_then(|u| u.as_str())
                                .unwrap_or("")
                                .to_string();
                            let detail = img
                                .get("detail")
                                .and_then(|d| d.as_str())
                                .map(std::string::ToString::to_string);
                            parts.push(CanonicalPart::ImageUrl { url, detail });
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }

    if let Some(ref tool_calls) = msg.tool_calls {
        for tc in tool_calls {
            let arguments =
                raw_value_from_string(tc.function.arguments.clone(), "OpenAI tool call")?;
            parts.push(CanonicalPart::ToolCall {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                arguments,
            });
        }
    }

    Ok(CanonicalMessage {
        role,
        parts: parts.into(),
        name: msg.name.clone(),
        tool_call_id: msg.tool_call_id.clone(),
        provider_extensions: None,
    })
}

fn decode_message_owned(msg: super::OpenAiMessage) -> Result<CanonicalMessage, CanonicalError> {
    let super::OpenAiMessage {
        role: wire_role,
        content,
        name,
        tool_calls,
        tool_call_id,
        refusal,
    } = msg;

    let role = openai_role_to_canonical(&wire_role);
    let mut parts: Vec<CanonicalPart> = Vec::with_capacity(
        usize::from(refusal.is_some())
            + tool_calls.as_ref().map_or(0, std::vec::Vec::len)
            + match &content {
                Some(Value::String(_)) => 1,
                Some(Value::Array(arr)) => arr.len(),
                _ => 0,
            },
    );

    if role == crate::protocol::canonical::CanonicalRole::Tool {
        let text = extract_text_content_owned(content).unwrap_or_default();
        parts.push(CanonicalPart::ToolResult {
            tool_call_id: tool_call_id.clone().unwrap_or_default(),
            content: text,
        });
        return Ok(CanonicalMessage {
            role,
            parts: parts.into(),
            name,
            tool_call_id,
            provider_extensions: None,
        });
    }

    if let Some(refusal_text) = refusal {
        parts.push(CanonicalPart::Refusal(refusal_text));
    }

    match content {
        Some(Value::String(s)) => {
            if !s.is_empty() {
                parts.push(CanonicalPart::Text(s));
            }
        }
        Some(Value::Array(arr)) => {
            for part in arr {
                match part.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                            parts.push(CanonicalPart::Text(t.to_string()));
                        }
                    }
                    Some("image_url") => {
                        if let Some(img) = part.get("image_url") {
                            let url = img
                                .get("url")
                                .and_then(|u| u.as_str())
                                .unwrap_or("")
                                .to_string();
                            let detail = img
                                .get("detail")
                                .and_then(|d| d.as_str())
                                .map(std::string::ToString::to_string);
                            parts.push(CanonicalPart::ImageUrl { url, detail });
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }

    if let Some(tool_calls) = tool_calls {
        for tc in tool_calls {
            let arguments = raw_value_from_string(tc.function.arguments, "OpenAI tool call")?;
            parts.push(CanonicalPart::ToolCall {
                id: tc.id,
                name: tc.function.name,
                arguments,
            });
        }
    }

    Ok(CanonicalMessage {
        role,
        parts: parts.into(),
        name,
        tool_call_id,
        provider_extensions: None,
    })
}

fn decode_tools(tools: Option<&[OpenAiTool]>) -> Vec<CanonicalToolSpec> {
    match tools {
        None => Vec::new(),
        Some(tools) => {
            let mut out = Vec::with_capacity(tools.len());
            for tool in tools {
                out.push(decode_tool(tool));
            }
            out
        }
    }
}

fn decode_tools_owned(tools: Option<Vec<OpenAiTool>>) -> Vec<CanonicalToolSpec> {
    match tools {
        None => Vec::new(),
        Some(tools) => tools.into_iter().map(decode_tool_owned).collect(),
    }
}

fn decode_tool(tool: &OpenAiTool) -> CanonicalToolSpec {
    CanonicalToolSpec {
        function: CanonicalToolFunction {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters: tool
                .function
                .parameters
                .clone()
                .unwrap_or(Value::Object(serde_json::Map::new())),
        },
    }
}

fn decode_tool_owned(tool: OpenAiTool) -> CanonicalToolSpec {
    CanonicalToolSpec {
        function: CanonicalToolFunction {
            name: tool.function.name,
            description: tool.function.description,
            parameters: tool
                .function
                .parameters
                .unwrap_or(Value::Object(serde_json::Map::new())),
        },
    }
}

fn decode_tool_choice(val: Option<&OpenAiToolChoice>) -> CanonicalToolChoice {
    match val {
        None => CanonicalToolChoice::Auto,
        Some(OpenAiToolChoice::Mode(s)) => decode_tool_choice_mode(s),
        Some(OpenAiToolChoice::Function(call)) => {
            CanonicalToolChoice::Specific(call.function.name.clone())
        }
    }
}

fn decode_tool_choice_owned(val: Option<OpenAiToolChoice>) -> CanonicalToolChoice {
    match val {
        None => CanonicalToolChoice::Auto,
        Some(OpenAiToolChoice::Mode(s)) => decode_tool_choice_mode(&s),
        Some(OpenAiToolChoice::Function(call)) => CanonicalToolChoice::Specific(call.function.name),
    }
}

fn decode_stop(val: Option<&OpenAiStop>) -> Option<Vec<String>> {
    match val {
        None => None,
        Some(OpenAiStop::Single(s)) => Some(vec![s.clone()]),
        Some(OpenAiStop::Multi(items)) => non_empty(items.clone()),
    }
}

fn decode_stop_owned(val: Option<OpenAiStop>) -> Option<Vec<String>> {
    match val {
        None => None,
        Some(OpenAiStop::Single(s)) => Some(vec![s]),
        Some(OpenAiStop::Multi(items)) => non_empty(items),
    }
}

#[inline]
fn decode_tool_choice_mode(mode: &str) -> CanonicalToolChoice {
    match mode {
        "none" => CanonicalToolChoice::None,
        "required" => CanonicalToolChoice::Required,
        _ => CanonicalToolChoice::Auto,
    }
}

#[inline]
fn non_empty(items: Vec<String>) -> Option<Vec<String>> {
    if items.is_empty() {
        None
    } else {
        Some(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_request(messages: &[Value]) -> OpenAiChatRequest {
        serde_json::from_value(json!({
            "model": "gpt-4",
            "messages": messages,
        }))
        .unwrap()
    }

    #[test]
    fn test_simple_user_message() {
        let req = make_request(&[json!({"role": "user", "content": "Hello"})]);
        let canon = decode_openai_chat_request(&req, uuid::Uuid::nil()).unwrap();
        assert_eq!(canon.messages.len(), 1);
        assert!(matches!(&canon.messages[0].parts[0], CanonicalPart::Text(t) if t == "Hello"));
    }

    #[test]
    fn test_system_extraction() {
        let req = make_request(&[
            json!({"role": "system", "content": "You are helpful."}),
            json!({"role": "user", "content": "Hi"}),
        ]);
        let canon = decode_openai_chat_request(&req, uuid::Uuid::nil()).unwrap();
        assert_eq!(canon.system_prompt.as_deref(), Some("You are helpful."));
        assert_eq!(canon.messages.len(), 1);
    }

    #[test]
    fn test_tool_choice_specific() {
        let tc = decode_tool_choice(Some(&OpenAiToolChoice::Function(
            super::super::OpenAiToolChoiceFunctionCall {
                type_: "function".to_string(),
                function: super::super::OpenAiToolChoiceFunction {
                    name: "get_weather".to_string(),
                },
            },
        )));
        assert_eq!(tc, CanonicalToolChoice::Specific("get_weather".to_string()));
    }

    #[test]
    fn test_tool_message() {
        let req = make_request(&[json!({
            "role": "tool",
            "content": "result data",
            "tool_call_id": "call_123"
        })]);
        let canon = decode_openai_chat_request(&req, uuid::Uuid::nil()).unwrap();
        assert!(matches!(
            &canon.messages[0].parts[0],
            CanonicalPart::ToolResult { tool_call_id, content }
                if tool_call_id == "call_123" && content == "result data"
        ));
    }

    #[test]
    fn test_assistant_tool_calls() {
        let req = make_request(&[json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\":\"NYC\"}"
                }
            }]
        })]);
        let canon = decode_openai_chat_request(&req, uuid::Uuid::nil()).unwrap();
        assert!(matches!(
            &canon.messages[0].parts[0],
            CanonicalPart::ToolCall { id, name, .. }
                if id == "call_abc" && name == "get_weather"
        ));
    }

    #[test]
    fn test_multipart_content() {
        let req = make_request(&[json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "high"}}
            ]
        })]);
        let canon = decode_openai_chat_request(&req, uuid::Uuid::nil()).unwrap();
        assert_eq!(canon.messages[0].parts.len(), 2);
        assert!(matches!(
            &canon.messages[0].parts[1],
            CanonicalPart::ImageUrl { url, detail }
                if url == "https://example.com/img.png" && detail.as_deref() == Some("high")
        ));
    }

    #[test]
    fn test_generation_params() {
        let req: OpenAiChatRequest = serde_json::from_value(json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "stop": ["END", "STOP"]
        }))
        .unwrap();
        let canon = decode_openai_chat_request(&req, uuid::Uuid::nil()).unwrap();
        assert_eq!(canon.generation.temperature, Some(0.7));
        assert_eq!(canon.generation.max_tokens, Some(100));
        assert_eq!(canon.generation.top_p, Some(0.9));
        assert_eq!(
            canon.generation.stop,
            Some(vec!["END".to_string(), "STOP".to_string()])
        );
    }
}
