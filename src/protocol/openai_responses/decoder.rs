use crate::error::CanonicalError;
use crate::protocol::canonical::{
    provider_extensions_from_map, CanonicalMessage, CanonicalPart, CanonicalRequest, CanonicalRole,
    CanonicalToolChoice, CanonicalToolFunction, CanonicalToolSpec, GenerationParams, IngressApi,
};
use crate::util::raw_value_from_string;

use super::{ResponsesRequest, ResponsesTool};

struct DecodedResponsesFields {
    model: String,
    stream: bool,
    system_prompt: Option<String>,
    messages: Vec<CanonicalMessage>,
    tools: Vec<CanonicalToolSpec>,
    tool_choice: CanonicalToolChoice,
    generation: GenerationParams,
    provider_extensions: serde_json::Map<String, serde_json::Value>,
}

/// Decode an `OpenAI` Responses API request into a canonical request.
///
/// # Errors
///
/// Returns [`CanonicalError`] when input items or tool-call arguments cannot be
/// decoded into canonical representations.
pub fn decode_responses_request(
    request: &ResponsesRequest,
    request_id: uuid::Uuid,
) -> Result<CanonicalRequest, CanonicalError> {
    let messages = decode_input(&request.input)?;
    let mut provider_extensions = request.extra.clone();
    let tools = decode_response_tools_borrowed(request.tools.as_deref(), &mut provider_extensions);
    add_previous_response_id(
        &mut provider_extensions,
        request.previous_response_id.as_deref(),
    );
    add_store(&mut provider_extensions, request.store);

    let decoded = DecodedResponsesFields {
        model: request.model.clone(),
        stream: request.stream.unwrap_or(false),
        system_prompt: request.instructions.clone(),
        messages,
        tools,
        tool_choice: decode_responses_tool_choice(request.tool_choice.as_ref()),
        generation: GenerationParams {
            temperature: request.temperature,
            max_tokens: request.max_output_tokens,
            top_p: request.top_p,
            ..Default::default()
        },
        provider_extensions,
    };

    Ok(build_responses_request(request_id, decoded))
}

/// Decode an `OpenAI` Responses API request by consuming ownership to reduce
/// clone cost on hot paths.
///
/// # Errors
///
/// Returns [`CanonicalError`] when input items or tool-call arguments cannot be
/// decoded into canonical representations.
pub fn decode_responses_request_owned(
    request: ResponsesRequest,
    request_id: uuid::Uuid,
) -> Result<CanonicalRequest, CanonicalError> {
    let ResponsesRequest {
        model,
        input,
        instructions,
        tools: req_tools,
        tool_choice,
        previous_response_id,
        store,
        stream,
        temperature,
        max_output_tokens,
        top_p,
        extra,
    } = request;

    let messages = decode_input_owned(input)?;
    let mut provider_extensions = extra;
    let tools = decode_response_tools_owned(req_tools, &mut provider_extensions);
    add_previous_response_id(&mut provider_extensions, previous_response_id.as_deref());
    add_store(&mut provider_extensions, store);

    let decoded = DecodedResponsesFields {
        model,
        stream: stream.unwrap_or(false),
        system_prompt: instructions,
        messages,
        tools,
        tool_choice: decode_responses_tool_choice(tool_choice.as_ref()),
        generation: GenerationParams {
            temperature,
            max_tokens: max_output_tokens,
            top_p,
            ..Default::default()
        },
        provider_extensions,
    };

    Ok(build_responses_request(request_id, decoded))
}

fn build_responses_request(
    request_id: uuid::Uuid,
    decoded: DecodedResponsesFields,
) -> CanonicalRequest {
    CanonicalRequest {
        request_id,
        ingress_api: IngressApi::OpenAiResponses,
        model: decoded.model,
        stream: decoded.stream,
        system_prompt: decoded.system_prompt,
        messages: decoded.messages,
        tools: decoded.tools.into(),
        tool_choice: decoded.tool_choice,
        generation: decoded.generation,
        provider_extensions: provider_extensions_from_map(decoded.provider_extensions),
    }
}

fn decode_response_tools_borrowed(
    req_tools: Option<&[ResponsesTool]>,
    provider_extensions: &mut serde_json::Map<String, serde_json::Value>,
) -> Vec<CanonicalToolSpec> {
    let Some(req_tools) = req_tools else {
        return Vec::new();
    };
    let mut tools = Vec::new();
    let mut builtin_tools = Vec::new();
    for tool in req_tools {
        match tool {
            ResponsesTool::Function {
                name,
                description,
                parameters,
            } => {
                tools.push(CanonicalToolSpec {
                    function: CanonicalToolFunction {
                        name: name.clone(),
                        description: description.clone(),
                        parameters: parameters
                            .clone()
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                    },
                });
            }
            ResponsesTool::WebSearch { extra } => {
                builtin_tools.push(build_builtin_tool_entry("web_search", extra.clone()));
            }
            ResponsesTool::FileSearch { extra } => {
                builtin_tools.push(build_builtin_tool_entry("file_search", extra.clone()));
            }
        }
    }
    if !builtin_tools.is_empty() {
        provider_extensions.insert(
            "responses_builtin_tools".into(),
            serde_json::Value::Array(builtin_tools),
        );
    }
    tools
}

fn decode_response_tools_owned(
    req_tools: Option<Vec<ResponsesTool>>,
    provider_extensions: &mut serde_json::Map<String, serde_json::Value>,
) -> Vec<CanonicalToolSpec> {
    let Some(req_tools) = req_tools else {
        return Vec::new();
    };
    let mut tools = Vec::new();
    let mut builtin_tools = Vec::new();
    for tool in req_tools {
        match tool {
            ResponsesTool::Function {
                name,
                description,
                parameters,
            } => {
                tools.push(CanonicalToolSpec {
                    function: CanonicalToolFunction {
                        name,
                        description,
                        parameters: parameters
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                    },
                });
            }
            ResponsesTool::WebSearch { extra } => {
                builtin_tools.push(build_builtin_tool_entry("web_search", extra));
            }
            ResponsesTool::FileSearch { extra } => {
                builtin_tools.push(build_builtin_tool_entry("file_search", extra));
            }
        }
    }
    if !builtin_tools.is_empty() {
        provider_extensions.insert(
            "responses_builtin_tools".into(),
            serde_json::Value::Array(builtin_tools),
        );
    }
    tools
}

#[inline]
fn build_builtin_tool_entry(
    tool_type: &str,
    mut extra: serde_json::Map<String, serde_json::Value>,
) -> serde_json::Value {
    extra.insert("type".into(), serde_json::Value::String(tool_type.into()));
    serde_json::Value::Object(extra)
}

#[inline]
fn add_previous_response_id(
    provider_extensions: &mut serde_json::Map<String, serde_json::Value>,
    previous_response_id: Option<&str>,
) {
    if let Some(prev_id) = previous_response_id {
        provider_extensions.insert(
            "previous_response_id".into(),
            serde_json::Value::String(prev_id.to_string()),
        );
    }
}

#[inline]
fn add_store(
    provider_extensions: &mut serde_json::Map<String, serde_json::Value>,
    store: Option<bool>,
) {
    if let Some(store) = store {
        provider_extensions.insert("store".into(), serde_json::Value::Bool(store));
    }
}

fn decode_responses_tool_choice(tool_choice: Option<&serde_json::Value>) -> CanonicalToolChoice {
    let Some(choice) = tool_choice else {
        return CanonicalToolChoice::Auto;
    };

    match choice {
        serde_json::Value::String(mode) => match mode.as_str() {
            "none" => CanonicalToolChoice::None,
            "required" => CanonicalToolChoice::Required,
            _ => CanonicalToolChoice::Auto,
        },
        serde_json::Value::Object(obj) => {
            if let Some(name) = obj
                .get("name")
                .and_then(serde_json::Value::as_str)
                .filter(|name| !name.is_empty())
            {
                return CanonicalToolChoice::Specific(name.to_string());
            }

            if let Some(name) = obj
                .get("function")
                .and_then(serde_json::Value::as_object)
                .and_then(|func| func.get("name"))
                .and_then(serde_json::Value::as_str)
                .filter(|name| !name.is_empty())
            {
                return CanonicalToolChoice::Specific(name.to_string());
            }

            let Some(choice_type) = obj.get("type").and_then(serde_json::Value::as_str) else {
                return CanonicalToolChoice::Auto;
            };

            match choice_type {
                "none" => CanonicalToolChoice::None,
                "required" => CanonicalToolChoice::Required,
                "function" => obj
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .map_or(CanonicalToolChoice::Auto, |name| {
                        CanonicalToolChoice::Specific(name.to_string())
                    }),
                "allowed_tools" => {
                    let mode = obj
                        .get("mode")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("auto");
                    if mode == "none" {
                        return CanonicalToolChoice::None;
                    }

                    if mode == "required" {
                        let specific = obj
                            .get("tools")
                            .and_then(serde_json::Value::as_array)
                            .and_then(|tools| {
                                if tools.len() != 1 {
                                    return None;
                                }
                                let tool = tools.first()?.as_object()?;
                                if tool.get("type").and_then(serde_json::Value::as_str)
                                    != Some("function")
                                {
                                    return None;
                                }
                                tool.get("name")
                                    .and_then(serde_json::Value::as_str)
                                    .map(std::string::ToString::to_string)
                            });
                        if let Some(name) = specific {
                            CanonicalToolChoice::Specific(name)
                        } else {
                            CanonicalToolChoice::Required
                        }
                    } else {
                        CanonicalToolChoice::Auto
                    }
                }
                _ => CanonicalToolChoice::Auto,
            }
        }
        _ => CanonicalToolChoice::Auto,
    }
}

/// Decode the `input` field which can be a string or an array of input items.
fn decode_input(input: &serde_json::Value) -> Result<Vec<CanonicalMessage>, CanonicalError> {
    match input {
        serde_json::Value::String(s) => Ok(vec![CanonicalMessage {
            role: CanonicalRole::User,
            parts: vec![CanonicalPart::Text(s.clone())].into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        }]),
        serde_json::Value::Array(items) => {
            let mut messages = Vec::new();
            for item in items {
                let msg = decode_input_item(item)?;
                messages.push(msg);
            }
            Ok(messages)
        }
        _ => Err(CanonicalError::InvalidRequest(
            "Responses API `input` must be a string or array".into(),
        )),
    }
}

fn decode_input_owned(input: serde_json::Value) -> Result<Vec<CanonicalMessage>, CanonicalError> {
    match input {
        serde_json::Value::String(s) => Ok(vec![CanonicalMessage {
            role: CanonicalRole::User,
            parts: vec![CanonicalPart::Text(s)].into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        }]),
        serde_json::Value::Array(items) => {
            let mut messages = Vec::new();
            for item in items {
                messages.push(decode_input_item_owned(item)?);
            }
            Ok(messages)
        }
        _ => Err(CanonicalError::InvalidRequest(
            "Responses API `input` must be a string or array".into(),
        )),
    }
}

#[inline]
fn decode_message_role(role_str: &str) -> CanonicalRole {
    match role_str {
        "assistant" => CanonicalRole::Assistant,
        "system" | "developer" => CanonicalRole::System,
        _ => CanonicalRole::User,
    }
}

#[inline]
fn build_message(role: CanonicalRole, parts: Vec<CanonicalPart>) -> CanonicalMessage {
    CanonicalMessage {
        role,
        parts: parts.into(),
        name: None,
        tool_call_id: None,
        provider_extensions: None,
    }
}

#[inline]
fn build_function_call_message(
    call_id: String,
    name: String,
    arguments: Box<serde_json::value::RawValue>,
) -> CanonicalMessage {
    CanonicalMessage {
        role: CanonicalRole::Assistant,
        parts: vec![CanonicalPart::ToolCall {
            id: call_id,
            name,
            arguments,
        }]
        .into(),
        name: None,
        tool_call_id: None,
        provider_extensions: None,
    }
}

#[inline]
fn build_function_call_output_message(call_id: String, output: String) -> CanonicalMessage {
    CanonicalMessage {
        role: CanonicalRole::Tool,
        parts: vec![CanonicalPart::ToolResult {
            tool_call_id: call_id.clone(),
            content: output,
        }]
        .into(),
        name: None,
        tool_call_id: Some(call_id),
        provider_extensions: None,
    }
}

fn decode_message_parts_borrowed(content: Option<&serde_json::Value>) -> Vec<CanonicalPart> {
    let mut parts = Vec::new();
    let Some(content) = content else {
        return parts;
    };
    match content {
        serde_json::Value::String(s) => parts.push(CanonicalPart::Text(s.clone())),
        serde_json::Value::Array(arr) => {
            for part in arr {
                if let Some(decoded) = decode_content_part_borrowed(part) {
                    parts.push(decoded);
                }
            }
        }
        _ => {}
    }
    parts
}

fn decode_content_part_borrowed(part: &serde_json::Value) -> Option<CanonicalPart> {
    let part_type = part
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("input_text");
    match part_type {
        "refusal" => part
            .get("refusal")
            .and_then(serde_json::Value::as_str)
            .map(|refusal| CanonicalPart::Refusal(refusal.to_string())),
        "input_image" => part
            .get("image_url")
            .and_then(serde_json::Value::as_str)
            .map(|url| CanonicalPart::ImageUrl {
                url: url.to_string(),
                detail: part
                    .get("detail")
                    .and_then(serde_json::Value::as_str)
                    .map(std::string::ToString::to_string),
            }),
        _ => part
            .get("text")
            .and_then(serde_json::Value::as_str)
            .map(|text| CanonicalPart::Text(text.to_string())),
    }
}

fn decode_message_parts_owned(content: Option<serde_json::Value>) -> Vec<CanonicalPart> {
    let mut parts = Vec::new();
    let Some(content) = content else {
        return parts;
    };
    match content {
        serde_json::Value::String(s) => parts.push(CanonicalPart::Text(s)),
        serde_json::Value::Array(arr) => {
            for part in arr {
                let serde_json::Value::Object(part_obj) = part else {
                    continue;
                };
                if let Some(decoded) = decode_content_part_owned(part_obj) {
                    parts.push(decoded);
                }
            }
        }
        _ => {}
    }
    parts
}

fn decode_content_part_owned(
    mut part_obj: serde_json::Map<String, serde_json::Value>,
) -> Option<CanonicalPart> {
    let part_type = match part_obj.remove("type") {
        Some(serde_json::Value::String(s)) => s,
        _ => "input_text".to_string(),
    };
    match part_type.as_str() {
        "refusal" => match part_obj.remove("refusal") {
            Some(serde_json::Value::String(refusal)) => Some(CanonicalPart::Refusal(refusal)),
            _ => None,
        },
        "input_image" => decode_input_image_part_owned(part_obj),
        _ => match part_obj.remove("text") {
            Some(serde_json::Value::String(text)) => Some(CanonicalPart::Text(text)),
            _ => None,
        },
    }
}

fn decode_input_image_part_owned(
    mut part_obj: serde_json::Map<String, serde_json::Value>,
) -> Option<CanonicalPart> {
    let image_url = part_obj.remove("image_url")?;
    let (url, detail) = match image_url {
        serde_json::Value::String(url) => {
            let detail = match part_obj.remove("detail") {
                Some(serde_json::Value::String(s)) => Some(s),
                _ => None,
            };
            (url, detail)
        }
        serde_json::Value::Object(mut image_obj) => {
            let url = match image_obj.remove("url") {
                Some(serde_json::Value::String(s)) => s,
                _ => String::new(),
            };
            let detail = match image_obj.remove("detail") {
                Some(serde_json::Value::String(s)) => Some(s),
                _ => None,
            };
            (url, detail)
        }
        _ => (String::new(), None),
    };
    if url.is_empty() {
        None
    } else {
        Some(CanonicalPart::ImageUrl { url, detail })
    }
}

/// Decode a single input item from the array form.
fn decode_input_item(item: &serde_json::Value) -> Result<CanonicalMessage, CanonicalError> {
    let item_type = item
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("message");

    match item_type {
        "message" => {
            let role = decode_message_role(
                item.get("role")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("user"),
            );
            Ok(build_message(
                role,
                decode_message_parts_borrowed(item.get("content")),
            ))
        }
        "function_call" => {
            let call_id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let arguments = parse_responses_call_arguments(item.get("arguments"))?;
            Ok(build_function_call_message(call_id, name, arguments))
        }
        "function_call_output" => {
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let output = item
                .get("output")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Ok(build_function_call_output_message(call_id, output))
        }
        _ => Err(CanonicalError::InvalidRequest(format!(
            "Unknown Responses API input item type: {item_type}"
        ))),
    }
}

fn decode_input_item_owned(item: serde_json::Value) -> Result<CanonicalMessage, CanonicalError> {
    let serde_json::Value::Object(mut obj) = item else {
        return Ok(build_message(CanonicalRole::User, Vec::new()));
    };

    let item_type = match obj.remove("type") {
        Some(serde_json::Value::String(s)) => s,
        _ => "message".to_string(),
    };

    match item_type.as_str() {
        "message" => {
            let role = match obj.remove("role") {
                Some(serde_json::Value::String(role_str)) => decode_message_role(&role_str),
                _ => CanonicalRole::User,
            };
            Ok(build_message(
                role,
                decode_message_parts_owned(obj.remove("content")),
            ))
        }
        "function_call" => {
            let call_id = match obj.remove("call_id").or_else(|| obj.remove("id")) {
                Some(serde_json::Value::String(s)) => s,
                _ => String::new(),
            };
            let name = match obj.remove("name") {
                Some(serde_json::Value::String(s)) => s,
                _ => String::new(),
            };
            let arguments = parse_responses_call_arguments(obj.get("arguments"))?;
            Ok(build_function_call_message(call_id, name, arguments))
        }
        "function_call_output" => {
            let call_id = match obj.remove("call_id") {
                Some(serde_json::Value::String(s)) => s,
                _ => String::new(),
            };
            let output = match obj.remove("output") {
                Some(serde_json::Value::String(s)) => s,
                _ => String::new(),
            };
            Ok(build_function_call_output_message(call_id, output))
        }
        _ => Err(CanonicalError::InvalidRequest(format!(
            "Unknown Responses API input item type: {item_type}"
        ))),
    }
}

fn parse_responses_call_arguments(
    value: Option<&serde_json::Value>,
) -> Result<Box<serde_json::value::RawValue>, CanonicalError> {
    let json = match value {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).map_err(|e| {
            CanonicalError::InvalidRequest(format!(
                "Invalid Responses function_call arguments: {e}"
            ))
        })?,
        None => "{}".to_string(),
    };

    raw_value_from_string(json, "Responses function_call arguments").map_err(|e| {
        CanonicalError::InvalidRequest(format!(
            "Responses function_call arguments must be valid JSON: {e}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_string_input() {
        let req = ResponsesRequest {
            model: "gpt-4o".into(),
            input: serde_json::Value::String("Hello".into()),
            instructions: Some("You are helpful.".into()),
            tools: None,
            tool_choice: None,
            previous_response_id: None,
            store: None,
            stream: None,
            temperature: Some(0.7),
            max_output_tokens: Some(1024),
            top_p: None,
            extra: serde_json::Map::new(),
        };

        let result = decode_responses_request(&req, uuid::Uuid::from_u128(1)).unwrap();
        assert_eq!(result.system_prompt, Some("You are helpful.".into()));
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, CanonicalRole::User);
        assert_eq!(result.generation.temperature, Some(0.7));
        assert_eq!(result.generation.max_tokens, Some(1024));
    }

    #[test]
    fn test_decode_array_input_with_function_call_output() {
        let input = serde_json::json!([
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather?"
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "{\"temp\": 72}"
            }
        ]);

        let req = ResponsesRequest {
            model: "gpt-4o".into(),
            input,
            instructions: None,
            tools: None,
            tool_choice: None,
            previous_response_id: Some("resp_abc".into()),
            store: None,
            stream: None,
            temperature: None,
            max_output_tokens: None,
            top_p: None,
            extra: serde_json::Map::new(),
        };

        let result = decode_responses_request(&req, uuid::Uuid::from_u128(1)).unwrap();
        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[1].role, CanonicalRole::Tool);
        assert_eq!(
            result.provider_extensions_ref().get("previous_response_id"),
            Some(&serde_json::Value::String("resp_abc".into()))
        );
    }

    #[test]
    fn test_decode_builtin_tools_passthrough() {
        let req = ResponsesRequest {
            model: "gpt-4o".into(),
            input: serde_json::Value::String("search the web".into()),
            instructions: None,
            tools: Some(vec![
                ResponsesTool::Function {
                    name: "get_weather".into(),
                    description: Some("Get weather".into()),
                    parameters: Some(serde_json::json!({"type": "object"})),
                },
                ResponsesTool::WebSearch {
                    extra: serde_json::Map::new(),
                },
            ]),
            tool_choice: None,
            previous_response_id: None,
            store: None,
            stream: None,
            temperature: None,
            max_output_tokens: None,
            top_p: None,
            extra: serde_json::Map::new(),
        };

        let result = decode_responses_request(&req, uuid::Uuid::from_u128(1)).unwrap();
        assert_eq!(result.tools.len(), 1);
        assert_eq!(result.tools[0].function.name, "get_weather");
        assert!(result
            .provider_extensions_ref()
            .contains_key("responses_builtin_tools"));
    }

    #[test]
    fn test_decode_owned_string_input() {
        let req = ResponsesRequest {
            model: "gpt-4o".into(),
            input: serde_json::Value::String("Hello".into()),
            instructions: Some("You are helpful.".into()),
            tools: None,
            tool_choice: None,
            previous_response_id: None,
            store: Some(true),
            stream: Some(true),
            temperature: Some(0.7),
            max_output_tokens: Some(1024),
            top_p: Some(0.9),
            extra: serde_json::Map::new(),
        };

        let result = decode_responses_request_owned(req, uuid::Uuid::from_u128(1)).unwrap();
        assert_eq!(result.model, "gpt-4o");
        assert!(result.stream);
        assert_eq!(result.system_prompt.as_deref(), Some("You are helpful."));
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, CanonicalRole::User);
        assert_eq!(result.generation.temperature, Some(0.7));
        assert_eq!(result.generation.max_tokens, Some(1024));
        assert_eq!(result.generation.top_p, Some(0.9));
        assert_eq!(
            result.provider_extensions_ref().get("store"),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[test]
    fn test_decode_function_call_item() {
        let req = ResponsesRequest {
            model: "gpt-4o".into(),
            input: serde_json::json!([
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"SF\"}"
                }
            ]),
            instructions: None,
            tools: None,
            tool_choice: None,
            previous_response_id: None,
            store: None,
            stream: None,
            temperature: None,
            max_output_tokens: None,
            top_p: None,
            extra: serde_json::Map::new(),
        };

        let result = decode_responses_request(&req, uuid::Uuid::from_u128(1)).unwrap();
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, CanonicalRole::Assistant);
        assert!(matches!(
            result.messages[0].parts.first(),
            Some(CanonicalPart::ToolCall { id, name, .. })
                if id == "call_123" && name == "get_weather"
        ));
    }

    #[test]
    fn test_decode_tool_choice_and_extra_passthrough() {
        let mut extra = serde_json::Map::new();
        extra.insert("parallel_tool_calls".into(), serde_json::Value::Bool(false));
        let req = ResponsesRequest {
            model: "gpt-4o".into(),
            input: serde_json::Value::String("Hello".into()),
            instructions: None,
            tools: None,
            tool_choice: Some(serde_json::json!({
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type":"function","name":"get_weather"}]
            })),
            previous_response_id: None,
            store: None,
            stream: None,
            temperature: None,
            max_output_tokens: None,
            top_p: None,
            extra,
        };

        let result = decode_responses_request(&req, uuid::Uuid::from_u128(1)).unwrap();
        assert_eq!(
            result.tool_choice,
            CanonicalToolChoice::Specific("get_weather".to_string())
        );
        assert_eq!(
            result.provider_extensions_ref().get("parallel_tool_calls"),
            Some(&serde_json::Value::Bool(false))
        );
    }
}
