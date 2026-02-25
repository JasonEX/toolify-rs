mod cache;
mod raw_inject;

use std::collections::HashMap;

use serde_json::Value;

use crate::config::FeaturesConfig;
use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::canonical::{CanonicalToolChoice, CanonicalToolFunction, CanonicalToolSpec};
use crate::protocol::openai_chat::{
    OpenAiChatRequest, OpenAiMessage, OpenAiTool, OpenAiToolChoice,
};

pub(crate) type OpenAiSimpleInjectBuild = raw_inject::OpenAiSimpleInjectBuild;

pub(crate) fn try_build_openai_simple_fc_inject_body_from_raw(
    body: &bytes::Bytes,
    actual_model: &str,
    features: &FeaturesConfig,
    ranges_hint: Option<&crate::api::common::CommonProbeRanges>,
) -> Result<Option<OpenAiSimpleInjectBuild>, CanonicalError> {
    raw_inject::try_build_openai_simple_fc_inject_body_from_raw(
        body,
        actual_model,
        features,
        ranges_hint,
    )
}

#[cfg(test)]
pub(crate) fn build_openai_simple_inject_json_body(
    body: &[u8],
    actual_model: &str,
    system_msg_bytes: &[u8],
    prevalidated_messages_inner_bounds: Option<(usize, usize)>,
) -> Result<Option<bytes::Bytes>, CanonicalError> {
    raw_inject::build_openai_simple_inject_json_body(
        body,
        actual_model,
        system_msg_bytes,
        prevalidated_messages_inner_bounds,
    )
}

#[cfg(test)]
pub(crate) fn messages_inner_bounds_if_simple(messages_token: &[u8]) -> Option<(usize, usize)> {
    raw_inject::messages_inner_bounds_if_simple(messages_token)
}

pub(crate) fn apply_fc_inject_openai_wire(
    request: &mut OpenAiChatRequest,
    features: &FeaturesConfig,
) -> Result<Vec<CanonicalToolSpec>, CanonicalError> {
    if openai_chat_prefers_structured_output(request) {
        return Ok(Vec::new());
    }

    let tool_choice = decode_openai_wire_tool_choice(request.tool_choice.as_ref());
    if matches!(tool_choice, CanonicalToolChoice::None) {
        return Ok(Vec::new());
    }

    let saved_tools = decode_openai_wire_tools(request.tools.take());
    let fc_prompt_artifacts = fc::prompt::generate_fc_prompt_artifacts(
        &saved_tools,
        &tool_choice,
        features.prompt_template.as_deref(),
    )?;
    let fc_prompt = fc_prompt_artifacts.prompt();

    if is_simple_fc_inject_openai_request(&request.messages) {
        request.messages.insert(
            0,
            OpenAiMessage {
                role: "system".to_string(),
                content: Some(Value::String(fc_prompt.to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
            },
        );
        request.tools = None;
        request.tool_choice = None;
        return Ok(saved_tools);
    }

    let mut tool_call_index: HashMap<String, (String, String)> = HashMap::new();
    if request.messages.iter().any(|msg| msg.role == "tool") {
        for msg in &request.messages {
            if msg.role != "assistant" {
                continue;
            }
            if let Some(tool_calls) = &msg.tool_calls {
                for call in tool_calls {
                    tool_call_index.insert(
                        call.id.clone(),
                        (call.function.name.clone(), call.function.arguments.clone()),
                    );
                }
            }
        }
    }

    let mut system_parts: Vec<String> = Vec::new();
    let mut transformed: Vec<OpenAiMessage> = Vec::with_capacity(request.messages.len() + 1);
    for mut msg in std::mem::take(&mut request.messages) {
        match msg.role.as_str() {
            "system" => {
                let text = extract_openai_message_text(msg.content.as_ref());
                if !text.is_empty() {
                    system_parts.push(text);
                }
            }
            "tool" => {
                let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("");
                let (tool_name, tool_arguments) = tool_call_index
                    .get(tool_call_id)
                    .map_or(("unknown", "{}"), |(name, args)| {
                        (name.as_str(), args.as_str())
                    });
                let content = extract_openai_message_text(msg.content.as_ref());
                let formatted = format!(
                    "Tool execution result:\n\
                     - Tool name: {tool_name}\n\
                     - Tool arguments: {tool_arguments}\n\
                     - Execution result:\n\
                     <tool_result>\n\
                     {content}\n\
                     </tool_result>"
                );

                msg.role = "user".to_string();
                msg.content = Some(Value::String(formatted));
                msg.name = None;
                msg.tool_calls = None;
                msg.tool_call_id = None;
                msg.refusal = None;
                transformed.push(msg);
            }
            "assistant" => {
                if let Some(tool_calls) = msg.tool_calls.take() {
                    if !tool_calls.is_empty() {
                        let original_text = extract_openai_message_text(msg.content.as_ref());
                        let mut formatted_tool_calls = String::new();
                        formatted_tool_calls.push_str(fc::prompt::get_trigger_signal());
                        formatted_tool_calls.push_str("\n<function_calls>\n");
                        for call in tool_calls {
                            let xml_call = format!(
                                "<function_call>\n\
                                 <id>{id}</id>\n\
                                 <tool>{name}</tool>\n\
                                 <args_json>{cdata}</args_json>\n\
                                 </function_call>\n",
                                id = call.id,
                                name = call.function.name,
                                cdata = wrap_cdata(call.function.arguments.as_str()),
                            );
                            formatted_tool_calls.push_str(&xml_call);
                        }
                        formatted_tool_calls.push_str("</function_calls>");

                        let mut final_content = String::with_capacity(
                            original_text.len() + formatted_tool_calls.len() + 1,
                        );
                        if !original_text.is_empty() {
                            final_content.push_str(&original_text);
                            final_content.push('\n');
                        }
                        final_content.push_str(&formatted_tool_calls);

                        msg.content = Some(Value::String(final_content.trim().to_string()));
                        msg.tool_call_id = None;
                        msg.refusal = None;
                    }
                }
                transformed.push(msg);
            }
            _ => transformed.push(msg),
        }
    }

    let system_prompt = if system_parts.is_empty() {
        fc_prompt.to_string()
    } else {
        format!("{}\n{fc_prompt}", system_parts.join("\n"))
    };
    transformed.insert(
        0,
        OpenAiMessage {
            role: "system".to_string(),
            content: Some(Value::String(system_prompt)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
        },
    );

    request.messages = transformed;
    request.tools = None;
    request.tool_choice = None;
    Ok(saved_tools)
}

#[inline]
fn openai_chat_prefers_structured_output(request: &OpenAiChatRequest) -> bool {
    request
        .extra
        .get("response_format")
        .is_some_and(response_format_is_json_mode)
}

#[inline]
fn response_format_is_json_mode(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(mode) => {
            mode.eq_ignore_ascii_case("json_object") || mode.eq_ignore_ascii_case("json_schema")
        }
        serde_json::Value::Object(obj) => obj
            .get("type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|mode| {
                mode.eq_ignore_ascii_case("json_object") || mode.eq_ignore_ascii_case("json_schema")
            }),
        _ => false,
    }
}

pub(super) fn response_format_token_is_json_mode(token: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(token)
        .ok()
        .as_ref()
        .is_some_and(response_format_is_json_mode)
}

pub(super) fn decode_openai_wire_tools(tools: Option<Vec<OpenAiTool>>) -> Vec<CanonicalToolSpec> {
    match tools {
        None => Vec::new(),
        Some(items) => items
            .into_iter()
            .map(|tool| CanonicalToolSpec {
                function: CanonicalToolFunction {
                    name: tool.function.name,
                    description: tool.function.description,
                    parameters: tool
                        .function
                        .parameters
                        .unwrap_or(Value::Object(serde_json::Map::new())),
                },
            })
            .collect(),
    }
}

pub(super) fn decode_openai_wire_tool_choice(
    tool_choice: Option<&OpenAiToolChoice>,
) -> CanonicalToolChoice {
    match tool_choice {
        None => CanonicalToolChoice::Auto,
        Some(OpenAiToolChoice::Mode(mode)) => match mode.as_str() {
            "none" => CanonicalToolChoice::None,
            "required" => CanonicalToolChoice::Required,
            _ => CanonicalToolChoice::Auto,
        },
        Some(OpenAiToolChoice::Function(call)) => {
            CanonicalToolChoice::Specific(call.function.name.clone())
        }
    }
}

fn is_simple_fc_inject_openai_request(messages: &[OpenAiMessage]) -> bool {
    messages.iter().all(|msg| {
        msg.role != "system"
            && msg.role != "tool"
            && msg.tool_calls.as_ref().is_none_or(Vec::is_empty)
    })
}

fn extract_openai_message_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(items)) => {
            let mut text = String::new();
            for item in items {
                if item.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(part) = item.get("text").and_then(Value::as_str) {
                        text.push_str(part);
                    }
                }
            }
            text
        }
        _ => String::new(),
    }
}

fn wrap_cdata(text: &str) -> String {
    let safe = text.replace("]]>", "]]]]><![CDATA[>");
    format!("<![CDATA[{safe}]]>")
}

pub(super) fn parse_openai_tools_token(
    token: &[u8],
) -> Result<Option<Vec<OpenAiTool>>, CanonicalError> {
    serde_json::from_slice(token).map_err(|e| {
        CanonicalError::InvalidRequest(format!(
            "Invalid OpenAI Chat request body: tools field has invalid format: {e}"
        ))
    })
}

pub(super) fn parse_openai_tool_choice_token(
    token: &[u8],
) -> Result<Option<OpenAiToolChoice>, CanonicalError> {
    serde_json::from_slice(token).map_err(|e| {
        CanonicalError::InvalidRequest(format!("Invalid OpenAI Chat request body: {e}"))
    })
}
