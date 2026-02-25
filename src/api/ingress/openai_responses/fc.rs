use rustc_hash::FxHashMap;
use serde::Deserialize;

use crate::config::FeaturesConfig;
use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::canonical::{CanonicalToolChoice, CanonicalToolFunction, CanonicalToolSpec};
use crate::protocol::openai_responses::{ResponsesRequest, ResponsesTool};

pub(crate) fn responses_tools_token_has_function(token: &[u8]) -> bool {
    #[derive(Deserialize)]
    struct ToolTypeProbe<'a> {
        #[serde(borrow, default)]
        r#type: Option<std::borrow::Cow<'a, str>>,
    }

    let Ok(items) = serde_json::from_slice::<Vec<ToolTypeProbe<'_>>>(token) else {
        // Parse failure should not cause false negatives in FC decision.
        return true;
    };
    items.iter().any(|item| {
        item.r#type
            .as_deref()
            .is_some_and(|tool_type| tool_type == "function")
    })
}

pub(crate) fn apply_fc_inject_responses_wire(
    request: &mut ResponsesRequest,
    features: &FeaturesConfig,
) -> Result<Vec<CanonicalToolSpec>, CanonicalError> {
    if responses_prefers_structured_output(request) {
        return Ok(Vec::new());
    }

    let tool_choice = decode_responses_wire_tool_choice(request.tool_choice.as_ref());
    if matches!(tool_choice, CanonicalToolChoice::None) {
        return Ok(Vec::new());
    }

    let (saved_tools, passthrough_tools) = decode_responses_wire_tools(request.tools.take());
    if saved_tools.is_empty() {
        request.tools = if passthrough_tools.is_empty() {
            None
        } else {
            Some(passthrough_tools)
        };
        return Ok(saved_tools);
    }
    let fc_prompt = fc::prompt::generate_fc_prompt(
        &saved_tools,
        &tool_choice,
        features.prompt_template.as_deref(),
    )?;

    request.instructions = Some(match request.instructions.take() {
        Some(existing) => format!("{existing}\n{fc_prompt}"),
        None => fc_prompt,
    });
    request.input = preprocess_responses_wire_input(std::mem::take(&mut request.input))?;
    request.tools = if passthrough_tools.is_empty() {
        None
    } else {
        Some(passthrough_tools)
    };
    request.tool_choice = None;

    Ok(saved_tools)
}

#[inline]
fn responses_prefers_structured_output(request: &ResponsesRequest) -> bool {
    request
        .extra
        .get("response_format")
        .is_some_and(response_format_is_json_mode)
        || request
            .extra
            .get("text")
            .is_some_and(responses_text_format_is_json_mode)
}

#[inline]
fn responses_text_format_is_json_mode(value: &serde_json::Value) -> bool {
    value
        .as_object()
        .and_then(|obj| obj.get("format"))
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

fn decode_responses_wire_tool_choice(
    tool_choice: Option<&serde_json::Value>,
) -> CanonicalToolChoice {
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
            if obj.get("type").and_then(serde_json::Value::as_str) == Some("allowed_tools")
                && obj
                    .get("mode")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("auto")
                    == "required"
            {
                if let Some(name) = obj
                    .get("tools")
                    .and_then(serde_json::Value::as_array)
                    .and_then(|tools| {
                        if tools.len() != 1 {
                            return None;
                        }
                        let tool = tools.first()?.as_object()?;
                        if tool.get("type").and_then(serde_json::Value::as_str) != Some("function")
                        {
                            return None;
                        }
                        tool.get("name")
                            .and_then(serde_json::Value::as_str)
                            .map(std::string::ToString::to_string)
                    })
                {
                    return CanonicalToolChoice::Specific(name);
                }
                return CanonicalToolChoice::Required;
            }

            if obj.get("type").and_then(serde_json::Value::as_str) == Some("function") {
                return obj
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .map_or(CanonicalToolChoice::Auto, |name| {
                        CanonicalToolChoice::Specific(name.to_string())
                    });
            }

            CanonicalToolChoice::Auto
        }
        _ => CanonicalToolChoice::Auto,
    }
}

fn decode_responses_wire_tools(
    tools: Option<Vec<ResponsesTool>>,
) -> (Vec<CanonicalToolSpec>, Vec<ResponsesTool>) {
    let Some(items) = tools else {
        return (Vec::new(), Vec::new());
    };

    let mut saved_tools: Vec<CanonicalToolSpec> = Vec::new();
    let mut passthrough_tools: Vec<ResponsesTool> = Vec::new();

    for tool in items {
        match tool {
            ResponsesTool::Function {
                name,
                description,
                parameters,
            } => saved_tools.push(CanonicalToolSpec {
                function: CanonicalToolFunction {
                    name,
                    description,
                    parameters: parameters
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                },
            }),
            other => passthrough_tools.push(other),
        }
    }

    (saved_tools, passthrough_tools)
}

pub(crate) fn preprocess_responses_wire_input(
    input: serde_json::Value,
) -> Result<serde_json::Value, CanonicalError> {
    match input {
        serde_json::Value::String(text) => Ok(serde_json::Value::String(text)),
        serde_json::Value::Array(items) => {
            let mut call_index: FxHashMap<String, (String, String)> = FxHashMap::default();
            for item in &items {
                let serde_json::Value::Object(obj) = item else {
                    continue;
                };
                if obj.get("type").and_then(serde_json::Value::as_str) != Some("function_call") {
                    continue;
                }
                let Some(call_id) = obj
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .filter(|id| !id.is_empty())
                else {
                    continue;
                };
                let name = obj
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                let args_json = responses_arguments_to_json_string(obj.get("arguments"));
                call_index.insert(call_id.to_string(), (name, args_json));
            }

            let mut transformed: Vec<serde_json::Value> = Vec::with_capacity(items.len());

            for item in items {
                let serde_json::Value::Object(mut obj) = item else {
                    continue;
                };

                let item_type = obj
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("message");

                match item_type {
                    "message" => transformed.push(serde_json::Value::Object(obj)),
                    "function_call" => {
                        let call_id = obj
                            .remove("call_id")
                            .and_then(|v| v.as_str().map(std::string::ToString::to_string))
                            .unwrap_or_default();
                        let name = obj
                            .remove("name")
                            .and_then(|v| v.as_str().map(std::string::ToString::to_string))
                            .unwrap_or_else(|| "unknown".to_string());
                        let args_json = responses_arguments_to_json_string(obj.get("arguments"));
                        let call_id_line = if call_id.is_empty() {
                            String::new()
                        } else {
                            format!("  <id>{call_id}</id>\n")
                        };
                        let xml_tool_call = format!(
                            "{trigger}\n\
                             <function_calls>\n\
                             <function_call>\n\
                             {call_id_line}\
                             <tool>{name}</tool>\n\
                             <args_json>{args}</args_json>\n\
                             </function_call>\n\
                             </function_calls>",
                            trigger = fc::prompt::get_trigger_signal(),
                            args = wrap_cdata(args_json.as_str()),
                        );
                        transformed.push(serde_json::json!({
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": xml_tool_call
                                }
                            ]
                        }));
                    }
                    "function_call_output" => {
                        let call_id = obj
                            .remove("call_id")
                            .and_then(|v| v.as_str().map(std::string::ToString::to_string))
                            .unwrap_or_default();
                        let (tool_name, tool_arguments) = call_index
                            .get(call_id.as_str())
                            .map_or(("unknown", "{}"), |(name, args)| {
                                (name.as_str(), args.as_str())
                            });
                        let output = obj
                            .remove("output")
                            .and_then(|v| v.as_str().map(std::string::ToString::to_string))
                            .unwrap_or_default();
                        let formatted = format!(
                            "Tool execution result:\n\
                             - Tool name: {tool_name}\n\
                             - Tool arguments: {tool_arguments}\n\
                             - Execution result:\n\
                             <tool_result>\n\
                             {output}\n\
                             </tool_result>"
                        );
                        transformed.push(serde_json::json!({
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": formatted
                                }
                            ]
                        }));
                    }
                    _ => {
                        return Err(CanonicalError::InvalidRequest(format!(
                            "Unknown Responses API input item type: {item_type}"
                        )));
                    }
                }
            }

            Ok(serde_json::Value::Array(transformed))
        }
        _ => Err(CanonicalError::InvalidRequest(
            "Responses API `input` must be a string or array".into(),
        )),
    }
}

fn responses_arguments_to_json_string(value: Option<&serde_json::Value>) -> String {
    match value {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
        None => "{}".to_string(),
    }
}

fn wrap_cdata(text: &str) -> String {
    let safe = text.replace("]]>", "]]]]><![CDATA[>");
    format!("<![CDATA[{safe}]]>")
}
