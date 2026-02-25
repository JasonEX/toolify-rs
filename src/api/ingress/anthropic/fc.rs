use std::collections::HashMap;

use crate::config::FeaturesConfig;
use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::anthropic::{AnthropicMessage, AnthropicRequest, AnthropicTool};
use crate::protocol::canonical::{CanonicalToolChoice, CanonicalToolFunction, CanonicalToolSpec};

pub(crate) fn apply_fc_inject_anthropic_wire(
    request: &mut AnthropicRequest,
    features: &FeaturesConfig,
) -> Result<Vec<CanonicalToolSpec>, CanonicalError> {
    let tool_choice = decode_anthropic_wire_tool_choice(request.tool_choice.as_ref());
    if matches!(tool_choice, CanonicalToolChoice::None) {
        return Ok(Vec::new());
    }

    let saved_tools = decode_anthropic_wire_tools(request.tools.take());
    let fc_prompt = fc::prompt::generate_fc_prompt(
        &saved_tools,
        &tool_choice,
        features.prompt_template.as_deref(),
    )?;

    let mut tool_call_index: HashMap<String, (String, String)> = HashMap::new();
    for msg in &request.messages {
        if msg.role != "assistant" {
            continue;
        }
        let serde_json::Value::Array(blocks) = &msg.content else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(serde_json::Value::as_str) != Some("tool_use") {
                continue;
            }
            let id = block
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            if id.is_empty() {
                continue;
            }
            let name = block
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            let args_json = serde_json::to_string(
                block
                    .get("input")
                    .unwrap_or(&serde_json::Value::Object(serde_json::Map::new())),
            )
            .unwrap_or_else(|_| "{}".to_string());
            tool_call_index.insert(id, (name, args_json));
        }
    }

    let existing_system = extract_anthropic_system_text(request.system.as_ref());
    let system_prompt = if existing_system.is_empty() {
        fc_prompt
    } else {
        format!("{existing_system}\n{fc_prompt}")
    };
    request.system = Some(serde_json::Value::String(system_prompt));

    let mut transformed: Vec<AnthropicMessage> = Vec::with_capacity(request.messages.len());
    for mut msg in std::mem::take(&mut request.messages) {
        match msg.role.as_str() {
            "assistant" => {
                if let serde_json::Value::Array(blocks) = &msg.content {
                    let mut assistant_text = String::new();
                    let mut has_tool_calls = false;
                    let mut formatted_tool_calls = String::new();
                    formatted_tool_calls.push_str(fc::prompt::get_trigger_signal());
                    formatted_tool_calls.push_str("\n<function_calls>\n");

                    for block in blocks {
                        match block.get("type").and_then(serde_json::Value::as_str) {
                            Some("text") => {
                                if let Some(text) =
                                    block.get("text").and_then(serde_json::Value::as_str)
                                {
                                    assistant_text.push_str(text);
                                }
                            }
                            Some("tool_use") => {
                                has_tool_calls = true;
                                let name = block
                                    .get("name")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("unknown");
                                let call_id = block
                                    .get("id")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("");
                                let args_json =
                                    serde_json::to_string(block.get("input").unwrap_or(
                                        &serde_json::Value::Object(serde_json::Map::new()),
                                    ))
                                    .unwrap_or_else(|_| "{}".to_string());
                                let xml_call = format!(
                                    "<function_call>\n\
                                     <id>{call_id}</id>\n\
                                     <tool>{name}</tool>\n\
                                     <args_json>{cdata}</args_json>\n\
                                     </function_call>\n",
                                    cdata = wrap_cdata(args_json.as_str()),
                                );
                                formatted_tool_calls.push_str(&xml_call);
                            }
                            _ => {}
                        }
                    }

                    if has_tool_calls {
                        formatted_tool_calls.push_str("</function_calls>");
                        let mut final_content = String::new();
                        if !assistant_text.is_empty() {
                            final_content.push_str(&assistant_text);
                            final_content.push('\n');
                        }
                        final_content.push_str(&formatted_tool_calls);
                        msg.content = serde_json::Value::Array(vec![serde_json::json!({
                            "type": "text",
                            "text": final_content.trim(),
                        })]);
                    }
                }
                transformed.push(msg);
            }
            "user" => {
                if let serde_json::Value::Array(blocks) = &msg.content {
                    let mut has_tool_results = false;
                    let mut text_chunks: Vec<String> = Vec::new();
                    let mut formatted_results: Vec<String> = Vec::new();
                    for block in blocks {
                        match block.get("type").and_then(serde_json::Value::as_str) {
                            Some("text") => {
                                if let Some(text) =
                                    block.get("text").and_then(serde_json::Value::as_str)
                                {
                                    if !text.is_empty() {
                                        text_chunks.push(text.to_string());
                                    }
                                }
                            }
                            Some("tool_result") => {
                                has_tool_results = true;
                                let tool_use_id = block
                                    .get("tool_use_id")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("");
                                let (tool_name, tool_arguments) = tool_call_index
                                    .get(tool_use_id)
                                    .map_or(("unknown", "{}"), |(name, args)| {
                                        (name.as_str(), args.as_str())
                                    });
                                let content =
                                    extract_anthropic_tool_result_content(block.get("content"));
                                formatted_results.push(format!(
                                    "Tool execution result:\n\
                                     - Tool name: {tool_name}\n\
                                     - Tool arguments: {tool_arguments}\n\
                                     - Execution result:\n\
                                     <tool_result>\n\
                                     {content}\n\
                                     </tool_result>"
                                ));
                            }
                            _ => {}
                        }
                    }

                    if has_tool_results {
                        let mut final_chunks: Vec<String> = Vec::new();
                        if !text_chunks.is_empty() {
                            final_chunks.push(text_chunks.join("\n"));
                        }
                        final_chunks.extend(formatted_results);
                        msg.content = serde_json::Value::String(final_chunks.join("\n\n"));
                    }
                }
                transformed.push(msg);
            }
            _ => transformed.push(msg),
        }
    }

    request.messages = transformed;
    request.tools = None;
    request.tool_choice = None;
    Ok(saved_tools)
}

fn decode_anthropic_wire_tools(tools: Option<Vec<AnthropicTool>>) -> Vec<CanonicalToolSpec> {
    match tools {
        None => Vec::new(),
        Some(items) => items
            .into_iter()
            .map(|tool| CanonicalToolSpec {
                function: CanonicalToolFunction {
                    name: tool.name,
                    description: tool.description,
                    parameters: tool.input_schema,
                },
            })
            .collect(),
    }
}

fn decode_anthropic_wire_tool_choice(
    tool_choice: Option<&serde_json::Value>,
) -> CanonicalToolChoice {
    let Some(tool_choice) = tool_choice else {
        return CanonicalToolChoice::Auto;
    };
    match tool_choice
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("auto")
    {
        "none" => CanonicalToolChoice::None,
        "any" => CanonicalToolChoice::Required,
        "tool" => CanonicalToolChoice::Specific(
            tool_choice
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string(),
        ),
        _ => CanonicalToolChoice::Auto,
    }
}

fn extract_anthropic_system_text(system: Option<&serde_json::Value>) -> String {
    match system {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| {
                if b.get("type").and_then(serde_json::Value::as_str) == Some("text") {
                    b.get("text")
                        .and_then(serde_json::Value::as_str)
                        .map(std::string::ToString::to_string)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn extract_anthropic_tool_result_content(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| {
                if item.get("type").and_then(serde_json::Value::as_str) == Some("text") {
                    item.get("text")
                        .and_then(serde_json::Value::as_str)
                        .map(std::string::ToString::to_string)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
        None => String::new(),
    }
}

fn wrap_cdata(text: &str) -> String {
    let safe = text.replace("]]>", "]]]]><![CDATA[>");
    format!("<![CDATA[{safe}]]>")
}
