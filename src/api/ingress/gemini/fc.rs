use std::collections::{HashMap, VecDeque};

use crate::config::FeaturesConfig;
use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::canonical::{CanonicalToolChoice, CanonicalToolFunction, CanonicalToolSpec};
use crate::protocol::gemini::{
    GeminiContent, GeminiPart, GeminiRequest, GeminiToolConfig, GeminiToolDeclaration,
};

pub(crate) fn apply_fc_inject_gemini_wire(
    request: &mut GeminiRequest,
    features: &FeaturesConfig,
) -> Result<Vec<CanonicalToolSpec>, CanonicalError> {
    let tool_choice = decode_gemini_wire_tool_choice(request.tool_config.as_ref());
    if matches!(tool_choice, CanonicalToolChoice::None) {
        return Ok(Vec::new());
    }

    let saved_tools = decode_gemini_wire_tools(request.tools.take());
    let fc_prompt = fc::prompt::generate_fc_prompt(
        &saved_tools,
        &tool_choice,
        features.prompt_template.as_deref(),
    )?;

    let mut call_args_by_name: HashMap<String, VecDeque<String>> = HashMap::new();
    for content in &request.contents {
        if content.role.as_deref() != Some("model") {
            continue;
        }
        for part in &content.parts {
            if let GeminiPart::FunctionCall { name, args } = part {
                call_args_by_name
                    .entry(name.clone())
                    .or_default()
                    .push_back(serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string()));
            }
        }
    }

    let existing_system = extract_gemini_system_text(request.system_instruction.as_ref());
    let system_prompt = if existing_system.is_empty() {
        fc_prompt
    } else {
        format!("{existing_system}\n{fc_prompt}")
    };
    request.system_instruction = Some(GeminiContent {
        role: None,
        parts: vec![GeminiPart::Text(system_prompt)],
    });

    let mut transformed: Vec<GeminiContent> = Vec::with_capacity(request.contents.len());
    for mut content in std::mem::take(&mut request.contents) {
        match content.role.as_deref() {
            Some("model") => {
                let original_parts = std::mem::take(&mut content.parts);
                let mut passthrough_parts: Vec<GeminiPart> = Vec::new();
                let mut model_text = String::new();
                let mut has_tool_calls = false;
                let mut formatted_tool_calls = String::new();
                formatted_tool_calls.push_str(fc::prompt::get_trigger_signal());
                formatted_tool_calls.push_str("\n<function_calls>\n");

                for part in original_parts {
                    match part {
                        GeminiPart::Text(text) => {
                            model_text.push_str(&text);
                            passthrough_parts.push(GeminiPart::Text(text));
                        }
                        GeminiPart::FunctionCall { name, args } => {
                            has_tool_calls = true;
                            let args_json =
                                serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                            let xml_call = format!(
                                "<function_call>\n\
                                 <tool>{name}</tool>\n\
                                 <args_json>{cdata}</args_json>\n\
                                 </function_call>\n",
                                cdata = wrap_cdata(args_json.as_str()),
                            );
                            formatted_tool_calls.push_str(&xml_call);
                        }
                        other => passthrough_parts.push(other),
                    }
                }

                if has_tool_calls {
                    formatted_tool_calls.push_str("</function_calls>");
                    let mut final_text = String::new();
                    if !model_text.is_empty() {
                        final_text.push_str(&model_text);
                        final_text.push('\n');
                    }
                    final_text.push_str(&formatted_tool_calls);
                    content.parts = vec![GeminiPart::Text(final_text.trim().to_string())];
                } else {
                    content.parts = passthrough_parts;
                }
                transformed.push(content);
            }
            Some("function") => {
                let original_parts = std::mem::take(&mut content.parts);
                let mut passthrough_parts: Vec<GeminiPart> = Vec::new();
                let mut text_chunks: Vec<String> = Vec::new();
                let mut formatted_results: Vec<String> = Vec::new();
                let mut has_tool_results = false;

                for part in original_parts {
                    match part {
                        GeminiPart::Text(text) => {
                            if !text.is_empty() {
                                text_chunks.push(text.clone());
                            }
                            passthrough_parts.push(GeminiPart::Text(text));
                        }
                        GeminiPart::FunctionResponse { name, response } => {
                            has_tool_results = true;
                            let tool_arguments = call_args_by_name
                                .get_mut(name.as_str())
                                .and_then(VecDeque::pop_front)
                                .unwrap_or_else(|| "{}".to_string());
                            let response_text = serde_json::to_string(&response)
                                .unwrap_or_else(|_| "{}".to_string());
                            formatted_results.push(format!(
                                "Tool execution result:\n\
                                 - Tool name: {name}\n\
                                 - Tool arguments: {tool_arguments}\n\
                                 - Execution result:\n\
                                 <tool_result>\n\
                                 {response_text}\n\
                                 </tool_result>"
                            ));
                        }
                        other => passthrough_parts.push(other),
                    }
                }

                if has_tool_results {
                    let mut final_chunks: Vec<String> = Vec::new();
                    if !text_chunks.is_empty() {
                        final_chunks.push(text_chunks.join("\n"));
                    }
                    final_chunks.extend(formatted_results);
                    content.role = Some("user".to_string());
                    content.parts = vec![GeminiPart::Text(final_chunks.join("\n\n"))];
                } else {
                    content.parts = passthrough_parts;
                }
                transformed.push(content);
            }
            _ => transformed.push(content),
        }
    }

    request.contents = transformed;
    request.tools = None;
    request.tool_config = None;
    Ok(saved_tools)
}

fn decode_gemini_wire_tools(tools: Option<Vec<GeminiToolDeclaration>>) -> Vec<CanonicalToolSpec> {
    match tools {
        None => Vec::new(),
        Some(items) => items
            .into_iter()
            .flat_map(|tool| {
                tool.function_declarations
                    .into_iter()
                    .map(|function| CanonicalToolSpec {
                        function: CanonicalToolFunction {
                            name: function.name,
                            description: function.description,
                            parameters: function
                                .parameters
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                        },
                    })
            })
            .collect(),
    }
}

fn decode_gemini_wire_tool_choice(tool_config: Option<&GeminiToolConfig>) -> CanonicalToolChoice {
    let Some(tool_config) = tool_config else {
        return CanonicalToolChoice::Auto;
    };
    let Some(function_calling) = tool_config.function_calling_config.as_ref() else {
        return CanonicalToolChoice::Auto;
    };
    match function_calling.mode.as_deref().unwrap_or("AUTO") {
        "NONE" => CanonicalToolChoice::None,
        "ANY" => {
            if let Some(allowed) = function_calling.allowed_function_names.as_ref() {
                if allowed.len() == 1 {
                    return CanonicalToolChoice::Specific(allowed[0].clone());
                }
            }
            CanonicalToolChoice::Required
        }
        _ => CanonicalToolChoice::Auto,
    }
}

fn extract_gemini_system_text(system_instruction: Option<&GeminiContent>) -> String {
    let Some(system_instruction) = system_instruction else {
        return String::new();
    };
    system_instruction
        .parts
        .iter()
        .filter_map(|part| match part {
            GeminiPart::Text(text) => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn wrap_cdata(text: &str) -> String {
    let safe = text.replace("]]>", "]]]]><![CDATA[>");
    format!("<![CDATA[{safe}]]>")
}
