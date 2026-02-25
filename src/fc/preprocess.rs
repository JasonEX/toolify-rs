use std::collections::HashMap;

use crate::protocol::canonical::{CanonicalMessage, CanonicalPart, CanonicalRole};

use super::prompt;

/// Build an index mapping `tool_call_id` -> (name, `arguments_json`) from
/// assistant messages in the conversation history.
fn build_tool_call_index(messages: &[CanonicalMessage]) -> HashMap<String, (String, String)> {
    let mut index: HashMap<String, (String, String)> = HashMap::new();
    for msg in messages {
        if msg.role != CanonicalRole::Assistant {
            continue;
        }
        for part in &msg.parts {
            if let CanonicalPart::ToolCall {
                id,
                name,
                arguments,
            } = part
            {
                index.insert(id.clone(), (name.clone(), arguments.get().to_string()));
            }
        }
    }
    index
}

/// Wrap text in CDATA, escaping any `]]>` sequences inside.
fn wrap_cdata(text: &str) -> String {
    let safe = text.replace("]]>", "]]]]><![CDATA[>");
    format!("<![CDATA[{safe}]]>")
}

/// Preprocess messages for FC inject mode.
///
/// - Convert `role=Tool` messages to `role=User` with formatted text content.
/// - Convert assistant messages containing `ToolCall` parts to XML format.
/// - If `convert_developer_to_system`, convert developer messages to system.
/// - Text content passes through unchanged.
#[must_use]
pub fn preprocess_messages(
    messages: &[CanonicalMessage],
    convert_developer_to_system: bool,
) -> Vec<CanonicalMessage> {
    preprocess_messages_owned(messages.to_vec(), convert_developer_to_system)
}

/// Same behavior as [`preprocess_messages`], but consumes message ownership so
/// unchanged messages can be moved through without cloning.
#[must_use]
pub fn preprocess_messages_owned(
    messages: Vec<CanonicalMessage>,
    _convert_developer_to_system: bool,
) -> Vec<CanonicalMessage> {
    let tool_call_index = build_tool_call_index(&messages);
    let trigger_signal = prompt::get_trigger_signal();

    let mut result: Vec<CanonicalMessage> = Vec::with_capacity(messages.len());

    for mut msg in messages {
        match msg.role {
            CanonicalRole::Tool => {
                // Convert Tool messages to User messages with formatted text.
                // Extract the tool_call_id and content from the message.
                let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("");

                // Collect text content from the tool result parts
                let content = collect_text_like_parts(&msg.parts, true);

                // Look up the tool info from the index
                let (tool_name, tool_arguments) = tool_call_index
                    .get(tool_call_id)
                    .map_or(("unknown", "{}"), |(name, arguments)| {
                        (name.as_str(), arguments.as_str())
                    });

                // Format like Python's format_tool_result_for_ai
                let formatted = format!(
                    "Tool execution result:\n\
                     - Tool name: {tool_name}\n\
                     - Tool arguments: {tool_arguments}\n\
                     - Execution result:\n\
                     <tool_result>\n\
                     {content}\n\
                     </tool_result>"
                );

                msg.role = CanonicalRole::User;
                msg.parts.clear();
                msg.parts.push(CanonicalPart::Text(formatted));
                msg.name = None;
                msg.tool_call_id = None;
                msg.provider_extensions = None;
                result.push(msg);
            }

            CanonicalRole::Assistant => {
                // Check if this assistant message has any ToolCall parts.
                let has_tool_calls = msg
                    .parts
                    .iter()
                    .any(|p| matches!(p, CanonicalPart::ToolCall { .. }));

                if has_tool_calls {
                    // Collect original text content
                    let original_text = collect_text_like_parts(&msg.parts, false);

                    // Convert tool calls to XML format.
                    let mut formatted_tool_calls = String::new();
                    formatted_tool_calls.push_str(trigger_signal);
                    formatted_tool_calls.push_str("\n<function_calls>\n");
                    for part in &msg.parts {
                        if let CanonicalPart::ToolCall {
                            id,
                            name,
                            arguments,
                        } = part
                        {
                            // arguments is RawValue — get the raw JSON string
                            let args_json = arguments.get();
                            let xml_call = format!(
                                "<function_call>\n\
                                 <id>{id}</id>\n\
                                 <tool>{name}</tool>\n\
                                 <args_json>{cdata}</args_json>\n\
                                 </function_call>",
                                cdata = wrap_cdata(args_json),
                            );
                            formatted_tool_calls.push_str(&xml_call);
                            formatted_tool_calls.push('\n');
                        }
                    }
                    formatted_tool_calls.push_str("</function_calls>");

                    let mut final_content =
                        String::with_capacity(original_text.len() + formatted_tool_calls.len() + 1);
                    if !original_text.is_empty() {
                        final_content.push_str(&original_text);
                        final_content.push('\n');
                    }
                    final_content.push_str(&formatted_tool_calls);

                    msg.parts.clear();
                    msg.parts
                        .push(CanonicalPart::Text(final_content.trim().to_string()));
                    msg.tool_call_id = None;
                    msg.provider_extensions = None;
                    result.push(msg);
                } else {
                    // No tool calls — pass through.
                    result.push(msg);
                }
            }

            // System and User messages pass through unchanged.
            _ => {
                result.push(msg);
            }
        }
    }

    result
}

fn collect_text_like_parts(parts: &[CanonicalPart], include_tool_result: bool) -> String {
    let mut content = String::new();
    for part in parts {
        match part {
            CanonicalPart::Text(text) => content.push_str(text),
            CanonicalPart::ToolResult { content: text, .. } if include_tool_result => {
                content.push_str(text);
            }
            _ => {}
        }
    }
    content
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::CanonicalRole;

    fn make_message(role: CanonicalRole, text: &str) -> CanonicalMessage {
        CanonicalMessage {
            role,
            parts: vec![CanonicalPart::Text(text.to_string())].into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        }
    }

    fn make_tool_call_part(id: &str, name: &str, args_json: &str) -> CanonicalPart {
        CanonicalPart::ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: serde_json::value::RawValue::from_string(args_json.to_string()).unwrap(),
        }
    }

    #[test]
    fn test_preprocess_text_passthrough() {
        let messages = vec![
            make_message(CanonicalRole::User, "hello"),
            make_message(CanonicalRole::Assistant, "world"),
        ];
        let result = preprocess_messages(&messages, true);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].role, CanonicalRole::User);
        assert_eq!(result[1].role, CanonicalRole::Assistant);
    }

    #[test]
    fn test_preprocess_tool_to_user() {
        let assistant_msg = CanonicalMessage {
            role: CanonicalRole::Assistant,
            parts: vec![make_tool_call_part(
                "call_123",
                "get_weather",
                r#"{"city": "London"}"#,
            )]
            .into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        };

        let tool_msg = CanonicalMessage {
            role: CanonicalRole::Tool,
            parts: vec![CanonicalPart::Text("Sunny, 22C".to_string())].into(),
            name: None,
            tool_call_id: Some("call_123".to_string()),
            provider_extensions: None,
        };

        let messages = vec![assistant_msg, tool_msg];
        let result = preprocess_messages(&messages, true);

        assert_eq!(result.len(), 2);
        assert_eq!(result[1].role, CanonicalRole::User);

        if let CanonicalPart::Text(t) = &result[1].parts[0] {
            assert!(t.contains("Tool execution result:"));
            assert!(t.contains("get_weather"));
            assert!(t.contains("Sunny, 22C"));
        } else {
            panic!("expected Text part");
        }
    }

    #[test]
    fn test_preprocess_assistant_tool_calls_to_xml() {
        let msg = CanonicalMessage {
            role: CanonicalRole::Assistant,
            parts: vec![
                CanonicalPart::Text("Let me check.".to_string()),
                make_tool_call_part("call_1", "search", r#"{"query": "test"}"#),
            ]
            .into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        };

        let result = preprocess_messages(&[msg], true);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, CanonicalRole::Assistant);

        if let CanonicalPart::Text(t) = &result[0].parts[0] {
            assert!(t.contains("Let me check."));
            assert!(t.contains("<function_calls>"));
            assert!(t.contains("<id>call_1</id>"));
            assert!(t.contains("<tool>search</tool>"));
            assert!(t.contains(prompt::get_trigger_signal()));
        } else {
            panic!("expected Text part");
        }
    }

    #[test]
    fn test_wrap_cdata_simple() {
        assert_eq!(wrap_cdata("hello"), "<![CDATA[hello]]>");
    }

    #[test]
    fn test_wrap_cdata_escapes_closing() {
        let result = wrap_cdata("a]]>b");
        assert_eq!(result, "<![CDATA[a]]]]><![CDATA[>b]]>");
    }

    #[test]
    fn test_build_tool_call_index() {
        let msg = CanonicalMessage {
            role: CanonicalRole::Assistant,
            parts: vec![make_tool_call_part(
                "call_abc",
                "search",
                r#"{"q": "test"}"#,
            )]
            .into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        };
        let index = build_tool_call_index(&[msg]);
        assert!(index.contains_key("call_abc"));
        let (name, args) = index.get("call_abc").unwrap();
        assert_eq!(name, "search");
        assert!(args.contains("test"));
    }
}
