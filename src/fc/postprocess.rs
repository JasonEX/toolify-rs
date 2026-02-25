use std::borrow::Cow;
use std::sync::LazyLock;

use crate::error::CanonicalError;
use crate::protocol::canonical::{
    CanonicalPart, CanonicalResponse, CanonicalStopReason, CanonicalToolSpec,
};
use crate::util::next_call_id;

use super::{parser, prompt, validator};

static TRIGGER_SIGNAL_BYTES: LazyLock<&'static [u8]> =
    LazyLock::new(|| prompt::get_trigger_signal().as_bytes());
static TRIGGER_SIGNAL_FINDER: LazyLock<memchr::memmem::Finder<'static>> =
    LazyLock::new(|| memchr::memmem::Finder::new(*TRIGGER_SIGNAL_BYTES));
static TRIGGER_SIGNAL_FIRST_BYTE: LazyLock<Option<u8>> =
    LazyLock::new(|| TRIGGER_SIGNAL_BYTES.first().copied());

/// Result of processing an FC response from the model.
#[derive(Debug)]
pub enum FcResult {
    /// Successfully parsed tool calls, ready to be returned as canonical parts.
    ToolCalls {
        tool_parts: Vec<CanonicalPart>,
        text_before: Option<String>,
    },
    /// No trigger signal found — the response is plain text.
    NoToolCalls,
    /// Trigger was found but parsing or validation failed.
    ParseError {
        trigger_found: bool,
        error: String,
        original_text: String,
    },
}

/// Convert a `ParsedToolCall` from the parser into a `CanonicalPart::ToolCall`.
///
/// Reuses parsed `tool_call_id` when present, otherwise generates a monotonic
/// `call_{hex}` ID.
fn parsed_to_canonical_tool_call(
    parsed: parser::ParsedToolCall,
) -> Result<CanonicalPart, CanonicalError> {
    let parser::ParsedToolCall {
        id,
        name,
        arguments,
        arguments_json,
    } = parsed;
    let id = id.map_or_else(next_call_id, String::from);
    let raw_value = if let Some(arguments_json) = arguments_json {
        serde_json::value::RawValue::from_string(arguments_json.into())
            .or_else(|_| serde_json::value::to_raw_value(&arguments))
            .map_err(|e| {
                CanonicalError::FcParse(format!(
                    "failed to serialize tool call arguments to RawValue: {e}"
                ))
            })?
    } else {
        serde_json::value::to_raw_value(&arguments).map_err(|e| {
            CanonicalError::FcParse(format!(
                "failed to serialize tool call arguments to RawValue: {e}"
            ))
        })?
    };

    Ok(CanonicalPart::ToolCall {
        id,
        name,
        arguments: raw_value,
    })
}

/// Process the text response from an FC-injected model and extract tool calls.
///
/// 1. Check for trigger signal.
/// 2. Parse function calls from XML.
/// 3. Validate against tool specs.
/// 4. Convert to canonical parts.
///
/// # Errors
///
/// Returns [`CanonicalError`] only when converting validated parsed tool calls
/// into canonical raw JSON argument payloads fails.
pub fn process_fc_response(
    response_text: &str,
    tools: &[CanonicalToolSpec],
) -> Result<FcResult, CanonicalError> {
    let trigger_signal = prompt::get_trigger_signal();

    let Some(trigger_pos) = response_text.find(trigger_signal) else {
        return Ok(FcResult::NoToolCalls);
    };

    let parsed = match parser::parse_function_calls(response_text, trigger_signal) {
        Ok(calls) => calls,
        Err(e) => {
            return Ok(FcResult::ParseError {
                trigger_found: true,
                error: e.to_string(),
                original_text: response_text.to_string(),
            });
        }
    };

    if let Err(errors) = validator::validate_parser_tool_calls(&parsed, tools) {
        let error_messages: Vec<String> = errors
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        return Ok(FcResult::ParseError {
            trigger_found: true,
            error: error_messages.join("; "),
            original_text: response_text.to_string(),
        });
    }

    let mut canonical_parts: Vec<CanonicalPart> = Vec::with_capacity(parsed.len());
    for call in parsed {
        canonical_parts.push(parsed_to_canonical_tool_call(call)?);
    }

    let text_before = {
        let prefix = response_text[..trigger_pos].trim();
        if prefix.is_empty() {
            None
        } else {
            Some(prefix.to_string())
        }
    };

    Ok(FcResult::ToolCalls {
        tool_parts: canonical_parts,
        text_before,
    })
}

/// Concatenate all Text parts from a response's content.
#[must_use]
pub fn extract_response_text(content: &[CanonicalPart]) -> String {
    let mut text = String::new();
    for part in content {
        if let CanonicalPart::Text(chunk) = part {
            text.push_str(chunk);
        }
    }
    text
}

/// Extract concatenated text only when the FC trigger signal appears.
///
/// This avoids hot-path allocation when responses are plain text without
/// function-call trigger markers.
#[must_use]
pub fn extract_response_text_if_trigger(content: &[CanonicalPart]) -> Option<Cow<'_, str>> {
    let trigger = prompt::get_trigger_signal();
    let trigger_overlap = trigger.len().saturating_sub(1);
    let mut text_parts = content.iter().filter_map(|part| match part {
        CanonicalPart::Text(text) => Some(text.as_str()),
        _ => None,
    });

    let first = text_parts.next()?;
    let second = text_parts.next();
    match second {
        None => {
            if response_text_contains_trigger(first.as_bytes()) {
                Some(Cow::Borrowed(first))
            } else {
                None
            }
        }
        Some(second) => {
            let mut merged = String::with_capacity(first.len() + second.len());
            merged.push_str(first);
            let mut found = response_text_contains_trigger(first.as_bytes());

            merged.push_str(second);
            if !found {
                let start = first.len().saturating_sub(trigger_overlap);
                found = TRIGGER_SIGNAL_FINDER
                    .find(&merged.as_bytes()[start..])
                    .is_some();
            }

            for part in text_parts {
                if found {
                    merged.push_str(part);
                } else {
                    let start = merged.len().saturating_sub(trigger_overlap);
                    merged.push_str(part);
                    found = TRIGGER_SIGNAL_FINDER
                        .find(&merged.as_bytes()[start..])
                        .is_some();
                }
            }
            if found {
                Some(Cow::Owned(merged))
            } else {
                None
            }
        }
    }
}

/// Check whether raw JSON response bytes contain the FC trigger signal.
///
/// This is used as a cheap guard before expensive decode/encode work in
/// FC-inject non-streaming fast paths.
#[must_use]
pub fn response_text_contains_trigger(bytes: &[u8]) -> bool {
    let trigger = *TRIGGER_SIGNAL_BYTES;
    if trigger.is_empty() || bytes.len() < trigger.len() {
        return false;
    }
    let Some(first_byte) = *TRIGGER_SIGNAL_FIRST_BYTE else {
        return false;
    };
    let Some(first_possible_start) = memchr::memchr(first_byte, bytes) else {
        return false;
    };
    let candidate = &bytes[first_possible_start..];
    if candidate.len() >= trigger.len() && candidate[..trigger.len()] == *trigger {
        return true;
    }
    if candidate.len() <= 1 {
        return false;
    }
    TRIGGER_SIGNAL_FINDER.find(&candidate[1..]).is_some()
}

/// Apply FC response post-processing in one-shot mode.
///
/// This is used when retry is disabled: parse at most once and pass through on
/// parse/validation failures.
///
/// # Errors
///
/// Returns [`CanonicalError`] when FC response processing fails unexpectedly.
pub fn apply_fc_postprocess_once(
    upstream_response: &mut CanonicalResponse,
    tools: &[CanonicalToolSpec],
) -> Result<(), CanonicalError> {
    let Some(response_text) = extract_response_text_if_trigger(&upstream_response.content) else {
        return Ok(());
    };
    let trigger = prompt::get_trigger_signal();

    // One-shot path: on parse/validation failures we pass through upstream
    // content without building retry-specific error payloads.
    let Ok(parsed_calls) = parser::parse_function_calls(response_text.as_ref(), trigger) else {
        return Ok(());
    };
    if validator::validate_parser_tool_calls(&parsed_calls, tools).is_err() {
        return Ok(());
    }

    let mut tool_parts: Vec<CanonicalPart> = Vec::with_capacity(parsed_calls.len());
    for call in parsed_calls {
        tool_parts.push(parsed_to_canonical_tool_call(call)?);
    }
    if tool_parts.is_empty() {
        return Ok(());
    }

    let text_before = response_text.find(trigger).and_then(|pos| {
        let trimmed = response_text[..pos].trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    });
    let mut new_content: Vec<CanonicalPart> =
        Vec::with_capacity(tool_parts.len() + usize::from(text_before.is_some()));
    if let Some(text_before) = text_before {
        new_content.push(CanonicalPart::Text(text_before));
    }
    new_content.append(&mut tool_parts);
    upstream_response.content = new_content;
    upstream_response.stop_reason = CanonicalStopReason::ToolCalls;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::CanonicalToolFunction;
    use serde_json::json;

    fn make_tool(name: &str, desc: &str, params: serde_json::Value) -> CanonicalToolSpec {
        CanonicalToolSpec {
            function: CanonicalToolFunction {
                name: name.to_string(),
                description: if desc.is_empty() {
                    None
                } else {
                    Some(desc.to_string())
                },
                parameters: params,
            },
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
    fn test_process_fc_response_no_trigger() {
        let tools = vec![make_tool("f", "desc", json!({}))];
        let result = process_fc_response("Just a normal response.", &tools).unwrap();
        assert!(matches!(result, FcResult::NoToolCalls));
    }

    #[test]
    fn test_process_fc_response_valid_tool_call() {
        let tools = vec![make_tool(
            "get_weather",
            "Get weather",
            json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
        )];

        let trigger = prompt::get_trigger_signal();
        let response_text = format!(
            "Let me check.\n{trigger}\n\
             <function_calls>\
             <invoke name=\"get_weather\">\
             <parameter name=\"city\">London</parameter>\
             </invoke>\
             </function_calls>"
        );

        let result = process_fc_response(&response_text, &tools).unwrap();
        match result {
            FcResult::ToolCalls {
                tool_parts: parts,
                text_before,
            } => {
                assert_eq!(parts.len(), 1);
                if let CanonicalPart::ToolCall { name, id, .. } = &parts[0] {
                    assert_eq!(name, "get_weather");
                    assert!(id.starts_with("call_"));
                } else {
                    panic!("expected ToolCall part");
                }
                assert_eq!(text_before.as_deref(), Some("Let me check."));
            }
            other => panic!("expected ToolCalls, got {other:?}"),
        }
    }

    #[test]
    fn test_process_fc_response_preserves_tool_call_id() {
        let tools = vec![make_tool(
            "get_weather",
            "Get weather",
            json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
        )];

        let trigger = prompt::get_trigger_signal();
        let response_text = format!(
            "{trigger}\n\
             <function_calls>\
             <function_call>\
             <id>call_preserved_1</id>\
             <tool>get_weather</tool>\
             <args_json>{{\"city\":\"London\"}}</args_json>\
             </function_call>\
             </function_calls>"
        );

        let result = process_fc_response(&response_text, &tools).unwrap();
        match result {
            FcResult::ToolCalls { tool_parts, .. } => {
                assert_eq!(tool_parts.len(), 1);
                if let CanonicalPart::ToolCall { id, name, .. } = &tool_parts[0] {
                    assert_eq!(id, "call_preserved_1");
                    assert_eq!(name, "get_weather");
                } else {
                    panic!("expected ToolCall part");
                }
            }
            other => panic!("expected ToolCalls, got {other:?}"),
        }
    }

    #[test]
    fn test_process_fc_response_parse_error() {
        let tools = vec![make_tool("f", "desc", json!({}))];
        let trigger = prompt::get_trigger_signal();
        let response_text = format!("{trigger}\nsome garbage");

        let result = process_fc_response(&response_text, &tools).unwrap();
        assert!(matches!(
            result,
            FcResult::ParseError {
                trigger_found: true,
                ..
            }
        ));
    }

    #[test]
    fn test_process_fc_response_validation_error() {
        let tools = vec![make_tool(
            "get_weather",
            "Get weather",
            json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
        )];

        let trigger = prompt::get_trigger_signal();
        let response_text = format!(
            "{trigger}\n\
             <function_calls>\
             <invoke name=\"get_weather\">\
             </invoke>\
             </function_calls>"
        );

        let result = process_fc_response(&response_text, &tools).unwrap();
        match result {
            FcResult::ParseError {
                trigger_found,
                error,
                ..
            } => {
                assert!(trigger_found);
                assert!(error.contains("missing required property"));
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    #[test]
    fn test_extract_response_text() {
        let parts = vec![
            CanonicalPart::Text("Hello ".to_string()),
            CanonicalPart::Text("world".to_string()),
            make_tool_call_part("id", "name", "{}"),
        ];
        assert_eq!(extract_response_text(&parts), "Hello world");
    }

    #[test]
    fn test_extract_response_text_empty() {
        let parts: Vec<CanonicalPart> = vec![];
        assert_eq!(extract_response_text(&parts), "");
    }

    #[test]
    fn test_extract_response_text_if_trigger_none() {
        let parts = vec![CanonicalPart::Text("plain text".to_string())];
        assert!(extract_response_text_if_trigger(&parts).is_none());
    }

    #[test]
    fn test_extract_response_text_if_trigger_single_part() {
        let trigger = prompt::get_trigger_signal();
        let parts = vec![CanonicalPart::Text(format!("before {trigger} after"))];
        assert!(extract_response_text_if_trigger(&parts).is_some());
    }

    #[test]
    fn test_extract_response_text_if_trigger_cross_part() {
        let trigger = prompt::get_trigger_signal();
        let split = trigger.len() / 2;
        let parts = vec![
            CanonicalPart::Text(trigger[..split].to_string()),
            CanonicalPart::Text(trigger[split..].to_string()),
        ];
        assert!(extract_response_text_if_trigger(&parts).is_some());
    }
}
