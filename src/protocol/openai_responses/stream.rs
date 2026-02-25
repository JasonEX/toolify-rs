use std::collections::HashMap;

use rustc_hash::FxHashMap;

use crate::protocol::canonical::{
    CanonicalRole, CanonicalStopReason, CanonicalStreamEvent, CanonicalUsage,
};
use crate::util::{parse_sse_data_json_line, push_json_string_escaped, push_usize_decimal};

use super::{ResponsesOutputItem, ResponsesStreamEvent};

/// Parse a single SSE line pair into a `ResponsesStreamEvent`.
///
/// The Responses API uses the format:
/// ```text
/// event: response.output_text.delta
/// data: {"type":"response.output_text.delta",...}
/// ```
///
/// This function expects the `data:` payload (JSON string) as input.
#[must_use]
pub fn parse_responses_sse_line(line: &str) -> Option<ResponsesStreamEvent> {
    parse_sse_data_json_line(line, true, true, false)
}

/// Convert a Responses API stream event into canonical stream events.
#[must_use]
pub fn decode_responses_stream_event(event: &ResponsesStreamEvent) -> Vec<CanonicalStreamEvent> {
    let mut events = Vec::with_capacity(4);
    decode_responses_stream_event_into(event, &mut events);
    events
}

/// Convert a Responses API stream event into a caller-provided canonical events buffer.
pub fn decode_responses_stream_event_into(
    event: &ResponsesStreamEvent,
    out: &mut Vec<CanonicalStreamEvent>,
) {
    match event {
        ResponsesStreamEvent::ResponseCreated { .. } => {
            // Passthrough — emits a MessageStart
            out.push(CanonicalStreamEvent::MessageStart {
                role: CanonicalRole::Assistant,
            });
        }
        ResponsesStreamEvent::ResponseInProgress { .. }
        | ResponsesStreamEvent::ContentPartAdded { .. }
        | ResponsesStreamEvent::ContentPartDone { .. }
        | ResponsesStreamEvent::OutputTextDone { .. }
        | ResponsesStreamEvent::FunctionCallArgumentsDone { .. } => {
            // No canonical equivalent; ignore
        }
        ResponsesStreamEvent::OutputItemAdded { output_index, item } => {
            if let ResponsesOutputItem::FunctionCall { call_id, name, .. } = item {
                out.push(CanonicalStreamEvent::ToolCallStart {
                    index: *output_index,
                    id: call_id.clone(),
                    name: name.clone(),
                });
            }
        }
        ResponsesStreamEvent::OutputItemDone { output_index, item } => match item {
            ResponsesOutputItem::FunctionCall { call_id, name, .. } => {
                out.push(CanonicalStreamEvent::ToolCallEnd {
                    index: *output_index,
                    call_id: Some(call_id.clone()),
                    call_name: Some(name.clone()),
                });
            }
            ResponsesOutputItem::FunctionCallOutput {
                call_id, output, ..
            } => out.push(CanonicalStreamEvent::ToolResult {
                tool_call_id: call_id.clone(),
                content: output.clone(),
            }),
            ResponsesOutputItem::Message { .. } => {}
        },
        ResponsesStreamEvent::OutputTextDelta { delta, .. } => {
            out.push(CanonicalStreamEvent::TextDelta(delta.clone()));
        }
        ResponsesStreamEvent::FunctionCallArgumentsDelta {
            output_index,
            delta,
        } => out.push(CanonicalStreamEvent::ToolCallArgsDelta {
            index: *output_index,
            delta: delta.clone(),
        }),
        ResponsesStreamEvent::ResponseCompleted { response } => {
            // Extract usage if present
            if let Some(ref usage) = response.usage {
                let total = usage
                    .total_tokens
                    .unwrap_or(usage.input_tokens + usage.output_tokens);
                out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
                    input_tokens: Some(usage.input_tokens),
                    output_tokens: Some(usage.output_tokens),
                    total_tokens: Some(total),
                }));
            }

            // Determine stop reason from output items
            let has_fc = response
                .output
                .iter()
                .any(|item| matches!(item, ResponsesOutputItem::FunctionCall { .. }));
            let has_tool_result = response
                .output
                .iter()
                .any(|item| matches!(item, ResponsesOutputItem::FunctionCallOutput { .. }));
            let stop_reason = if has_fc || has_tool_result {
                CanonicalStopReason::ToolCalls
            } else {
                CanonicalStopReason::EndOfTurn
            };

            out.push(CanonicalStreamEvent::MessageEnd { stop_reason });
            out.push(CanonicalStreamEvent::Done);
        }
        ResponsesStreamEvent::Error { message } => {
            out.push(CanonicalStreamEvent::Error {
                status: 500,
                message: message.clone(),
            });
        }
    }
}

/// Convert a Responses API stream event by value into a caller-provided
/// canonical events buffer.
///
/// This avoids `String` clones when callers already own the decoded stream
/// event.
pub fn decode_responses_stream_event_owned_into(
    event: ResponsesStreamEvent,
    out: &mut Vec<CanonicalStreamEvent>,
) {
    match event {
        ResponsesStreamEvent::ResponseCreated { .. } => {
            out.push(CanonicalStreamEvent::MessageStart {
                role: CanonicalRole::Assistant,
            });
        }
        ResponsesStreamEvent::ResponseInProgress { .. }
        | ResponsesStreamEvent::ContentPartAdded { .. }
        | ResponsesStreamEvent::ContentPartDone { .. }
        | ResponsesStreamEvent::OutputTextDone { .. }
        | ResponsesStreamEvent::FunctionCallArgumentsDone { .. } => {}
        ResponsesStreamEvent::OutputItemAdded { output_index, item } => {
            if let ResponsesOutputItem::FunctionCall { call_id, name, .. } = item {
                out.push(CanonicalStreamEvent::ToolCallStart {
                    index: output_index,
                    id: call_id,
                    name,
                });
            }
        }
        ResponsesStreamEvent::OutputItemDone { output_index, item } => match item {
            ResponsesOutputItem::FunctionCall { call_id, name, .. } => {
                out.push(CanonicalStreamEvent::ToolCallEnd {
                    index: output_index,
                    call_id: Some(call_id),
                    call_name: Some(name),
                });
            }
            ResponsesOutputItem::FunctionCallOutput {
                call_id, output, ..
            } => out.push(CanonicalStreamEvent::ToolResult {
                tool_call_id: call_id,
                content: output,
            }),
            ResponsesOutputItem::Message { .. } => {}
        },
        ResponsesStreamEvent::OutputTextDelta { delta, .. } => {
            out.push(CanonicalStreamEvent::TextDelta(delta));
        }
        ResponsesStreamEvent::FunctionCallArgumentsDelta {
            output_index,
            delta,
        } => out.push(CanonicalStreamEvent::ToolCallArgsDelta {
            index: output_index,
            delta,
        }),
        ResponsesStreamEvent::ResponseCompleted { response } => {
            if let Some(usage) = response.usage {
                let total = usage
                    .total_tokens
                    .unwrap_or(usage.input_tokens + usage.output_tokens);
                out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
                    input_tokens: Some(usage.input_tokens),
                    output_tokens: Some(usage.output_tokens),
                    total_tokens: Some(total),
                }));
            }

            let mut has_fc = false;
            let mut has_tool_result = false;
            for item in &response.output {
                match item {
                    ResponsesOutputItem::FunctionCall { .. } => has_fc = true,
                    ResponsesOutputItem::FunctionCallOutput { .. } => has_tool_result = true,
                    ResponsesOutputItem::Message { .. } => {}
                }
                if has_fc && has_tool_result {
                    break;
                }
            }

            let stop_reason = if has_fc || has_tool_result {
                CanonicalStopReason::ToolCalls
            } else {
                CanonicalStopReason::EndOfTurn
            };
            out.push(CanonicalStreamEvent::MessageEnd { stop_reason });
            out.push(CanonicalStreamEvent::Done);
        }
        ResponsesStreamEvent::Error { message } => {
            out.push(CanonicalStreamEvent::Error {
                status: 500,
                message,
            });
        }
    }
}

/// Encode a canonical stream event into Responses API SSE event(s).
///
/// Returns a list of `(event_type, json_data)` pairs. Each pair becomes:
/// ```text
/// event: {event_type}
/// data: {json_data}
/// ```
#[must_use]
pub fn encode_canonical_event_to_responses_sse(
    event: &CanonicalStreamEvent,
    model: &str,
    response_id: &str,
) -> Vec<(&'static str, String)> {
    let mut tool_result_seq = FxHashMap::default();
    let mut out = Vec::with_capacity(1);
    encode_canonical_event_to_responses_sse_with_state_into(
        event,
        model,
        response_id,
        &mut tool_result_seq,
        &mut out,
    );
    out
}

/// Encode a canonical stream event into Responses API SSE event(s) with state.
///
/// `tool_result_seq` tracks per-call sequence numbers so repeated tool results
/// keep unique `function_call_output.id` values.
pub fn encode_canonical_event_to_responses_sse_with_state<S>(
    event: &CanonicalStreamEvent,
    model: &str,
    response_id: &str,
    tool_result_seq: &mut HashMap<String, usize, S>,
) -> Vec<(&'static str, String)>
where
    S: std::hash::BuildHasher,
{
    let mut out = Vec::with_capacity(1);
    encode_canonical_event_to_responses_sse_with_state_into(
        event,
        model,
        response_id,
        tool_result_seq,
        &mut out,
    );
    out
}

/// Encode a canonical stream event into a caller-provided Responses SSE pair buffer.
pub fn encode_canonical_event_to_responses_sse_with_state_into<S>(
    event: &CanonicalStreamEvent,
    model: &str,
    response_id: &str,
    tool_result_seq: &mut HashMap<String, usize, S>,
    out: &mut Vec<(&'static str, String)>,
) where
    S: std::hash::BuildHasher,
{
    out.clear();
    match event {
        CanonicalStreamEvent::MessageStart { .. } => {
            let data =
                build_response_envelope_data(model, response_id, "response.created", "in_progress");
            out.push(("response.created", data));
        }
        CanonicalStreamEvent::TextDelta(delta) => {
            let mut data = String::with_capacity(72 + delta.len());
            data.push_str("{\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":");
            push_json_string_escaped(&mut data, delta);
            data.push('}');
            out.push(("response.output_text.delta", data));
        }
        CanonicalStreamEvent::ToolCallStart { index, id, name } => {
            let mut data = String::with_capacity(112 + id.len() + name.len());
            data.push_str("{\"type\":\"response.output_item.added\",\"output_index\":");
            push_usize_decimal(&mut data, *index);
            data.push_str(",\"item\":{\"type\":\"function_call\",\"id\":\"fc_");
            push_usize_decimal(&mut data, *index);
            data.push_str("\",\"call_id\":");
            push_json_string_escaped(&mut data, id);
            data.push_str(",\"name\":");
            push_json_string_escaped(&mut data, name);
            data.push_str(",\"arguments\":\"\"}}");
            out.push(("response.output_item.added", data));
        }
        CanonicalStreamEvent::ToolCallArgsDelta { index, delta } => {
            let mut data = String::with_capacity(80 + delta.len());
            data.push_str("{\"type\":\"response.function_call_arguments.delta\",\"output_index\":");
            push_usize_decimal(&mut data, *index);
            data.push_str(",\"delta\":");
            push_json_string_escaped(&mut data, delta);
            data.push('}');
            out.push(("response.function_call_arguments.delta", data));
        }
        CanonicalStreamEvent::ToolCallEnd {
            index,
            call_id,
            call_name,
        } => {
            let call_id = call_id.as_deref().unwrap_or("");
            let name = call_name.as_deref().unwrap_or("");
            let mut data = String::with_capacity(112 + call_id.len() + name.len());
            data.push_str("{\"type\":\"response.output_item.done\",\"output_index\":");
            push_usize_decimal(&mut data, *index);
            data.push_str(",\"item\":{\"type\":\"function_call\",\"id\":\"fc_");
            push_usize_decimal(&mut data, *index);
            data.push_str("\",\"call_id\":");
            push_json_string_escaped(&mut data, call_id);
            data.push_str(",\"name\":");
            push_json_string_escaped(&mut data, name);
            data.push_str(",\"arguments\":\"\"}}");
            out.push(("response.output_item.done", data));
        }
        CanonicalStreamEvent::ToolResult {
            tool_call_id,
            content,
        } => {
            let seq = next_tool_result_sequence(tool_result_seq, tool_call_id);
            let mut data = String::with_capacity(104 + tool_call_id.len() * 2 + content.len());
            data.push_str("{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call_output\",\"id\":\"fco_");
            data.push_str(tool_call_id);
            data.push('_');
            push_usize_decimal(&mut data, seq);
            data.push_str("\",\"call_id\":");
            push_json_string_escaped(&mut data, tool_call_id);
            data.push_str(",\"output\":");
            push_json_string_escaped(&mut data, content);
            data.push_str("}}");
            out.push(("response.output_item.added", data));
        }
        CanonicalStreamEvent::Usage(usage) => {
            // Usage is typically bundled with response.completed; emit nothing standalone.
            let _ = usage;
        }
        CanonicalStreamEvent::MessageEnd { .. } | CanonicalStreamEvent::ReasoningDelta(_) => {
            // No-op for Responses API; response.completed is emitted on Done.
        }
        CanonicalStreamEvent::Done => {
            let data =
                build_response_envelope_data(model, response_id, "response.completed", "completed");
            out.push(("response.completed", data));
        }
        CanonicalStreamEvent::Error { message, .. } => {
            let mut data = String::with_capacity(26 + message.len());
            data.push_str("{\"type\":\"error\",\"message\":");
            push_json_string_escaped(&mut data, message);
            data.push('}');
            out.push(("error", data));
        }
    }
}

/// Encode a canonical stream event directly into a full Responses SSE frame.
///
/// Produces one frame:
/// ```text
/// event: {event_type}
/// data: {json_data}
///
/// ```
///
/// Returns `true` when a frame is produced and written into `out`.
pub fn encode_canonical_event_to_responses_sse_frame_with_state<S>(
    event: &CanonicalStreamEvent,
    model: &str,
    response_id: &str,
    tool_result_seq: &mut HashMap<String, usize, S>,
    out: &mut String,
) -> bool
where
    S: std::hash::BuildHasher,
{
    out.clear();
    match event {
        CanonicalStreamEvent::MessageStart { .. } => {
            out.push_str("event: response.created\ndata: ");
            push_response_envelope_data(out, model, response_id, "response.created", "in_progress");
            out.push_str("\n\n");
            true
        }
        CanonicalStreamEvent::TextDelta(delta) => {
            out.push_str("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":");
            push_json_string_escaped(out, delta);
            out.push_str("}\n\n");
            true
        }
        CanonicalStreamEvent::ToolCallStart { index, id, name } => {
            out.push_str(
                "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":",
            );
            push_usize_decimal(out, *index);
            out.push_str(",\"item\":{\"type\":\"function_call\",\"id\":\"fc_");
            push_usize_decimal(out, *index);
            out.push_str("\",\"call_id\":");
            push_json_string_escaped(out, id);
            out.push_str(",\"name\":");
            push_json_string_escaped(out, name);
            out.push_str(",\"arguments\":\"\"}}\n\n");
            true
        }
        CanonicalStreamEvent::ToolCallArgsDelta { index, delta } => {
            out.push_str(
                "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":",
            );
            push_usize_decimal(out, *index);
            out.push_str(",\"delta\":");
            push_json_string_escaped(out, delta);
            out.push_str("}\n\n");
            true
        }
        CanonicalStreamEvent::ToolCallEnd {
            index,
            call_id,
            call_name,
        } => {
            let call_id = call_id.as_deref().unwrap_or("");
            let name = call_name.as_deref().unwrap_or("");
            out.push_str(
                "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":",
            );
            push_usize_decimal(out, *index);
            out.push_str(",\"item\":{\"type\":\"function_call\",\"id\":\"fc_");
            push_usize_decimal(out, *index);
            out.push_str("\",\"call_id\":");
            push_json_string_escaped(out, call_id);
            out.push_str(",\"name\":");
            push_json_string_escaped(out, name);
            out.push_str(",\"arguments\":\"\"}}\n\n");
            true
        }
        CanonicalStreamEvent::ToolResult {
            tool_call_id,
            content,
        } => {
            let seq = next_tool_result_sequence(tool_result_seq, tool_call_id);
            out.push_str(
                "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call_output\",\"id\":\"fco_",
            );
            out.push_str(tool_call_id);
            out.push('_');
            push_usize_decimal(out, seq);
            out.push_str("\",\"call_id\":");
            push_json_string_escaped(out, tool_call_id);
            out.push_str(",\"output\":");
            push_json_string_escaped(out, content);
            out.push_str("}}\n\n");
            true
        }
        CanonicalStreamEvent::Usage(_)
        | CanonicalStreamEvent::MessageEnd { .. }
        | CanonicalStreamEvent::ReasoningDelta(_) => false,
        CanonicalStreamEvent::Done => {
            out.push_str("event: response.completed\ndata: ");
            push_response_envelope_data(out, model, response_id, "response.completed", "completed");
            out.push_str("\n\n");
            true
        }
        CanonicalStreamEvent::Error { message, .. } => {
            out.push_str("event: error\ndata: {\"type\":\"error\",\"message\":");
            push_json_string_escaped(out, message);
            out.push_str("}\n\n");
            true
        }
    }
}

fn build_response_envelope_data(
    model: &str,
    response_id: &str,
    event_type: &str,
    status: &str,
) -> String {
    let mut data = String::with_capacity(
        88 + model.len() + response_id.len() + event_type.len() + status.len(),
    );
    push_response_envelope_data(&mut data, model, response_id, event_type, status);
    data
}

fn push_response_envelope_data(
    out: &mut String,
    model: &str,
    response_id: &str,
    event_type: &str,
    status: &str,
) {
    out.push_str("{\"type\":");
    push_json_string_escaped(out, event_type);
    out.push_str(",\"response\":{\"id\":");
    push_json_string_escaped(out, response_id);
    out.push_str(",\"object\":\"response\",\"model\":");
    push_json_string_escaped(out, model);
    out.push_str(",\"output\":[],\"status\":");
    push_json_string_escaped(out, status);
    out.push_str("}}");
}

#[inline]
fn next_tool_result_sequence<S>(
    tool_result_seq: &mut HashMap<String, usize, S>,
    tool_call_id: &str,
) -> usize
where
    S: std::hash::BuildHasher,
{
    if let Some(seq) = tool_result_seq.get_mut(tool_call_id) {
        *seq += 1;
        *seq
    } else {
        tool_result_seq.insert(tool_call_id.to_owned(), 0);
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::openai_responses::{ResponsesOutput, ResponsesUsage};

    #[test]
    fn test_parse_sse_text_delta() {
        let line = r#"data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hello"}"#;
        let event = parse_responses_sse_line(line).unwrap();
        match event {
            ResponsesStreamEvent::OutputTextDelta { delta, .. } => {
                assert_eq!(delta, "Hello");
            }
            _ => panic!("Expected OutputTextDelta"),
        }
    }

    #[test]
    fn test_parse_sse_done() {
        assert!(parse_responses_sse_line("data: [DONE]").is_none());
    }

    #[test]
    fn test_decode_text_delta() {
        let event = ResponsesStreamEvent::OutputTextDelta {
            output_index: 0,
            content_index: 0,
            delta: "Hi".into(),
        };
        let canonical = decode_responses_stream_event(&event);
        assert_eq!(canonical.len(), 1);
        assert!(matches!(&canonical[0], CanonicalStreamEvent::TextDelta(t) if t == "Hi"));
    }

    #[test]
    fn test_decode_function_call_start() {
        let event = ResponsesStreamEvent::OutputItemAdded {
            output_index: 1,
            item: ResponsesOutputItem::FunctionCall {
                id: "fc_0".into(),
                call_id: "call_abc".into(),
                name: "get_weather".into(),
                arguments: String::new(),
            },
        };
        let canonical = decode_responses_stream_event(&event);
        assert_eq!(canonical.len(), 1);
        match &canonical[0] {
            CanonicalStreamEvent::ToolCallStart { index, id, name } => {
                assert_eq!(*index, 1);
                assert_eq!(id, "call_abc");
                assert_eq!(name, "get_weather");
            }
            _ => panic!("Expected ToolCallStart"),
        }
    }

    #[test]
    fn test_decode_response_completed() {
        let event = ResponsesStreamEvent::ResponseCompleted {
            response: ResponsesOutput {
                id: "resp_1".into(),
                object: "response".into(),
                model: "gpt-4o".into(),
                output: vec![],
                usage: Some(ResponsesUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    total_tokens: Some(15),
                }),
                status: Some("completed".into()),
                extra: serde_json::Map::new(),
            },
        };
        let canonical = decode_responses_stream_event(&event);
        // Usage + MessageEnd + Done
        assert_eq!(canonical.len(), 3);
    }

    #[test]
    fn test_encode_text_delta() {
        let event = CanonicalStreamEvent::TextDelta("world".into());
        let pairs = encode_canonical_event_to_responses_sse(&event, "gpt-4o", "resp_test");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "response.output_text.delta");
        assert!(pairs[0].1.contains("world"));
    }

    #[test]
    fn test_encode_tool_call_start() {
        let event = CanonicalStreamEvent::ToolCallStart {
            index: 0,
            id: "call_123".into(),
            name: "search".into(),
        };
        let pairs = encode_canonical_event_to_responses_sse(&event, "gpt-4o", "resp_test");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "response.output_item.added");
    }

    #[test]
    fn test_decode_function_call_output() {
        let event = ResponsesStreamEvent::OutputItemDone {
            output_index: 2,
            item: ResponsesOutputItem::FunctionCallOutput {
                id: "fco_0".into(),
                call_id: "call_abc".into(),
                output: r#"{"temp":72}"#.into(),
            },
        };

        let canonical = decode_responses_stream_event(&event);
        assert_eq!(canonical.len(), 1);
        match &canonical[0] {
            CanonicalStreamEvent::ToolResult {
                tool_call_id,
                content,
            } => {
                assert_eq!(tool_call_id, "call_abc");
                assert_eq!(content, r#"{"temp":72}"#);
            }
            _ => panic!("Expected ToolResult"),
        }
    }

    #[test]
    fn test_decode_response_completed_with_tool_result() {
        let event = ResponsesStreamEvent::ResponseCompleted {
            response: ResponsesOutput {
                id: "resp_1".into(),
                object: "response".into(),
                model: "gpt-4o".into(),
                output: vec![ResponsesOutputItem::FunctionCallOutput {
                    id: "fco_1".into(),
                    call_id: "call_xyz".into(),
                    output: "done".into(),
                }],
                usage: None,
                status: Some("completed".into()),
                extra: serde_json::Map::new(),
            },
        };

        let canonical = decode_responses_stream_event(&event);
        assert_eq!(canonical.len(), 2);

        match &canonical[0] {
            CanonicalStreamEvent::MessageEnd { stop_reason } => {
                assert_eq!(*stop_reason, CanonicalStopReason::ToolCalls);
            }
            _ => panic!("Expected MessageEnd"),
        }

        assert!(matches!(&canonical[1], CanonicalStreamEvent::Done));
    }

    #[test]
    fn test_encode_tool_result() {
        let event = CanonicalStreamEvent::ToolResult {
            tool_call_id: "call_abc".into(),
            content: r#"{"temp":72}"#.into(),
        };
        let pairs = encode_canonical_event_to_responses_sse(&event, "gpt-4o", "resp_test");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "response.output_item.added");

        let payload: serde_json::Value = serde_json::from_str(&pairs[0].1).unwrap();
        assert_eq!(payload["type"], "response.output_item.added");
        assert_eq!(payload["item"]["type"], "function_call_output");
        assert_eq!(payload["item"]["call_id"], "call_abc");
    }

    #[test]
    fn test_encode_text_delta_frame_matches_pair_render() {
        let event = CanonicalStreamEvent::TextDelta("hello".into());
        let pairs = encode_canonical_event_to_responses_sse(&event, "gpt-4o", "resp_test");
        assert_eq!(pairs.len(), 1);
        let expected = format!("event: {}\ndata: {}\n\n", pairs[0].0, pairs[0].1);

        let mut seq: HashMap<String, usize> = HashMap::new();
        let mut frame = String::new();
        let produced = encode_canonical_event_to_responses_sse_frame_with_state(
            &event,
            "gpt-4o",
            "resp_test",
            &mut seq,
            &mut frame,
        );
        assert!(produced);
        assert_eq!(frame, expected);
    }

    #[test]
    fn test_encode_done_frame_matches_pair_render() {
        let event = CanonicalStreamEvent::Done;
        let pairs = encode_canonical_event_to_responses_sse(&event, "gpt-4o", "resp_test");
        assert_eq!(pairs.len(), 1);
        let expected = format!("event: {}\ndata: {}\n\n", pairs[0].0, pairs[0].1);

        let mut seq: HashMap<String, usize> = HashMap::new();
        let mut frame = String::new();
        let produced = encode_canonical_event_to_responses_sse_frame_with_state(
            &event,
            "gpt-4o",
            "resp_test",
            &mut seq,
            &mut frame,
        );
        assert!(produced);
        assert_eq!(frame, expected);
    }
}
