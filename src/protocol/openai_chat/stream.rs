use std::time::{SystemTime, UNIX_EPOCH};

use crate::protocol::canonical::{CanonicalRole, CanonicalStreamEvent, CanonicalUsage};
use crate::protocol::mapping::{canonical_stop_to_openai, openai_stop_to_canonical};
use crate::util::{parse_sse_data_json_line, push_json_string_escaped, push_u64_decimal};

use super::OpenAiStreamChunk;

const DONE_FRAME: &str = "data: [DONE]\n\n";

/// Parse a single SSE line into an `OpenAiStreamChunk`.
/// Returns `None` for comments, empty lines, `data: [DONE]`, and non-data lines.
#[must_use]
pub fn parse_openai_sse_line(line: &str) -> Option<OpenAiStreamChunk> {
    parse_sse_data_json_line(line, false, false, false)
}

/// Decode an `OpenAI` stream chunk into canonical stream events.
#[must_use]
pub fn decode_openai_stream_chunk(chunk: OpenAiStreamChunk) -> Vec<CanonicalStreamEvent> {
    let mut events = Vec::with_capacity(chunk.choices.len().saturating_mul(3) + 1);
    decode_openai_stream_chunk_into(chunk, &mut events);
    events
}

/// Decode an `OpenAI` stream chunk into a caller-provided canonical events buffer.
pub fn decode_openai_stream_chunk_into(
    chunk: OpenAiStreamChunk,
    out: &mut Vec<CanonicalStreamEvent>,
) {
    for choice in chunk.choices {
        if let Some(role) = choice.delta.role {
            let canonical_role = match role.as_str() {
                "user" => CanonicalRole::User,
                "system" => CanonicalRole::System,
                "tool" => CanonicalRole::Tool,
                _ => CanonicalRole::Assistant,
            };
            out.push(CanonicalStreamEvent::MessageStart {
                role: canonical_role,
            });
        }

        if let Some(content) = choice.delta.content {
            if !content.is_empty() {
                out.push(CanonicalStreamEvent::TextDelta(content));
            }
        }

        if let Some(tool_calls) = choice.delta.tool_calls {
            for tc in tool_calls {
                let index = tc.index as usize;
                if let Some(mut func) = tc.function {
                    if let Some(id) = tc.id {
                        let name = func.name.take().unwrap_or_default();
                        out.push(CanonicalStreamEvent::ToolCallStart { index, id, name });
                    }
                    if let Some(args) = func.arguments {
                        if !args.is_empty() {
                            out.push(CanonicalStreamEvent::ToolCallArgsDelta {
                                index,
                                delta: args,
                            });
                        }
                    }
                } else if let Some(id) = tc.id {
                    out.push(CanonicalStreamEvent::ToolCallStart {
                        index,
                        id,
                        name: String::new(),
                    });
                }
            }
        }

        if let Some(finish_reason) = choice.finish_reason {
            out.push(CanonicalStreamEvent::MessageEnd {
                stop_reason: openai_stop_to_canonical(&finish_reason),
            });
        }
    }

    if let Some(usage) = chunk.usage {
        out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
            input_tokens: Some(usage.prompt_tokens),
            output_tokens: Some(usage.completion_tokens),
            total_tokens: Some(usage.total_tokens),
        }));
    }
}

/// Encode a canonical stream event into an `OpenAI` SSE formatted line.
/// Returns `None` for events with no `OpenAI` SSE representation.
#[must_use]
pub fn encode_canonical_event_to_openai_sse(
    event: &CanonicalStreamEvent,
    model: &str,
    id: &str,
) -> Option<String> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    encode_canonical_event_to_openai_sse_with_created(event, model, id, created)
}

/// Same as [`encode_canonical_event_to_openai_sse`] but reuses a caller-provided
/// unix timestamp for lower per-event overhead on hot streaming paths.
#[must_use]
pub fn encode_canonical_event_to_openai_sse_with_created(
    event: &CanonicalStreamEvent,
    model: &str,
    id: &str,
    created: u64,
) -> Option<String> {
    match event {
        CanonicalStreamEvent::MessageStart { role } => {
            let role_str = match role {
                CanonicalRole::Assistant => "assistant",
                CanonicalRole::User => "user",
                CanonicalRole::System => "system",
                CanonicalRole::Tool => "tool",
            };
            let mut out = String::with_capacity(128 + id.len() + model.len() + role_str.len());
            push_openai_chunk_prefix(&mut out, id, model, created);
            out.push_str(",\"choices\":[{\"index\":0,\"delta\":{\"role\":");
            push_json_string_escaped(&mut out, role_str);
            out.push_str("},\"finish_reason\":null}]}\n\n");
            Some(out)
        }
        CanonicalStreamEvent::TextDelta(text) => {
            let mut out = String::with_capacity(128 + id.len() + model.len() + text.len());
            push_openai_chunk_prefix(&mut out, id, model, created);
            out.push_str(",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
            push_json_string_escaped(&mut out, text);
            out.push_str("},\"finish_reason\":null}]}\n\n");
            Some(out)
        }
        CanonicalStreamEvent::ToolCallStart {
            index,
            id: tc_id,
            name,
        } => {
            let tool_index = u32::try_from(*index).ok()?;
            let mut out =
                String::with_capacity(196 + id.len() + model.len() + tc_id.len() + name.len());
            push_openai_chunk_prefix(&mut out, id, model, created);
            out.push_str(",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
            push_u64_decimal(&mut out, u64::from(tool_index));
            out.push_str(",\"id\":");
            push_json_string_escaped(&mut out, tc_id);
            out.push_str(",\"type\":\"function\",\"function\":{\"name\":");
            push_json_string_escaped(&mut out, name);
            out.push_str("}}]},\"finish_reason\":null}]}\n\n");
            Some(out)
        }
        CanonicalStreamEvent::ToolCallArgsDelta { index, delta } => {
            let tool_index = u32::try_from(*index).ok()?;
            let mut out = String::with_capacity(176 + id.len() + model.len() + delta.len());
            push_openai_chunk_prefix(&mut out, id, model, created);
            out.push_str(",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
            push_u64_decimal(&mut out, u64::from(tool_index));
            out.push_str(",\"function\":{\"arguments\":");
            push_json_string_escaped(&mut out, delta);
            out.push_str("}}]},\"finish_reason\":null}]}\n\n");
            Some(out)
        }
        CanonicalStreamEvent::MessageEnd { stop_reason } => {
            let mut out = String::with_capacity(128 + id.len() + model.len() + 16);
            push_openai_chunk_prefix(&mut out, id, model, created);
            out.push_str(",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
            push_json_string_escaped(&mut out, canonical_stop_to_openai(*stop_reason));
            out.push_str("}]}\n\n");
            Some(out)
        }
        CanonicalStreamEvent::Usage(usage) => {
            let mut out = String::with_capacity(160 + id.len() + model.len());
            push_openai_chunk_prefix(&mut out, id, model, created);
            out.push_str(",\"choices\":[],\"usage\":{\"prompt_tokens\":");
            push_u64_decimal(&mut out, usage.input_tokens.unwrap_or(0));
            out.push_str(",\"completion_tokens\":");
            push_u64_decimal(&mut out, usage.output_tokens.unwrap_or(0));
            out.push_str(",\"total_tokens\":");
            push_u64_decimal(&mut out, usage.total_tokens.unwrap_or(0));
            out.push_str("}}\n\n");
            Some(out)
        }
        CanonicalStreamEvent::Done => Some(DONE_FRAME.to_owned()),
        CanonicalStreamEvent::ToolCallEnd { .. }
        | CanonicalStreamEvent::ToolResult { .. }
        | CanonicalStreamEvent::ReasoningDelta(_) => None,
        CanonicalStreamEvent::Error { message, .. } => {
            let mut out = String::with_capacity(40 + message.len());
            out.push_str("data: {\"error\":{\"message\":");
            push_json_string_escaped(&mut out, message);
            out.push_str("}}\n\n");
            Some(out)
        }
    }
}

fn push_openai_chunk_prefix(out: &mut String, id: &str, model: &str, created: u64) {
    out.push_str("data: {\"id\":");
    push_json_string_escaped(out, id);
    out.push_str(",\"object\":\"chat.completion.chunk\",\"created\":");
    push_u64_decimal(out, created);
    out.push_str(",\"model\":");
    push_json_string_escaped(out, model);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::CanonicalStopReason;

    #[test]
    fn test_parse_sse_data_line() {
        let line = r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#;
        let chunk = parse_openai_sse_line(line).unwrap();
        assert_eq!(chunk.id, "chatcmpl-1");
        assert_eq!(chunk.choices[0].delta.content, Some("Hi".to_string()));
    }

    #[test]
    fn test_parse_sse_done() {
        assert!(parse_openai_sse_line("data: [DONE]").is_none());
    }

    #[test]
    fn test_parse_sse_empty() {
        assert!(parse_openai_sse_line("").is_none());
        assert!(parse_openai_sse_line(": comment").is_none());
    }

    #[test]
    fn test_decode_text_delta() {
        let chunk: OpenAiStreamChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": null
            }]
        }))
        .unwrap();
        let events = decode_openai_stream_chunk(chunk);
        assert!(matches!(&events[0], CanonicalStreamEvent::TextDelta(t) if t == "Hello"));
    }

    #[test]
    fn test_decode_role_delta() {
        let chunk: OpenAiStreamChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": null
            }]
        }))
        .unwrap();
        let events = decode_openai_stream_chunk(chunk);
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::MessageStart {
                role: CanonicalRole::Assistant
            }
        ));
    }

    #[test]
    fn test_decode_finish_reason() {
        let chunk: OpenAiStreamChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }))
        .unwrap();
        let events = decode_openai_stream_chunk(chunk);
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::MessageEnd {
                stop_reason: CanonicalStopReason::EndOfTurn
            }
        ));
    }

    #[test]
    fn test_decode_tool_call_stream() {
        let chunk: OpenAiStreamChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""}
                    }]
                },
                "finish_reason": null
            }]
        }))
        .unwrap();
        let events = decode_openai_stream_chunk(chunk);
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::ToolCallStart { index: 0, id, name }
                if id == "call_abc" && name == "get_weather"
        ));
    }

    #[test]
    fn test_decode_tool_call_args_delta() {
        let chunk: OpenAiStreamChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "{\"loc"}
                    }]
                },
                "finish_reason": null
            }]
        }))
        .unwrap();
        let events = decode_openai_stream_chunk(chunk);
        assert!(matches!(
            &events[0],
            CanonicalStreamEvent::ToolCallArgsDelta { index: 0, delta }
                if delta == "{\"loc"
        ));
    }

    #[test]
    fn test_encode_done() {
        let result =
            encode_canonical_event_to_openai_sse(&CanonicalStreamEvent::Done, "gpt-4", "id-1");
        assert_eq!(result, Some("data: [DONE]\n\n".to_string()));
    }

    #[test]
    fn test_encode_text_delta() {
        let result = encode_canonical_event_to_openai_sse(
            &CanonicalStreamEvent::TextDelta("world".to_string()),
            "gpt-4",
            "id-1",
        );
        let line = result.unwrap();
        assert!(line.starts_with("data: "));
        let json_str = line.trim_start_matches("data: ").trim();
        let chunk: OpenAiStreamChunk = serde_json::from_str(json_str).unwrap();
        assert_eq!(chunk.choices[0].delta.content, Some("world".to_string()));
    }

    #[test]
    fn test_roundtrip_stream_text() {
        let event = CanonicalStreamEvent::TextDelta("test".to_string());
        let sse = encode_canonical_event_to_openai_sse(&event, "gpt-4", "id-1").unwrap();
        let chunk = parse_openai_sse_line(sse.trim()).unwrap();
        let decoded = decode_openai_stream_chunk(chunk);
        assert!(matches!(&decoded[0], CanonicalStreamEvent::TextDelta(t) if t == "test"));
    }
}
