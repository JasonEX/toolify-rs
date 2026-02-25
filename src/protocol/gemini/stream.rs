use std::collections::HashMap;

use rustc_hash::FxHashMap;

use crate::protocol::canonical::{CanonicalStopReason, CanonicalStreamEvent, CanonicalUsage};
use crate::protocol::gemini::{GeminiPart, GeminiResponse};
use crate::protocol::mapping::{canonical_stop_to_gemini, gemini_stop_to_canonical};
use crate::util::{
    next_call_id, parse_sse_data_json_line, push_json_string_escaped, push_u64_decimal,
};

/// Parse a single SSE line from a Gemini stream into a `GeminiResponse`.
///
/// Gemini streams as `data: {JSON}\n\n`. Each line prefixed with `data: `
/// contains a full `GeminiResponse` JSON object.
#[must_use]
pub fn parse_gemini_sse_line(line: &str) -> Option<GeminiResponse> {
    parse_sse_data_json_line(line, true, false, true)
}

/// Decode a Gemini stream chunk (one `GeminiResponse`) into canonical stream events.
#[must_use]
pub fn decode_gemini_stream_chunk(chunk: &GeminiResponse) -> Vec<CanonicalStreamEvent> {
    let mut events = Vec::with_capacity(6);
    decode_gemini_stream_chunk_into(chunk, &mut events);
    events
}

/// Decode a Gemini stream chunk into a caller-provided canonical events buffer.
pub fn decode_gemini_stream_chunk_into(
    chunk: &GeminiResponse,
    out: &mut Vec<CanonicalStreamEvent>,
) {
    if let Some(candidates) = &chunk.candidates {
        if let Some(candidate) = candidates.first() {
            let mut has_tool_calls = false;
            for (idx, part) in candidate.content.parts.iter().enumerate() {
                match part {
                    GeminiPart::Text(t) => {
                        out.push(CanonicalStreamEvent::TextDelta(t.clone()));
                    }
                    GeminiPart::FunctionCall { name, args } => {
                        has_tool_calls = true;
                        let call_id = next_call_id();
                        out.push(CanonicalStreamEvent::ToolCallStart {
                            index: idx,
                            id: call_id,
                            name: name.clone(),
                        });
                        // Emit the full args as a single delta.
                        let args_str = serde_json::to_string(args).unwrap_or_else(|_| "{}".into());
                        out.push(CanonicalStreamEvent::ToolCallArgsDelta {
                            index: idx,
                            delta: args_str,
                        });
                        out.push(CanonicalStreamEvent::ToolCallEnd {
                            index: idx,
                            call_id: None,
                            call_name: None,
                        });
                        // no call_id/call_name for function call completion in Gemini stream:
                        // canonical stream does not carry upstream call metadata.
                    }
                    GeminiPart::FunctionResponse { .. } | GeminiPart::InlineData { .. } => {
                        // Not expected in streaming response chunks; skip.
                    }
                }
            }

            // Handle finish reason.
            if let Some(fr) = &candidate.finish_reason {
                let base = gemini_stop_to_canonical(fr);
                // Check if we emitted any tool calls to decide ToolCalls vs EndOfTurn.
                let stop_reason = if base == CanonicalStopReason::EndOfTurn && has_tool_calls {
                    CanonicalStopReason::ToolCalls
                } else {
                    base
                };
                out.push(CanonicalStreamEvent::MessageEnd { stop_reason });
            }
        }
    }

    // Handle usage metadata.
    if let Some(um) = &chunk.usage_metadata {
        out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
            input_tokens: um.prompt_token_count,
            output_tokens: um.candidates_token_count,
            total_tokens: um.total_token_count,
        }));
    }
}

/// Decode a Gemini stream chunk by value into a caller-provided canonical
/// events buffer.
///
/// This variant avoids `String` clones when callers already own the decoded
/// chunk value.
pub fn decode_gemini_stream_chunk_owned_into(
    chunk: GeminiResponse,
    out: &mut Vec<CanonicalStreamEvent>,
) {
    if let Some(candidates) = chunk.candidates {
        if let Some(candidate) = candidates.into_iter().next() {
            let mut has_tool_calls = false;
            for (idx, part) in candidate.content.parts.into_iter().enumerate() {
                match part {
                    GeminiPart::Text(text) => {
                        out.push(CanonicalStreamEvent::TextDelta(text));
                    }
                    GeminiPart::FunctionCall { name, args } => {
                        has_tool_calls = true;
                        let call_id = next_call_id();
                        out.push(CanonicalStreamEvent::ToolCallStart {
                            index: idx,
                            id: call_id,
                            name,
                        });
                        out.push(CanonicalStreamEvent::ToolCallArgsDelta {
                            index: idx,
                            delta: args.to_string(),
                        });
                        out.push(CanonicalStreamEvent::ToolCallEnd {
                            index: idx,
                            call_id: None,
                            call_name: None,
                        });
                    }
                    GeminiPart::FunctionResponse { .. } | GeminiPart::InlineData { .. } => {}
                }
            }

            if let Some(finish_reason) = candidate.finish_reason {
                let base = gemini_stop_to_canonical(&finish_reason);
                let stop_reason = if base == CanonicalStopReason::EndOfTurn && has_tool_calls {
                    CanonicalStopReason::ToolCalls
                } else {
                    base
                };
                out.push(CanonicalStreamEvent::MessageEnd { stop_reason });
            }
        }
    }

    if let Some(usage) = chunk.usage_metadata {
        out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
            input_tokens: usage.prompt_token_count,
            output_tokens: usage.candidates_token_count,
            total_tokens: usage.total_token_count,
        }));
    }
}

/// Encode a canonical stream event into a Gemini SSE line.
///
/// Returns `None` for events that have no Gemini SSE representation.
#[must_use]
pub fn encode_canonical_event_to_gemini_sse(event: &CanonicalStreamEvent) -> Option<String> {
    let mut call_name_by_id = FxHashMap::default();
    encode_canonical_event_to_gemini_sse_with_bindings(event, &mut call_name_by_id)
}

/// Encode a canonical stream event into a Gemini SSE line with call-id bindings.
///
/// `call_name_by_id` tracks `tool_call_id -> function_name` so `ToolResult` can be
/// emitted with the correct Gemini `functionResponse.name`.
pub fn encode_canonical_event_to_gemini_sse_with_bindings<S>(
    event: &CanonicalStreamEvent,
    call_name_by_id: &mut HashMap<String, String, S>,
) -> Option<String>
where
    S: std::hash::BuildHasher,
{
    match event {
        CanonicalStreamEvent::MessageStart { .. } => {
            // Gemini doesn't have an explicit message start event.
            None
        }
        CanonicalStreamEvent::TextDelta(text) | CanonicalStreamEvent::ReasoningDelta(text) => {
            Some(encode_gemini_text_delta_sse(text))
        }
        CanonicalStreamEvent::ToolCallStart { id, name, .. } => {
            call_name_by_id.insert(id.clone(), name.clone());
            // Emit a function call chunk. Args will follow in ToolCallArgsDelta.
            Some(encode_gemini_tool_call_start_sse(name))
        }
        CanonicalStreamEvent::ToolCallArgsDelta { .. } => {
            // In Gemini, function call args come as a complete object within the FunctionCall part.
            // We already emitted an empty args in ToolCallStart; in practice Gemini sends
            // complete function calls in one chunk. Skip incremental deltas.
            None
        }
        CanonicalStreamEvent::ToolCallEnd { .. } => {
            // No separate end event in Gemini streaming.
            None
        }
        CanonicalStreamEvent::Usage(usage) => Some(encode_gemini_usage_sse(usage)),
        CanonicalStreamEvent::MessageEnd { stop_reason } => Some(encode_gemini_message_end_sse(
            canonical_stop_to_gemini(*stop_reason),
        )),
        CanonicalStreamEvent::ToolResult {
            tool_call_id,
            content,
        } => {
            if let Some(function_name) = call_name_by_id.remove(tool_call_id) {
                Some(encode_gemini_tool_result_sse(&function_name, content))
            } else {
                Some(encode_gemini_tool_result_sse(tool_call_id, content))
            }
        }
        CanonicalStreamEvent::Done => None,
        CanonicalStreamEvent::Error { status, message } => {
            Some(encode_gemini_error_sse(u64::from(*status), message))
        }
    }
}

fn encode_gemini_text_delta_sse(text: &str) -> String {
    let mut out = String::with_capacity(64 + text.len());
    out.push_str("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":");
    push_json_string_escaped(&mut out, text);
    out.push_str("}]},\"index\":0}]}\n\n");
    out
}

fn encode_gemini_message_end_sse(stop_reason: &str) -> String {
    let mut out = String::with_capacity(88 + stop_reason.len());
    out.push_str(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[]},\"finishReason\":",
    );
    push_json_string_escaped(&mut out, stop_reason);
    out.push_str(",\"index\":0}]}\n\n");
    out
}

fn encode_gemini_tool_call_start_sse(name: &str) -> String {
    let mut out = String::with_capacity(96 + name.len());
    out.push_str(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":",
    );
    push_json_string_escaped(&mut out, name);
    out.push_str(",\"args\":{}}]},\"index\":0}]}\n\n");
    out
}

fn encode_gemini_usage_sse(usage: &CanonicalUsage) -> String {
    let mut out = String::with_capacity(112);
    out.push_str("data: {\"usageMetadata\":{");
    let mut wrote_any = false;
    if let Some(prompt) = usage.input_tokens {
        out.push_str("\"promptTokenCount\":");
        push_u64_decimal(&mut out, prompt);
        wrote_any = true;
    }
    if let Some(candidates) = usage.output_tokens {
        if wrote_any {
            out.push(',');
        }
        out.push_str("\"candidatesTokenCount\":");
        push_u64_decimal(&mut out, candidates);
        wrote_any = true;
    }
    if let Some(total) = usage.total_tokens {
        if wrote_any {
            out.push(',');
        }
        out.push_str("\"totalTokenCount\":");
        push_u64_decimal(&mut out, total);
    }
    out.push_str("}}\n\n");
    out
}

fn encode_gemini_tool_result_sse(function_name: &str, content: &str) -> String {
    let mut out = String::with_capacity(120 + function_name.len() + content.len());
    out.push_str(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionResponse\":{\"name\":",
    );
    push_json_string_escaped(&mut out, function_name);
    out.push_str(",\"response\":{\"result\":");
    push_json_string_escaped(&mut out, content);
    out.push_str("}}}]},\"index\":0}]}\n\n");
    out
}

fn encode_gemini_error_sse(status: u64, message: &str) -> String {
    let mut out = String::with_capacity(72 + message.len());
    out.push_str("data: {\"error\":{\"code\":");
    push_u64_decimal(&mut out, status);
    out.push_str(",\"message\":");
    push_json_string_escaped(&mut out, message);
    out.push_str(",\"status\":\"INTERNAL\"}}\n\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::gemini::{GeminiCandidate, GeminiContent, GeminiUsageMetadata};

    #[test]
    fn test_parse_sse_line_data() {
        let line = r#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]},"index":0}]}"#;
        let resp = parse_gemini_sse_line(line).unwrap();
        let cand = resp.candidates.unwrap();
        match &cand[0].content.parts[0] {
            GeminiPart::Text(t) => assert_eq!(t, "Hi"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_sse_line_done() {
        let line = "data: [DONE]";
        assert!(parse_gemini_sse_line(line).is_none());
    }

    #[test]
    fn test_parse_sse_line_empty() {
        assert!(parse_gemini_sse_line("").is_none());
        assert!(parse_gemini_sse_line(": comment").is_none());
        assert!(parse_gemini_sse_line("event: message").is_none());
    }

    #[test]
    fn test_decode_stream_chunk_text() {
        let chunk = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: GeminiContent {
                    role: Some("model".into()),
                    parts: vec![GeminiPart::Text("Hello".into())],
                },
                finish_reason: None,
                index: Some(0),
            }]),
            usage_metadata: None,
            model_version: None,
        };

        let events = decode_gemini_stream_chunk(&chunk);
        assert_eq!(events.len(), 1);
        match &events[0] {
            CanonicalStreamEvent::TextDelta(t) => assert_eq!(t, "Hello"),
            other => panic!("expected TextDelta, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_stream_chunk_finish() {
        let chunk = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: GeminiContent {
                    role: Some("model".into()),
                    parts: vec![],
                },
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }]),
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(20),
                total_token_count: Some(30),
            }),
            model_version: None,
        };

        let events = decode_gemini_stream_chunk(&chunk);
        // Should have MessageEnd + Usage
        assert!(events.iter().any(|e| matches!(
            e,
            CanonicalStreamEvent::MessageEnd {
                stop_reason: CanonicalStopReason::EndOfTurn
            }
        )));
        assert!(events
            .iter()
            .any(|e| matches!(e, CanonicalStreamEvent::Usage(_))));
    }

    #[test]
    fn test_encode_text_delta_to_sse() {
        let event = CanonicalStreamEvent::TextDelta("Hi there".into());
        let sse = encode_canonical_event_to_gemini_sse(&event).unwrap();
        assert!(sse.starts_with("data: "));
        assert!(sse.contains("Hi there"));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn test_encode_message_end_to_sse() {
        let event = CanonicalStreamEvent::MessageEnd {
            stop_reason: CanonicalStopReason::MaxTokens,
        };
        let sse = encode_canonical_event_to_gemini_sse(&event).unwrap();
        assert!(sse.contains("MAX_TOKENS"));
    }

    #[test]
    fn test_encode_done_returns_none() {
        let event = CanonicalStreamEvent::Done;
        assert!(encode_canonical_event_to_gemini_sse(&event).is_none());
    }
}
