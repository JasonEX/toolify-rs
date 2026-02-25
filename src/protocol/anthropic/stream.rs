use crate::protocol::anthropic::{AnthropicContentBlock, AnthropicDelta, AnthropicStreamEvent};
use crate::protocol::canonical::{CanonicalRole, CanonicalStreamEvent, CanonicalUsage};
use crate::protocol::mapping::{anthropic_stop_to_canonical, canonical_stop_to_anthropic};
use crate::util::{push_json_string_escaped, push_usize_decimal};

/// Parse an Anthropic SSE named event into a typed stream event.
///
/// Anthropic uses named events: `event: message_start\ndata: {...}\n\n`
/// The `event_type` is the value after "event:", and `data` is the JSON payload.
#[must_use]
pub fn parse_anthropic_sse(event_type: &str, data: &str) -> Option<AnthropicStreamEvent> {
    parse_anthropic_sse_bytes(event_type, data.as_bytes())
}

/// Parse an Anthropic SSE named event from raw JSON bytes.
#[must_use]
pub fn parse_anthropic_sse_bytes(event_type: &str, data: &[u8]) -> Option<AnthropicStreamEvent> {
    match event_type {
        "message_start"
        | "content_block_start"
        | "content_block_delta"
        | "content_block_stop"
        | "message_delta"
        | "message_stop"
        | "ping"
        | "error" => serde_json::from_slice(data).ok(),
        _ => None,
    }
}

/// Decode an Anthropic stream event into zero or more canonical stream events.
#[must_use]
pub fn decode_anthropic_stream_event(event: &AnthropicStreamEvent) -> Vec<CanonicalStreamEvent> {
    let mut events = Vec::new();
    decode_anthropic_stream_event_into(event, &mut events);
    events
}

/// Decode an Anthropic stream event into a caller-provided canonical events buffer.
pub fn decode_anthropic_stream_event_into(
    event: &AnthropicStreamEvent,
    out: &mut Vec<CanonicalStreamEvent>,
) {
    match event {
        AnthropicStreamEvent::MessageStart { message } => {
            out.push(CanonicalStreamEvent::MessageStart {
                role: CanonicalRole::Assistant,
            });
            // Emit initial usage if present
            let usage = &message.usage;
            if usage.input_tokens > 0 || usage.output_tokens > 0 {
                out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
                    input_tokens: Some(usage.input_tokens),
                    output_tokens: Some(usage.output_tokens),
                    total_tokens: Some(usage.input_tokens + usage.output_tokens),
                }));
            }
        }
        AnthropicStreamEvent::ContentBlockStart {
            index,
            content_block,
        } => match content_block {
            AnthropicContentBlock::Text { text } => {
                if !text.is_empty() {
                    out.push(CanonicalStreamEvent::TextDelta(text.clone()));
                }
            }
            AnthropicContentBlock::ToolUse { id, name, .. } => {
                out.push(CanonicalStreamEvent::ToolCallStart {
                    index: *index,
                    id: id.clone(),
                    name: name.clone(),
                });
            }
            AnthropicContentBlock::Thinking { thinking } => {
                if !thinking.is_empty() {
                    out.push(CanonicalStreamEvent::ReasoningDelta(thinking.clone()));
                }
            }
            AnthropicContentBlock::ToolResult { .. } => {}
        },
        AnthropicStreamEvent::ContentBlockDelta { index, delta } => match delta {
            AnthropicDelta::TextDelta { text } => {
                out.push(CanonicalStreamEvent::TextDelta(text.clone()));
            }
            AnthropicDelta::ThinkingDelta { thinking } => {
                out.push(CanonicalStreamEvent::ReasoningDelta(thinking.clone()));
            }
            AnthropicDelta::InputJsonDelta { partial_json } => {
                out.push(CanonicalStreamEvent::ToolCallArgsDelta {
                    index: *index,
                    delta: partial_json.clone(),
                });
            }
        },
        AnthropicStreamEvent::ContentBlockStop { .. } | AnthropicStreamEvent::Ping {} => {
            // Block stop is handled by StatefulAnthropicStreamDecoder which
            // tracks block types and only emits ToolCallEnd for tool_use blocks.
        }
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            // Usage update
            out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
                input_tokens: Some(usage.input_tokens),
                output_tokens: Some(usage.output_tokens),
                total_tokens: Some(usage.input_tokens + usage.output_tokens),
            }));
            // Stop reason → MessageEnd
            if let Some(reason_str) = &delta.stop_reason {
                out.push(CanonicalStreamEvent::MessageEnd {
                    stop_reason: anthropic_stop_to_canonical(reason_str),
                });
            }
        }
        AnthropicStreamEvent::MessageStop {} => {
            out.push(CanonicalStreamEvent::Done);
        }
        AnthropicStreamEvent::Error { error } => {
            out.push(CanonicalStreamEvent::Error {
                status: 500,
                message: error.message.clone(),
            });
        }
    }
}

/// Decode an Anthropic stream event by value into a caller-provided canonical
/// events buffer.
///
/// This variant avoids `String` clones when callers already own the decoded
/// wire event (for example after `serde_json::from_str` in streaming fallback).
pub fn decode_anthropic_stream_event_owned_into(
    event: AnthropicStreamEvent,
    out: &mut Vec<CanonicalStreamEvent>,
) {
    match event {
        AnthropicStreamEvent::MessageStart { message } => {
            out.push(CanonicalStreamEvent::MessageStart {
                role: CanonicalRole::Assistant,
            });
            let usage = message.usage;
            if usage.input_tokens > 0 || usage.output_tokens > 0 {
                out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
                    input_tokens: Some(usage.input_tokens),
                    output_tokens: Some(usage.output_tokens),
                    total_tokens: Some(usage.input_tokens + usage.output_tokens),
                }));
            }
        }
        AnthropicStreamEvent::ContentBlockStart {
            index,
            content_block,
        } => match content_block {
            AnthropicContentBlock::Text { text } => {
                if !text.is_empty() {
                    out.push(CanonicalStreamEvent::TextDelta(text));
                }
            }
            AnthropicContentBlock::ToolUse { id, name, .. } => {
                out.push(CanonicalStreamEvent::ToolCallStart { index, id, name });
            }
            AnthropicContentBlock::Thinking { thinking } => {
                if !thinking.is_empty() {
                    out.push(CanonicalStreamEvent::ReasoningDelta(thinking));
                }
            }
            AnthropicContentBlock::ToolResult { .. } => {}
        },
        AnthropicStreamEvent::ContentBlockDelta { index, delta } => match delta {
            AnthropicDelta::TextDelta { text } => {
                out.push(CanonicalStreamEvent::TextDelta(text));
            }
            AnthropicDelta::ThinkingDelta { thinking } => {
                out.push(CanonicalStreamEvent::ReasoningDelta(thinking));
            }
            AnthropicDelta::InputJsonDelta { partial_json } => {
                out.push(CanonicalStreamEvent::ToolCallArgsDelta {
                    index,
                    delta: partial_json,
                });
            }
        },
        AnthropicStreamEvent::ContentBlockStop { .. } | AnthropicStreamEvent::Ping {} => {
            // Block stop is handled by StatefulAnthropicStreamDecoder which
            // tracks block types and only emits ToolCallEnd for tool_use blocks.
        }
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
                input_tokens: Some(usage.input_tokens),
                output_tokens: Some(usage.output_tokens),
                total_tokens: Some(usage.input_tokens + usage.output_tokens),
            }));
            if let Some(reason_str) = delta.stop_reason {
                out.push(CanonicalStreamEvent::MessageEnd {
                    stop_reason: anthropic_stop_to_canonical(&reason_str),
                });
            }
        }
        AnthropicStreamEvent::MessageStop {} => {
            out.push(CanonicalStreamEvent::Done);
        }
        AnthropicStreamEvent::Error { error } => {
            out.push(CanonicalStreamEvent::Error {
                status: 500,
                message: error.message,
            });
        }
    }
}

/// Stateful Anthropic stream decoder that tracks content block types.
///
/// The raw `decode_anthropic_stream_event` function is stateless and cannot
/// distinguish `content_block_stop` for text blocks vs `tool_use` blocks.
/// This wrapper records block types on `ContentBlockStart` so that
/// `ContentBlockStop` only emits `ToolCallEnd` for `tool_use` blocks.
pub struct StatefulAnthropicStreamDecoder {
    block_types: Vec<Option<BlockType>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockType {
    Text,
    Thinking,
    ToolUse,
    ToolResult,
}

impl StatefulAnthropicStreamDecoder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            block_types: Vec::new(),
        }
    }

    pub fn decode(&mut self, event: &AnthropicStreamEvent) -> Vec<CanonicalStreamEvent> {
        let mut events = Vec::new();
        self.decode_into(event, &mut events);
        events
    }

    pub fn decode_into(
        &mut self,
        event: &AnthropicStreamEvent,
        out: &mut Vec<CanonicalStreamEvent>,
    ) {
        match event {
            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                let block_type = match content_block {
                    AnthropicContentBlock::Text { .. } => BlockType::Text,
                    AnthropicContentBlock::Thinking { .. } => BlockType::Thinking,
                    AnthropicContentBlock::ToolUse { .. } => BlockType::ToolUse,
                    AnthropicContentBlock::ToolResult { .. } => BlockType::ToolResult,
                };
                if *index >= self.block_types.len() {
                    self.block_types.resize(*index + 1, None);
                }
                self.block_types[*index] = Some(block_type);
                decode_anthropic_stream_event_into(event, out);
            }
            AnthropicStreamEvent::ContentBlockStop { index } => {
                let block_type = self.block_types.get_mut(*index).and_then(Option::take);
                if block_type == Some(BlockType::ToolUse) {
                    out.push(CanonicalStreamEvent::ToolCallEnd {
                        index: *index,
                        call_id: None,
                        call_name: None,
                    });
                }
            }
            _ => decode_anthropic_stream_event_into(event, out),
        }
    }

    pub fn decode_owned_into(
        &mut self,
        event: AnthropicStreamEvent,
        out: &mut Vec<CanonicalStreamEvent>,
    ) {
        match event {
            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                let block_type = match &content_block {
                    AnthropicContentBlock::Text { .. } => BlockType::Text,
                    AnthropicContentBlock::Thinking { .. } => BlockType::Thinking,
                    AnthropicContentBlock::ToolUse { .. } => BlockType::ToolUse,
                    AnthropicContentBlock::ToolResult { .. } => BlockType::ToolResult,
                };
                if index >= self.block_types.len() {
                    self.block_types.resize(index + 1, None);
                }
                self.block_types[index] = Some(block_type);
                decode_anthropic_stream_event_owned_into(
                    AnthropicStreamEvent::ContentBlockStart {
                        index,
                        content_block,
                    },
                    out,
                );
            }
            AnthropicStreamEvent::ContentBlockStop { index } => {
                let block_type = self.block_types.get_mut(index).and_then(Option::take);
                if block_type == Some(BlockType::ToolUse) {
                    out.push(CanonicalStreamEvent::ToolCallEnd {
                        index,
                        call_id: None,
                        call_name: None,
                    });
                }
            }
            _ => decode_anthropic_stream_event_owned_into(event, out),
        }
    }

    /// Register content block kind in fast-path decoders that bypass serde.
    ///
    /// Only `tool_use` blocks need to be preserved semantically for later
    /// `content_block_stop` -> `ToolCallEnd` conversion.
    pub(crate) fn set_block_type_for_fast_path(&mut self, index: usize, is_tool_use: bool) {
        if index >= self.block_types.len() {
            self.block_types.resize(index + 1, None);
        }
        self.block_types[index] = Some(if is_tool_use {
            BlockType::ToolUse
        } else {
            BlockType::Text
        });
    }
}

impl Default for StatefulAnthropicStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode a canonical stream event into one or more Anthropic SSE (`event_type`, `json_data`) pairs.
///
/// `model` and `id` are used to fill in the `message_start` envelope.
/// Returns a vec of (`event_name`, `json_data`) tuples suitable for writing as
/// `event: {name}\ndata: {json}\n\n`.
#[must_use]
pub fn encode_canonical_event_to_anthropic_sse(
    event: &CanonicalStreamEvent,
    model: &str,
    id: &str,
) -> Vec<(&'static str, String)> {
    let mut out = Vec::with_capacity(2);
    encode_canonical_event_to_anthropic_sse_into(event, model, id, &mut out);
    out
}

/// Encode a canonical stream event into a caller-provided Anthropic SSE pair buffer.
pub fn encode_canonical_event_to_anthropic_sse_into(
    event: &CanonicalStreamEvent,
    model: &str,
    id: &str,
    out: &mut Vec<(&'static str, String)>,
) {
    out.clear();
    match event {
        CanonicalStreamEvent::MessageStart { role: _ } => {
            let mut json = String::with_capacity(120 + id.len() + model.len());
            json.push_str("{\"type\":\"message_start\",\"message\":{\"id\":");
            push_json_string_escaped(&mut json, id);
            json.push_str(",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
            push_json_string_escaped(&mut json, model);
            json.push_str(",\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}");
            out.push(("message_start", json));
        }
        CanonicalStreamEvent::TextDelta(text) => {
            let mut json = String::with_capacity(76 + text.len());
            json.push_str("{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":");
            push_json_string_escaped(&mut json, text);
            json.push_str("}}");
            out.push(("content_block_delta", json));
        }
        CanonicalStreamEvent::ReasoningDelta(text) => {
            let mut json = String::with_capacity(84 + text.len());
            json.push_str(
                "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":",
            );
            push_json_string_escaped(&mut json, text);
            json.push_str("}}");
            out.push(("content_block_delta", json));
        }
        CanonicalStreamEvent::ToolCallStart {
            index,
            id: call_id,
            name,
        } => {
            let mut json = String::with_capacity(112 + call_id.len() + name.len());
            json.push_str("{\"type\":\"content_block_start\",\"index\":");
            push_usize_decimal(&mut json, *index);
            json.push_str(",\"content_block\":{\"type\":\"tool_use\",\"id\":");
            push_json_string_escaped(&mut json, call_id);
            json.push_str(",\"name\":");
            push_json_string_escaped(&mut json, name);
            json.push_str(",\"input\":{}}}");
            out.push(("content_block_start", json));
        }
        CanonicalStreamEvent::ToolCallArgsDelta { index, delta } => {
            let mut json = String::with_capacity(96 + delta.len());
            json.push_str("{\"type\":\"content_block_delta\",\"index\":");
            push_usize_decimal(&mut json, *index);
            json.push_str(",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":");
            push_json_string_escaped(&mut json, delta);
            json.push_str("}}");
            out.push(("content_block_delta", json));
        }
        CanonicalStreamEvent::ToolCallEnd { index, .. } => {
            let mut json = String::with_capacity(48);
            json.push_str("{\"type\":\"content_block_stop\",\"index\":");
            push_usize_decimal(&mut json, *index);
            json.push('}');
            out.push(("content_block_stop", json));
        }
        CanonicalStreamEvent::Usage(_usage) => {
            // Usage is typically bundled with message_delta; emit standalone as ping placeholder
        }
        CanonicalStreamEvent::MessageEnd { stop_reason } => {
            let mut json = String::with_capacity(112);
            json.push_str("{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":");
            push_json_string_escaped(&mut json, canonical_stop_to_anthropic(*stop_reason));
            json.push_str(
                ",\"stop_sequence\":null},\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}",
            );
            out.push(("message_delta", json));
        }
        CanonicalStreamEvent::Done => {
            out.push(("message_stop", "{\"type\":\"message_stop\"}".to_string()));
        }
        CanonicalStreamEvent::ToolResult {
            tool_call_id,
            content,
        } => {
            let mut json = String::with_capacity(112 + tool_call_id.len() + content.len());
            json.push_str("{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_result\",\"tool_use_id\":");
            push_json_string_escaped(&mut json, tool_call_id);
            json.push_str(",\"content\":");
            push_json_string_escaped(&mut json, content);
            json.push_str("}}");
            out.push(("content_block_start", json));
        }
        CanonicalStreamEvent::Error { status: _, message } => {
            let mut json = String::with_capacity(48 + message.len());
            json.push_str("{\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":");
            push_json_string_escaped(&mut json, message);
            json.push_str("}}");
            out.push(("error", json));
        }
    }
}

/// Encode a canonical stream event directly into a full Anthropic SSE frame.
///
/// Returns `true` when a frame is produced and written into `out`.
pub fn encode_canonical_event_to_anthropic_sse_frame(
    event: &CanonicalStreamEvent,
    model: &str,
    id: &str,
    out: &mut String,
) -> bool {
    out.clear();
    match event {
        CanonicalStreamEvent::MessageStart { role: _ } => {
            out.push_str(
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":",
            );
            push_json_string_escaped(out, id);
            out.push_str(",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
            push_json_string_escaped(out, model);
            out.push_str(",\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n");
            true
        }
        CanonicalStreamEvent::TextDelta(text) => {
            out.push_str("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":");
            push_json_string_escaped(out, text);
            out.push_str("}}\n\n");
            true
        }
        CanonicalStreamEvent::ReasoningDelta(text) => {
            out.push_str(
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":",
            );
            push_json_string_escaped(out, text);
            out.push_str("}}\n\n");
            true
        }
        CanonicalStreamEvent::ToolCallStart {
            index,
            id: call_id,
            name,
        } => {
            out.push_str(
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":",
            );
            push_usize_decimal(out, *index);
            out.push_str(",\"content_block\":{\"type\":\"tool_use\",\"id\":");
            push_json_string_escaped(out, call_id);
            out.push_str(",\"name\":");
            push_json_string_escaped(out, name);
            out.push_str(",\"input\":{}}}\n\n");
            true
        }
        CanonicalStreamEvent::ToolCallArgsDelta { index, delta } => {
            out.push_str(
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":",
            );
            push_usize_decimal(out, *index);
            out.push_str(",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":");
            push_json_string_escaped(out, delta);
            out.push_str("}}\n\n");
            true
        }
        CanonicalStreamEvent::ToolCallEnd { index, .. } => {
            out.push_str(
                "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":",
            );
            push_usize_decimal(out, *index);
            out.push_str("}\n\n");
            true
        }
        CanonicalStreamEvent::Usage(_) => false,
        CanonicalStreamEvent::MessageEnd { stop_reason } => {
            out.push_str("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":");
            push_json_string_escaped(out, canonical_stop_to_anthropic(*stop_reason));
            out.push_str(
                ",\"stop_sequence\":null},\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}\n\n",
            );
            true
        }
        CanonicalStreamEvent::Done => {
            out.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
            true
        }
        CanonicalStreamEvent::ToolResult {
            tool_call_id,
            content,
        } => {
            out.push_str("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_result\",\"tool_use_id\":");
            push_json_string_escaped(out, tool_call_id);
            out.push_str(",\"content\":");
            push_json_string_escaped(out, content);
            out.push_str("}}\n\n");
            true
        }
        CanonicalStreamEvent::Error { status: _, message } => {
            out.push_str("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":");
            push_json_string_escaped(out, message);
            out.push_str("}}\n\n");
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_text_delta_frame_matches_pair_render() {
        let event = CanonicalStreamEvent::TextDelta("hello".into());
        let pairs = encode_canonical_event_to_anthropic_sse(&event, "claude-3-5-sonnet", "msg_1");
        assert_eq!(pairs.len(), 1);
        let expected = format!("event: {}\ndata: {}\n\n", pairs[0].0, pairs[0].1);

        let mut frame = String::new();
        let produced = encode_canonical_event_to_anthropic_sse_frame(
            &event,
            "claude-3-5-sonnet",
            "msg_1",
            &mut frame,
        );
        assert!(produced);
        assert_eq!(frame, expected);
    }

    #[test]
    fn test_encode_done_frame_matches_pair_render() {
        let event = CanonicalStreamEvent::Done;
        let pairs = encode_canonical_event_to_anthropic_sse(&event, "claude-3-5-sonnet", "msg_1");
        assert_eq!(pairs.len(), 1);
        let expected = format!("event: {}\ndata: {}\n\n", pairs[0].0, pairs[0].1);

        let mut frame = String::new();
        let produced = encode_canonical_event_to_anthropic_sse_frame(
            &event,
            "claude-3-5-sonnet",
            "msg_1",
            &mut frame,
        );
        assert!(produced);
        assert_eq!(frame, expected);
    }
}
