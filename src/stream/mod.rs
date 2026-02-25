pub mod sse;
pub mod transcoder;

pub use sse::{sse_frame_stream, SseFrame, SseParser};
pub use transcoder::StreamTranscoder;

use crate::error::CanonicalError;
use crate::fc::detector::{DetectorAction, DetectorState, StreamingFcDetector};
use crate::fc::parser::parse_function_calls;
use crate::fc::parser::ParsedToolCall;
use crate::protocol::canonical::{CanonicalStopReason, CanonicalStreamEvent, CanonicalToolSpec};
use crate::util::{next_call_id, push_json_string_escaped, push_u64_decimal};
use memchr::{memchr, memchr2};

/// A parsed SSE frame from the upstream.
#[derive(Debug, Clone, Default)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
    pub id: Option<String>,
    pub retry: Option<u64>,
}

/// Parse a single SSE frame from raw text.
///
/// Feeds the raw text through an [`sse::SseParser`] and returns the first
/// complete event, if any.
#[must_use]
pub fn parse_sse_frame(raw: &str) -> Option<SseEvent> {
    parse_sse_frame_bytes(raw.as_bytes())
}

/// Parse a single SSE frame from raw bytes.
///
/// Supports `\n\n` and `\r\n\r\n` separators and standard SSE fields
/// (`event`, `data`, `id`, `retry`).
#[must_use]
pub fn parse_sse_frame_bytes(raw: &[u8]) -> Option<SseEvent> {
    if let Some(frame) = try_parse_data_only_sse_frame(raw) {
        return Some(frame);
    }
    if let Some(frame) = try_parse_event_and_data_sse_frame(raw) {
        return Some(frame);
    }

    let mut event: Option<String> = None;
    let mut data = String::new();
    let mut has_data = false;
    let mut id: Option<String> = None;
    let mut retry: Option<u64> = None;
    let mut line_start = 0usize;

    while let Some(rel_pos) = memchr(b'\n', &raw[line_start..]) {
        let line_end = line_start + rel_pos;
        let mut line = &raw[line_start..line_end];
        if line.last().copied() == Some(b'\r') {
            line = &line[..line.len() - 1];
        }

        if line.is_empty() {
            if has_data {
                return Some(SseEvent {
                    event,
                    data,
                    id,
                    retry,
                });
            }
            line_start = line_end + 1;
            continue;
        }

        if line.first().copied() == Some(b':') {
            line_start = line_end + 1;
            continue;
        }

        if let Some(value) = line.strip_prefix(b"data:") {
            let value = value.strip_prefix(b" ").unwrap_or(value);
            let value = std::str::from_utf8(value).ok()?;
            if has_data {
                data.push('\n');
            } else {
                has_data = true;
            }
            data.push_str(value);
        } else if let Some(value) = line.strip_prefix(b"event:") {
            let value = value.strip_prefix(b" ").unwrap_or(value);
            let value = std::str::from_utf8(value).ok()?;
            event = Some(value.to_string());
        } else if let Some(value) = line.strip_prefix(b"id:") {
            let value = value.strip_prefix(b" ").unwrap_or(value);
            let value = std::str::from_utf8(value).ok()?;
            id = Some(value.to_string());
        } else if let Some(value) = line.strip_prefix(b"retry:") {
            let value = value.strip_prefix(b" ").unwrap_or(value);
            let value = std::str::from_utf8(value).ok()?;
            retry = value.trim().parse::<u64>().ok();
        }
        line_start = line_end + 1;
    }

    None
}

#[inline]
fn frame_payload_end(raw: &[u8]) -> Option<usize> {
    if raw.ends_with(b"\r\n\r\n") {
        Some(raw.len().saturating_sub(4))
    } else if raw.ends_with(b"\n\n") {
        Some(raw.len().saturating_sub(2))
    } else {
        None
    }
}

#[inline]
fn try_parse_data_only_sse_frame(raw: &[u8]) -> Option<SseEvent> {
    if !raw.starts_with(b"data:") {
        return None;
    }
    let end = frame_payload_end(raw)?;
    if end < 5 {
        return None;
    }
    let start = 5 + usize::from(raw.get(5) == Some(&b' '));
    let data_bytes = raw.get(start..end)?;
    if memchr2(b'\n', b'\r', data_bytes).is_some() {
        return None;
    }
    let data = std::str::from_utf8(data_bytes).ok()?.to_string();
    Some(SseEvent {
        event: None,
        data,
        id: None,
        retry: None,
    })
}

#[inline]
fn try_parse_event_and_data_sse_frame(raw: &[u8]) -> Option<SseEvent> {
    if !raw.starts_with(b"event:") {
        return None;
    }
    let end = frame_payload_end(raw)?;
    let first_newline = memchr(b'\n', raw)?;
    if first_newline + 1 >= end {
        return None;
    }

    let mut event_line = &raw[..first_newline];
    if event_line.last().copied() == Some(b'\r') {
        event_line = &event_line[..event_line.len() - 1];
    }
    let event_value = event_line.strip_prefix(b"event:")?;
    let event_value = event_value.strip_prefix(b" ").unwrap_or(event_value);
    let event = std::str::from_utf8(event_value).ok()?.to_string();

    let mut data_line = raw.get(first_newline + 1..end)?;
    if memchr(b'\n', data_line).is_some() {
        return None;
    }
    if data_line.last().copied() == Some(b'\r') {
        data_line = &data_line[..data_line.len() - 1];
    }
    let data_value = data_line.strip_prefix(b"data:")?;
    let data_value = data_value.strip_prefix(b" ").unwrap_or(data_value);
    let data = std::str::from_utf8(data_value).ok()?.to_string();

    Some(SseEvent {
        event: Some(event),
        data,
        id: None,
        retry: None,
    })
}

/// Encode a canonical stream event back into an SSE text frame.
///
/// Maps canonical events to their SSE wire representation. Returns
/// `Ok(text)` for events that have an SSE encoding, or an error for
/// events that cannot be encoded.
///
/// # Errors
///
/// Returns [`CanonicalError::Translation`] for canonical events that require
/// provider-specific encoding context.
pub fn encode_sse_event(event: &CanonicalStreamEvent) -> Result<String, CanonicalError> {
    match event {
        CanonicalStreamEvent::Done => Ok(sse::done_frame()),
        CanonicalStreamEvent::Error { status, message } => {
            let mut out = String::with_capacity(40 + message.len());
            out.push_str("data: {\"error\":{\"status\":");
            push_u64_decimal(&mut out, u64::from(*status));
            out.push_str(",\"message\":");
            push_json_string_escaped(&mut out, message);
            out.push_str("}}\n\n");
            Ok(out)
        }
        CanonicalStreamEvent::TextDelta(text) => {
            let mut out = String::with_capacity(32 + text.len());
            out.push_str("data: {\"delta\":{\"text\":");
            push_json_string_escaped(&mut out, text);
            out.push_str("}}\n\n");
            Ok(out)
        }
        CanonicalStreamEvent::ToolResult {
            tool_call_id,
            content,
        } => {
            let mut out = String::with_capacity(64 + tool_call_id.len() + content.len());
            out.push_str("data: {\"delta\":{\"tool_result\":{\"tool_call_id\":");
            push_json_string_escaped(&mut out, tool_call_id);
            out.push_str(",\"content\":");
            push_json_string_escaped(&mut out, content);
            out.push_str("}}}\n\n");
            Ok(out)
        }
        _ => Err(CanonicalError::Translation(format!(
            "cannot encode canonical event {event:?} as generic SSE without provider context"
        ))),
    }
}

// ---------------------------------------------------------------------------
// StreamingFcProcessor
// ---------------------------------------------------------------------------

/// Streaming function-call processor.
///
/// Sits between the upstream SSE stream and the client, using a
/// [`StreamingFcDetector`] to intercept tool-call XML blocks and a
/// [`StreamTranscoder`] to decode/encode events across protocols.
pub struct StreamingFcProcessor {
    detector: StreamingFcDetector,
    transcoder: StreamTranscoder,
    decode_buffer: Vec<CanonicalStreamEvent>,
    fc_enabled: bool,
    pending_stop_reason: Option<CanonicalStopReason>,
    synthesize_termination: bool,
    /// Running tool-call index for emitted tool calls.
    tool_call_index: usize,
}

impl StreamingFcProcessor {
    /// Create a new streaming FC processor.
    ///
    /// - `transcoder`: handles upstream decode and client encode.
    /// - `fc_enabled`: whether to look for injected function calls.
    /// - `trigger_signal`: the FC trigger signal string.
    #[must_use]
    pub fn new(
        transcoder: StreamTranscoder,
        fc_enabled: bool,
        _tools: &[CanonicalToolSpec],
        trigger_signal: &'static str,
    ) -> Self {
        Self {
            detector: StreamingFcDetector::new(trigger_signal),
            transcoder,
            decode_buffer: Vec::with_capacity(8),
            fc_enabled,
            pending_stop_reason: None,
            synthesize_termination: fc_enabled,
            tool_call_index: 0,
        }
    }

    /// Process a single upstream SSE frame and append SSE strings to `output`.
    ///
    /// Pipeline:
    /// 1. Decode the upstream frame into canonical events via the transcoder.
    /// 2. For each canonical event, run it through the FC detector (if enabled)
    ///    or encode it directly.
    pub fn process_frame_into(&mut self, frame: &SseEvent, output: &mut Vec<String>) {
        self.transcoder
            .decode_upstream_frame_into(frame, &mut self.decode_buffer);
        self.process_decoded_events_into(output);
    }

    /// Process a single upstream SSE frame and append SSE bytes to `output`.
    pub fn process_frame_into_bytes(&mut self, frame: &SseEvent, output: &mut Vec<bytes::Bytes>) {
        self.transcoder
            .decode_upstream_frame_into(frame, &mut self.decode_buffer);
        self.process_decoded_events_into_bytes(output);
    }

    /// Process a complete raw SSE frame and append SSE strings to `output`.
    ///
    /// Returns `true` when the frame was parsed as SSE and processed;
    /// `false` when parsing failed and caller should use raw passthrough.
    pub fn try_process_raw_frame_into(
        &mut self,
        raw_frame: &[u8],
        output: &mut Vec<String>,
    ) -> bool {
        if !self
            .transcoder
            .try_decode_upstream_raw_frame_into(raw_frame, &mut self.decode_buffer)
        {
            return false;
        }
        self.process_decoded_events_into(output);
        true
    }

    /// Process a complete raw SSE frame and append SSE bytes to `output`.
    ///
    /// Returns `true` when the frame was parsed as SSE and processed;
    /// `false` when parsing failed and caller should use raw passthrough.
    pub fn try_process_raw_frame_into_bytes(
        &mut self,
        raw_frame: &[u8],
        output: &mut Vec<bytes::Bytes>,
    ) -> bool {
        if !self
            .transcoder
            .try_decode_upstream_raw_frame_into(raw_frame, &mut self.decode_buffer)
        {
            return false;
        }
        self.process_decoded_events_into_bytes(output);
        true
    }

    /// Process a complete raw SSE frame and append SSE strings to `output`.
    pub fn process_raw_frame_into(&mut self, raw_frame: &[u8], output: &mut Vec<String>) {
        let _ = self.try_process_raw_frame_into(raw_frame, output);
    }

    /// Process a complete raw SSE frame and append SSE bytes to `output`.
    pub fn process_raw_frame_into_bytes(
        &mut self,
        raw_frame: &[u8],
        output: &mut Vec<bytes::Bytes>,
    ) {
        let _ = self.try_process_raw_frame_into_bytes(raw_frame, output);
    }

    /// Fast-path processing when caller already extracted an OpenAI-compatible
    /// SSE `data` payload.
    pub fn process_openai_data_frame_into(&mut self, data: &str, output: &mut Vec<String>) {
        self.transcoder
            .decode_openai_data_payload_into(data, &mut self.decode_buffer);
        self.process_decoded_events_into(output);
    }

    /// Fast-path processing when caller already extracted an OpenAI-compatible
    /// SSE `data` payload as raw bytes.
    ///
    /// Returns `true` when bytes were decoded and processed; `false` when the
    /// payload was not decodable and caller should fall back to raw passthrough.
    pub fn try_process_openai_data_frame_bytes_into(
        &mut self,
        data: &[u8],
        output: &mut Vec<String>,
    ) -> bool {
        if !self
            .transcoder
            .try_decode_openai_data_payload_bytes_into(data, &mut self.decode_buffer)
        {
            return false;
        }
        self.process_decoded_events_into(output);
        true
    }

    /// Fast-path processing when caller already extracted an OpenAI-compatible
    /// SSE `data` payload as raw bytes, emitting SSE bytes.
    ///
    /// Returns `true` when bytes were decoded and processed; `false` when the
    /// payload was not decodable and caller should fall back to raw passthrough.
    pub fn try_process_openai_data_frame_bytes_into_bytes(
        &mut self,
        data: &[u8],
        output: &mut Vec<bytes::Bytes>,
    ) -> bool {
        if !self
            .transcoder
            .try_decode_openai_data_payload_bytes_into(data, &mut self.decode_buffer)
        {
            return false;
        }
        self.process_decoded_events_into_bytes(output);
        true
    }

    fn process_decoded_events_into(&mut self, output: &mut Vec<String>) {
        output.clear();
        if self.decode_buffer.len() > output.capacity() {
            output.reserve(self.decode_buffer.len() - output.capacity());
        }

        for event in self.decode_buffer.drain(..) {
            match event {
                CanonicalStreamEvent::TextDelta(text) if self.fc_enabled => {
                    let action = self.detector.feed_owned(text);
                    match action {
                        DetectorAction::PassThrough(pass_text) => {
                            if !pass_text.is_empty() {
                                let ev = CanonicalStreamEvent::TextDelta(pass_text);
                                if let Some(encoded) = self.transcoder.encode_client_event(&ev) {
                                    output.push(encoded);
                                }
                            }
                        }
                        DetectorAction::Buffer => {
                            // Text is buffered in the detector; nothing to send.
                        }
                        DetectorAction::TriggerFound { text_before } => {
                            // Send any text before the trigger to the client.
                            if !text_before.is_empty() {
                                let ev = CanonicalStreamEvent::TextDelta(text_before);
                                if let Some(encoded) = self.transcoder.encode_client_event(&ev) {
                                    output.push(encoded);
                                }
                            }
                            // The rest is buffered for XML parsing — nothing more to send.
                        }
                        DetectorAction::BufferOverflow(overflow_text) => {
                            // Buffer exceeded limit — flush everything as text and
                            // disable FC for the rest of this response.
                            if !overflow_text.is_empty() {
                                let ev = CanonicalStreamEvent::TextDelta(overflow_text);
                                if let Some(encoded) = self.transcoder.encode_client_event(&ev) {
                                    output.push(encoded);
                                }
                            }
                            self.fc_enabled = false;
                            self.synthesize_termination = false;
                            self.pending_stop_reason = None;
                        }
                    }
                }
                CanonicalStreamEvent::MessageEnd { stop_reason }
                    if self.fc_enabled && self.synthesize_termination =>
                {
                    // Suppress upstream stop while FC is active. We'll emit one
                    // synthesized terminal event in finalize().
                    self.pending_stop_reason = Some(stop_reason);
                }
                CanonicalStreamEvent::Done if self.fc_enabled && self.synthesize_termination => {
                    // Suppress upstream done while FC is active. finalize() emits done once.
                }
                CanonicalStreamEvent::TextDelta(_) => {
                    // FC not enabled — forward as-is.
                    if let Some(encoded) = self.transcoder.encode_client_event(&event) {
                        output.push(encoded);
                    }
                }
                // All other events (MessageStart, ToolCallStart, Usage,
                // MessageEnd, Done, Error, etc.) pass through directly.
                _ => {
                    if let Some(encoded) = self.transcoder.encode_client_event(&event) {
                        output.push(encoded);
                    }
                }
            }
        }
    }

    fn process_decoded_events_into_bytes(&mut self, output: &mut Vec<bytes::Bytes>) {
        output.clear();
        if self.decode_buffer.len() > output.capacity() {
            output.reserve(self.decode_buffer.len() - output.capacity());
        }

        for event in self.decode_buffer.drain(..) {
            match event {
                CanonicalStreamEvent::TextDelta(text) if self.fc_enabled => {
                    let action = self.detector.feed_owned(text);
                    match action {
                        DetectorAction::PassThrough(pass_text) => {
                            if !pass_text.is_empty() {
                                let ev = CanonicalStreamEvent::TextDelta(pass_text);
                                if let Some(encoded) =
                                    self.transcoder.encode_client_event_bytes(&ev)
                                {
                                    output.push(encoded);
                                }
                            }
                        }
                        DetectorAction::Buffer => {
                            // Text is buffered in the detector; nothing to send.
                        }
                        DetectorAction::TriggerFound { text_before } => {
                            // Send any text before the trigger to the client.
                            if !text_before.is_empty() {
                                let ev = CanonicalStreamEvent::TextDelta(text_before);
                                if let Some(encoded) =
                                    self.transcoder.encode_client_event_bytes(&ev)
                                {
                                    output.push(encoded);
                                }
                            }
                            // The rest is buffered for XML parsing — nothing more to send.
                        }
                        DetectorAction::BufferOverflow(overflow_text) => {
                            // Buffer exceeded limit — flush everything as text and
                            // disable FC for the rest of this response.
                            if !overflow_text.is_empty() {
                                let ev = CanonicalStreamEvent::TextDelta(overflow_text);
                                if let Some(encoded) =
                                    self.transcoder.encode_client_event_bytes(&ev)
                                {
                                    output.push(encoded);
                                }
                            }
                            self.fc_enabled = false;
                            self.synthesize_termination = false;
                            self.pending_stop_reason = None;
                        }
                    }
                }
                CanonicalStreamEvent::MessageEnd { stop_reason }
                    if self.fc_enabled && self.synthesize_termination =>
                {
                    // Suppress upstream stop while FC is active. We'll emit one
                    // synthesized terminal event in finalize().
                    self.pending_stop_reason = Some(stop_reason);
                }
                CanonicalStreamEvent::Done if self.fc_enabled && self.synthesize_termination => {
                    // Suppress upstream done while FC is active. finalize() emits done once.
                }
                CanonicalStreamEvent::TextDelta(_) => {
                    // FC not enabled — forward as-is.
                    if let Some(encoded) = self.transcoder.encode_client_event_bytes(&event) {
                        output.push(encoded);
                    }
                }
                // All other events (MessageStart, ToolCallStart, Usage,
                // MessageEnd, Done, Error, etc.) pass through directly.
                _ => {
                    if let Some(encoded) = self.transcoder.encode_client_event_bytes(&event) {
                        output.push(encoded);
                    }
                }
            }
        }
    }

    /// Process a single upstream SSE frame and return SSE strings for the client.
    #[must_use]
    pub fn process_frame(&mut self, frame: &SseEvent) -> Vec<String> {
        let mut output: Vec<String> = Vec::with_capacity(8);
        self.process_frame_into(frame, &mut output);
        output
    }

    /// Finalize the stream when the upstream ends.
    ///
    /// If the detector is in `ToolParsing` state, attempt to parse the
    /// buffered XML into tool calls. On success, emit synthetic tool-call
    /// events. On failure (D5 fallback), flush the buffer as text and
    /// close with `stop` finish reason.
    ///
    /// If the detector is still in Detecting state, flush any remaining
    /// partial buffer as text.
    ///
    /// Always emits a Done event at the end.
    pub fn finalize_into(&mut self, output: &mut Vec<String>) {
        output.clear();

        // FC detector may have been disabled mid-stream (e.g., overflow fallback).
        // In that mode, upstream terminal events are already forwarded verbatim.
        if !self.synthesize_termination {
            if let Some(remaining) = self.detector.finalize() {
                if !remaining.is_empty() {
                    let ev = CanonicalStreamEvent::TextDelta(remaining);
                    if let Some(encoded) = self.transcoder.encode_client_event(&ev) {
                        output.push(encoded);
                    }
                }
            }
            return;
        }

        match self.detector.state().clone() {
            DetectorState::ToolParsing | DetectorState::Completed => {
                // Get remaining buffer from the detector.
                let remaining = self.detector.finalize().unwrap_or_default();

                // Parse only buffered text from trigger onward.
                match parse_function_calls(&remaining, self.detector.trigger_signal()) {
                    Ok(parsed_calls) if !parsed_calls.is_empty() => {
                        self.emit_parsed_tool_calls_into(parsed_calls, output);
                    }
                    _ => {
                        // D5 fallback: parse failed — flush buffer as text.
                        if !remaining.is_empty() {
                            let ev = CanonicalStreamEvent::TextDelta(remaining);
                            if let Some(encoded) = self.transcoder.encode_client_event(&ev) {
                                output.push(encoded);
                            }
                        }
                        // End with stop reason.
                        let end_ev = CanonicalStreamEvent::MessageEnd {
                            stop_reason: self
                                .pending_stop_reason
                                .unwrap_or(CanonicalStopReason::EndOfTurn),
                        };
                        if let Some(encoded) = self.transcoder.encode_client_event(&end_ev) {
                            output.push(encoded);
                        }
                    }
                }
            }
            DetectorState::Detecting => {
                // Flush any remaining partial buffer from the detector.
                if let Some(remaining) = self.detector.finalize() {
                    if !remaining.is_empty() {
                        let ev = CanonicalStreamEvent::TextDelta(remaining);
                        if let Some(encoded) = self.transcoder.encode_client_event(&ev) {
                            output.push(encoded);
                        }
                    }
                }
                // End with EndOfTurn.
                let end_ev = CanonicalStreamEvent::MessageEnd {
                    stop_reason: self
                        .pending_stop_reason
                        .unwrap_or(CanonicalStopReason::EndOfTurn),
                };
                if let Some(encoded) = self.transcoder.encode_client_event(&end_ev) {
                    output.push(encoded);
                }
            }
        }

        // Always emit Done at the end.
        let done_ev = CanonicalStreamEvent::Done;
        if let Some(encoded) = self.transcoder.encode_client_event(&done_ev) {
            output.push(encoded);
        }
    }

    /// Finalize the stream when the upstream ends and append SSE bytes.
    pub fn finalize_into_bytes(&mut self, output: &mut Vec<bytes::Bytes>) {
        output.clear();

        // FC detector may have been disabled mid-stream (e.g., overflow fallback).
        // In that mode, upstream terminal events are already forwarded verbatim.
        if !self.synthesize_termination {
            if let Some(remaining) = self.detector.finalize() {
                if !remaining.is_empty() {
                    let ev = CanonicalStreamEvent::TextDelta(remaining);
                    if let Some(encoded) = self.transcoder.encode_client_event_bytes(&ev) {
                        output.push(encoded);
                    }
                }
            }
            return;
        }

        match self.detector.state().clone() {
            DetectorState::ToolParsing | DetectorState::Completed => {
                // Get remaining buffer from the detector.
                let remaining = self.detector.finalize().unwrap_or_default();

                // Parse only buffered text from trigger onward.
                match parse_function_calls(&remaining, self.detector.trigger_signal()) {
                    Ok(parsed_calls) if !parsed_calls.is_empty() => {
                        self.emit_parsed_tool_calls_into_bytes(parsed_calls, output);
                    }
                    _ => {
                        // D5 fallback: parse failed — flush buffer as text.
                        if !remaining.is_empty() {
                            let ev = CanonicalStreamEvent::TextDelta(remaining);
                            if let Some(encoded) = self.transcoder.encode_client_event_bytes(&ev) {
                                output.push(encoded);
                            }
                        }
                        // End with stop reason.
                        let end_ev = CanonicalStreamEvent::MessageEnd {
                            stop_reason: self
                                .pending_stop_reason
                                .unwrap_or(CanonicalStopReason::EndOfTurn),
                        };
                        if let Some(encoded) = self.transcoder.encode_client_event_bytes(&end_ev) {
                            output.push(encoded);
                        }
                    }
                }
            }
            DetectorState::Detecting => {
                // Flush any remaining partial buffer from the detector.
                if let Some(remaining) = self.detector.finalize() {
                    if !remaining.is_empty() {
                        let ev = CanonicalStreamEvent::TextDelta(remaining);
                        if let Some(encoded) = self.transcoder.encode_client_event_bytes(&ev) {
                            output.push(encoded);
                        }
                    }
                }
                // End with EndOfTurn.
                let end_ev = CanonicalStreamEvent::MessageEnd {
                    stop_reason: self
                        .pending_stop_reason
                        .unwrap_or(CanonicalStopReason::EndOfTurn),
                };
                if let Some(encoded) = self.transcoder.encode_client_event_bytes(&end_ev) {
                    output.push(encoded);
                }
            }
        }

        // Always emit Done at the end.
        let done_ev = CanonicalStreamEvent::Done;
        if let Some(encoded) = self.transcoder.encode_client_event_bytes(&done_ev) {
            output.push(encoded);
        }
    }

    /// Finalize the stream when the upstream ends and return SSE chunks.
    #[must_use]
    pub fn finalize(&mut self) -> Vec<String> {
        let mut output: Vec<String> = Vec::new();
        self.finalize_into(&mut output);
        output
    }

    /// Finalize the stream when the upstream ends and return SSE bytes.
    #[must_use]
    pub fn finalize_bytes(&mut self) -> Vec<bytes::Bytes> {
        let mut output: Vec<bytes::Bytes> = Vec::new();
        self.finalize_into_bytes(&mut output);
        output
    }

    /// Emit `ToolCallStart`, `ToolCallArgsDelta`, `ToolCallEnd` events for each
    /// parsed tool call, followed by `MessageEnd` with `ToolCalls` stop reason.
    fn emit_parsed_tool_calls_into(
        &mut self,
        parsed_calls: Vec<ParsedToolCall>,
        output: &mut Vec<String>,
    ) {
        for ParsedToolCall {
            id,
            name,
            arguments,
            arguments_json,
        } in parsed_calls
        {
            let index = self.tool_call_index;
            let id = id.map_or_else(next_call_id, String::from);

            let start_ev = CanonicalStreamEvent::ToolCallStart { index, id, name };
            if let Some(encoded) = self.transcoder.encode_client_event(&start_ev) {
                output.push(encoded);
            }

            let args_delta_ev = CanonicalStreamEvent::ToolCallArgsDelta {
                index,
                delta: parsed_call_arguments_delta(&arguments, arguments_json),
            };
            if let Some(encoded) = self.transcoder.encode_client_event(&args_delta_ev) {
                output.push(encoded);
            }

            let end_ev = CanonicalStreamEvent::ToolCallEnd {
                index,
                call_id: None,
                call_name: None,
            };
            if let Some(encoded) = self.transcoder.encode_client_event(&end_ev) {
                output.push(encoded);
            }

            self.tool_call_index += 1;
        }

        let msg_end = CanonicalStreamEvent::MessageEnd {
            stop_reason: CanonicalStopReason::ToolCalls,
        };
        if let Some(encoded) = self.transcoder.encode_client_event(&msg_end) {
            output.push(encoded);
        }
    }

    /// Emit `ToolCallStart`, `ToolCallArgsDelta`, `ToolCallEnd` events for each
    /// tool call, followed by `MessageEnd` with `ToolCalls` stop reason.
    fn emit_parsed_tool_calls_into_bytes(
        &mut self,
        parsed_calls: Vec<ParsedToolCall>,
        output: &mut Vec<bytes::Bytes>,
    ) {
        for ParsedToolCall {
            id,
            name,
            arguments,
            arguments_json,
        } in parsed_calls
        {
            let index = self.tool_call_index;
            let id = id.map_or_else(next_call_id, String::from);

            let start_ev = CanonicalStreamEvent::ToolCallStart { index, id, name };
            if let Some(encoded) = self.transcoder.encode_client_event_bytes(&start_ev) {
                output.push(encoded);
            }

            let args_delta_ev = CanonicalStreamEvent::ToolCallArgsDelta {
                index,
                delta: parsed_call_arguments_delta(&arguments, arguments_json),
            };
            if let Some(encoded) = self.transcoder.encode_client_event_bytes(&args_delta_ev) {
                output.push(encoded);
            }

            let end_ev = CanonicalStreamEvent::ToolCallEnd {
                index,
                call_id: None,
                call_name: None,
            };
            if let Some(encoded) = self.transcoder.encode_client_event_bytes(&end_ev) {
                output.push(encoded);
            }

            self.tool_call_index += 1;
        }

        let msg_end = CanonicalStreamEvent::MessageEnd {
            stop_reason: CanonicalStopReason::ToolCalls,
        };
        if let Some(encoded) = self.transcoder.encode_client_event_bytes(&msg_end) {
            output.push(encoded);
        }
    }
}

#[inline]
fn parsed_call_arguments_delta(
    arguments: &serde_json::Value,
    arguments_json: Option<Box<str>>,
) -> String {
    arguments_json.map_or_else(
        || serde_json::to_string(arguments).unwrap_or_else(|_| String::from("{}")),
        Into::into,
    )
}

#[cfg(test)]
mod tests {
    use super::{encode_sse_event, parse_sse_frame, parsed_call_arguments_delta};
    use crate::protocol::canonical::CanonicalStreamEvent;
    use serde_json::{json, Value};

    #[test]
    fn parse_sse_frame_data_only() {
        let raw = "data: hello\n\n";
        let frame = parse_sse_frame(raw).expect("expected frame");
        assert_eq!(frame.data, "hello");
        assert!(frame.event.is_none());
        assert!(frame.id.is_none());
        assert!(frame.retry.is_none());
    }

    #[test]
    fn parse_sse_frame_named_with_id_and_retry() {
        let raw = "event: message_start\nid: abc-1\nretry: 250\ndata: {\"x\":1}\n\n";
        let frame = parse_sse_frame(raw).expect("expected frame");
        assert_eq!(frame.event.as_deref(), Some("message_start"));
        assert_eq!(frame.id.as_deref(), Some("abc-1"));
        assert_eq!(frame.retry, Some(250));
        assert_eq!(frame.data, "{\"x\":1}");
    }

    #[test]
    fn parse_sse_frame_multiline_data() {
        let raw = "data: line1\ndata: line2\n\n";
        let frame = parse_sse_frame(raw).expect("expected frame");
        assert_eq!(frame.data, "line1\nline2");
    }

    #[test]
    fn parse_sse_frame_requires_dispatch_boundary() {
        let raw = "data: hello";
        assert!(parse_sse_frame(raw).is_none());
    }

    #[test]
    fn encode_sse_event_text_delta_is_valid_json_sse() {
        let encoded = encode_sse_event(&CanonicalStreamEvent::TextDelta("hello".to_string()))
            .expect("encode text delta");
        assert!(encoded.starts_with("data: "));
        assert!(encoded.ends_with("\n\n"));
        let payload = encoded.trim_start_matches("data: ").trim();
        let json: Value = serde_json::from_str(payload).expect("decode json");
        assert_eq!(json["delta"]["text"], "hello");
    }

    #[test]
    fn encode_sse_event_error_is_valid_json_sse() {
        let encoded = encode_sse_event(&CanonicalStreamEvent::Error {
            status: 503,
            message: "overloaded".to_string(),
        })
        .expect("encode error");
        let payload = encoded.trim_start_matches("data: ").trim();
        let json: Value = serde_json::from_str(payload).expect("decode json");
        assert_eq!(json["error"]["status"], 503);
        assert_eq!(json["error"]["message"], "overloaded");
    }

    #[test]
    fn parsed_call_arguments_delta_prefers_raw_json() {
        let args = json!({ "x": 1 });
        let delta = parsed_call_arguments_delta(&args, Some(Box::<str>::from("{\"x\": 1}")));
        assert_eq!(delta, "{\"x\": 1}");
    }

    #[test]
    fn parsed_call_arguments_delta_falls_back_to_serialization() {
        let args = json!({ "x": 1 });
        let delta = parsed_call_arguments_delta(&args, None);
        assert_eq!(delta, "{\"x\":1}");
    }
}
