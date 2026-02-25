use memchr::{memchr, memchr2, memmem};
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::sync::LazyLock;

use crate::json_scan::{parse_json_string_end, parse_json_value_end, skip_ws};
use crate::protocol::anthropic::stream::{
    decode_anthropic_stream_event_owned_into, encode_canonical_event_to_anthropic_sse_frame,
    parse_anthropic_sse_bytes, StatefulAnthropicStreamDecoder,
};
use crate::protocol::anthropic::AnthropicStreamEvent;
use crate::protocol::canonical::{
    CanonicalRole, CanonicalStopReason, CanonicalStreamEvent, CanonicalUsage, IngressApi,
    ProviderKind,
};
use crate::protocol::gemini::stream::{
    decode_gemini_stream_chunk_owned_into, encode_canonical_event_to_gemini_sse_with_bindings,
};
use crate::protocol::gemini::GeminiResponse;
use crate::protocol::mapping::{
    anthropic_stop_to_canonical, gemini_stop_to_canonical, openai_stop_to_canonical,
};
use crate::protocol::openai_chat::stream::{
    decode_openai_stream_chunk_into, encode_canonical_event_to_openai_sse_with_created,
};
use crate::protocol::openai_chat::OpenAiStreamChunk;
use crate::protocol::openai_responses::stream::{
    decode_responses_stream_event_owned_into,
    encode_canonical_event_to_responses_sse_frame_with_state,
};
use crate::protocol::openai_responses::ResponsesStreamEvent;
use crate::stream::SseEvent;
use crate::util::next_call_id;

/// Converts upstream provider stream events into the client's expected format.
///
/// The transcoder operates in two phases per SSE frame:
/// 1. Decode: upstream SSE frame -> Vec<CanonicalStreamEvent>
/// 2. Encode: `CanonicalStreamEvent` -> client SSE string
///
/// When upstream and client speak the same protocol, `is_passthrough()` returns true
/// and the caller can forward raw bytes without decode/re-encode.
pub struct StreamTranscoder {
    upstream_provider: ProviderKind,
    client_api: IngressApi,
    model: String,
    response_id: String,
    openai_created_unix_secs: u64,
    anthropic_decoder: Option<StatefulAnthropicStreamDecoder>,
    gemini_call_name_bindings: Option<FxHashMap<String, String>>,
    responses_tool_result_seq: Option<FxHashMap<String, usize>>,
    anthropic_done_sse: Option<String>,
    responses_done_sse: Option<String>,
    decode_buffer: Vec<CanonicalStreamEvent>,
    openai_message_started: bool,
    emit_usage: bool,
}

impl StreamTranscoder {
    #[must_use]
    pub fn new(
        upstream_provider: ProviderKind,
        client_api: IngressApi,
        model: String,
        response_id: String,
    ) -> Self {
        let openai_created_unix_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let anthropic_decoder = if upstream_provider == ProviderKind::Anthropic {
            Some(StatefulAnthropicStreamDecoder::new())
        } else {
            None
        };
        let gemini_call_name_bindings = if client_api == IngressApi::Gemini {
            Some(FxHashMap::default())
        } else {
            None
        };
        let responses_tool_result_seq = if client_api == IngressApi::OpenAiResponses {
            Some(FxHashMap::default())
        } else {
            None
        };
        let responses_done_sse = if client_api == IngressApi::OpenAiResponses {
            let mut scratch_seq = FxHashMap::default();
            let mut scratch_frame = String::new();
            let encoded = encode_canonical_event_to_responses_sse_frame_with_state(
                &CanonicalStreamEvent::Done,
                &model,
                &response_id,
                &mut scratch_seq,
                &mut scratch_frame,
            );
            if encoded {
                Some(scratch_frame)
            } else {
                None
            }
        } else {
            None
        };
        let anthropic_done_sse = if client_api == IngressApi::Anthropic {
            let mut scratch_frame = String::new();
            let encoded = encode_canonical_event_to_anthropic_sse_frame(
                &CanonicalStreamEvent::Done,
                &model,
                &response_id,
                &mut scratch_frame,
            );
            if encoded {
                Some(scratch_frame)
            } else {
                None
            }
        } else {
            None
        };

        Self {
            upstream_provider,
            client_api,
            model,
            response_id,
            openai_created_unix_secs,
            anthropic_decoder,
            gemini_call_name_bindings,
            responses_tool_result_seq,
            anthropic_done_sse,
            responses_done_sse,
            decode_buffer: Vec::with_capacity(8),
            openai_message_started: false,
            emit_usage: emits_usage_event(client_api),
        }
    }

    /// Decode an upstream SSE frame into canonical stream events.
    ///
    /// Dispatches based on the upstream provider kind to the appropriate
    /// protocol-specific decoder.
    pub fn decode_upstream_frame(&mut self, frame: &SseEvent) -> Vec<CanonicalStreamEvent> {
        let mut out = Vec::with_capacity(8);
        self.decode_upstream_frame_into(frame, &mut out);
        out
    }

    /// Decode an upstream SSE frame into a caller-provided canonical event buffer.
    pub fn decode_upstream_frame_into(
        &mut self,
        frame: &SseEvent,
        out: &mut Vec<CanonicalStreamEvent>,
    ) {
        out.clear();
        self.decode_upstream_event_data_into(frame.event.as_deref(), frame.data.as_bytes(), out);
    }

    /// Decode one complete raw SSE frame into canonical events.
    ///
    /// Returns `true` when the frame was parsed as SSE and decoded (including
    /// frames that decode to zero canonical events), `false` when the frame
    /// could not be parsed as SSE.
    pub fn try_decode_upstream_raw_frame_into(
        &mut self,
        raw_frame: &[u8],
        out: &mut Vec<CanonicalStreamEvent>,
    ) -> bool {
        out.clear();
        if matches!(
            self.upstream_provider,
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi
        ) {
            if let Some(data) = parse_raw_sse_data_only_frame(raw_frame) {
                self.decode_upstream_event_data_into(None, data, out);
                return true;
            }
            let Some(frame) = crate::stream::parse_sse_frame_bytes(raw_frame) else {
                return false;
            };
            self.decode_upstream_event_data_into(None, frame.data.as_bytes(), out);
            return true;
        }

        if let Some(data) = parse_raw_sse_data_only_frame(raw_frame) {
            self.decode_upstream_event_data_into(None, data, out);
            return true;
        }
        if let Some((event_type, data)) = parse_raw_sse_event_and_data_frame(raw_frame) {
            self.decode_upstream_event_data_into(Some(event_type), data, out);
            return true;
        }
        let Some(frame) = crate::stream::parse_sse_frame_bytes(raw_frame) else {
            return false;
        };
        self.decode_upstream_event_data_into(frame.event.as_deref(), frame.data.as_bytes(), out);
        true
    }

    /// Decode one complete raw SSE frame into canonical events.
    pub fn decode_upstream_raw_frame_into(
        &mut self,
        raw_frame: &[u8],
        out: &mut Vec<CanonicalStreamEvent>,
    ) {
        let _ = self.try_decode_upstream_raw_frame_into(raw_frame, out);
    }

    fn decode_upstream_event_data_into(
        &mut self,
        event_type: Option<&str>,
        data: &[u8],
        out: &mut Vec<CanonicalStreamEvent>,
    ) {
        let emit_usage = self.emit_usage;
        match self.upstream_provider {
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => {
                let _ = event_type;
                self.decode_openai_data_frame_bytes_into(data, out, emit_usage);
            }
            ProviderKind::Anthropic => {
                let event_type = event_type.unwrap_or("");
                if try_fast_decode_anthropic_stream_event(
                    event_type,
                    data,
                    self.anthropic_decoder.as_mut(),
                    out,
                    emit_usage,
                ) {
                    return;
                }
                if let Some(event) = parse_anthropic_sse_bytes(event_type, data) {
                    if let Some(decoder) = self.anthropic_decoder.as_mut() {
                        decoder.decode_owned_into(event, out);
                    } else {
                        decode_anthropic_stream_event_owned_into(event, out);
                    }
                }
            }
            ProviderKind::Gemini => {
                if data == b"[DONE]" {
                    out.push(CanonicalStreamEvent::Done);
                    return;
                }
                if try_fast_decode_gemini_stream_chunk(data, out, emit_usage) {
                    return;
                }
                if let Ok(chunk) = serde_json::from_slice::<GeminiResponse>(data) {
                    decode_gemini_stream_chunk_owned_into(chunk, out);
                }
            }
            ProviderKind::OpenAiResponses => {
                if data == b"[DONE]" {
                    out.push(CanonicalStreamEvent::Done);
                    return;
                }
                if try_fast_decode_responses_stream_event(event_type, data, out, emit_usage) {
                    return;
                }
                if let Ok(event) = serde_json::from_slice::<ResponsesStreamEvent>(data) {
                    decode_responses_stream_event_owned_into(event, out);
                }
            }
        }
    }

    /// Decode an OpenAI-compatible SSE `data` payload into canonical events.
    ///
    /// This is a fast-path helper for callers that already split SSE frames and
    /// only need the `data:` content.
    pub fn decode_openai_data_payload_into(
        &mut self,
        data: &str,
        out: &mut Vec<CanonicalStreamEvent>,
    ) {
        let _ = self.try_decode_openai_data_payload_bytes_into(data.as_bytes(), out);
    }

    /// Decode an OpenAI-compatible SSE `data` payload from bytes into canonical events.
    ///
    /// Returns `true` when payload decoding was handled by either fast-path
    /// parsing or the serde fallback, `false` when the payload was not decodable.
    pub fn try_decode_openai_data_payload_bytes_into(
        &mut self,
        data: &[u8],
        out: &mut Vec<CanonicalStreamEvent>,
    ) -> bool {
        out.clear();
        self.decode_openai_data_frame_bytes_into(data, out, self.emit_usage)
    }

    /// Decode an OpenAI-compatible SSE data payload bytes into canonical events.
    fn decode_openai_data_frame_bytes_into(
        &mut self,
        data: &[u8],
        out: &mut Vec<CanonicalStreamEvent>,
        emit_usage: bool,
    ) -> bool {
        if !matches!(
            self.upstream_provider,
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi
        ) {
            return false;
        }

        if data == b"[DONE]" {
            out.push(CanonicalStreamEvent::Done);
            return true;
        }

        if try_fast_decode_openai_stream_chunk(
            data,
            out,
            &mut self.openai_message_started,
            emit_usage,
        ) {
            return true;
        }

        if let Ok(chunk) = serde_json::from_slice::<OpenAiStreamChunk>(data) {
            let saw_message_start = !self.openai_message_started
                && chunk.choices.iter().any(|c| c.delta.role.is_some());
            decode_openai_stream_chunk_into(chunk, out);
            if saw_message_start {
                self.openai_message_started = true;
            }
            return true;
        }
        false
    }

    /// Encode a canonical stream event into the client's SSE format.
    ///
    /// Returns `None` for events that have no representation in the target protocol.
    pub fn encode_client_event(&mut self, event: &CanonicalStreamEvent) -> Option<String> {
        match self.client_api {
            IngressApi::OpenAiChat => encode_canonical_event_to_openai_sse_with_created(
                event,
                &self.model,
                &self.response_id,
                self.openai_created_unix_secs,
            ),
            IngressApi::Anthropic => {
                if matches!(event, CanonicalStreamEvent::Usage(_)) {
                    return None;
                }
                if let CanonicalStreamEvent::Done = event {
                    return self.anthropic_done_sse.clone();
                }
                let mut frame = String::with_capacity(estimated_anthropic_frame_capacity(
                    event,
                    self.model.len(),
                    self.response_id.len(),
                ));
                if encode_canonical_event_to_anthropic_sse_frame(
                    event,
                    &self.model,
                    &self.response_id,
                    &mut frame,
                ) {
                    Some(frame)
                } else {
                    None
                }
            }
            IngressApi::Gemini => {
                let bindings = self.gemini_call_name_bindings.as_mut()?;
                encode_canonical_event_to_gemini_sse_with_bindings(event, bindings)
            }
            IngressApi::OpenAiResponses => {
                if matches!(
                    event,
                    CanonicalStreamEvent::Usage(_)
                        | CanonicalStreamEvent::MessageEnd { .. }
                        | CanonicalStreamEvent::ReasoningDelta(_)
                ) {
                    return None;
                }
                if let CanonicalStreamEvent::Done = event {
                    return self.responses_done_sse.clone();
                }
                let seq = self.responses_tool_result_seq.as_mut()?;
                let mut frame = String::with_capacity(estimated_responses_frame_capacity(
                    event,
                    self.model.len(),
                    self.response_id.len(),
                ));
                if encode_canonical_event_to_responses_sse_frame_with_state(
                    event,
                    &self.model,
                    &self.response_id,
                    seq,
                    &mut frame,
                ) {
                    Some(frame)
                } else {
                    None
                }
            }
        }
    }

    /// Encode a canonical stream event into the client's SSE format and return
    /// bytes ready for HTTP body streaming.
    #[inline]
    pub fn encode_client_event_bytes(
        &mut self,
        event: &CanonicalStreamEvent,
    ) -> Option<bytes::Bytes> {
        self.encode_client_event(event).map(bytes::Bytes::from)
    }

    /// Returns true when upstream and client use the same protocol
    /// and raw SSE bytes can be forwarded without decode/re-encode.
    #[must_use]
    pub fn is_passthrough(&self) -> bool {
        matches!(
            (self.upstream_provider, self.client_api),
            (
                ProviderKind::OpenAi | ProviderKind::GeminiOpenAi,
                IngressApi::OpenAiChat
            ) | (ProviderKind::Anthropic, IngressApi::Anthropic)
                | (ProviderKind::Gemini, IngressApi::Gemini)
                | (ProviderKind::OpenAiResponses, IngressApi::OpenAiResponses)
        )
    }

    /// Process one SSE frame end-to-end: decode from upstream, encode for client.
    ///
    /// Appends SSE-formatted strings ready to send to the client into `out`.
    /// Event ordering is preserved; no reordering occurs.
    pub fn transcode_frame_into(&mut self, frame: &SseEvent, out: &mut Vec<String>) {
        let mut decode_buffer = std::mem::take(&mut self.decode_buffer);
        self.transcode_frame_into_with_decode_buffer(frame, &mut decode_buffer, out);
        self.decode_buffer = decode_buffer;
    }

    /// Process one SSE frame end-to-end using a caller-provided decode buffer.
    ///
    /// This avoids per-frame scratch-buffer ownership swaps in hot streaming loops.
    pub fn transcode_frame_into_with_decode_buffer(
        &mut self,
        frame: &SseEvent,
        decode_buffer: &mut Vec<CanonicalStreamEvent>,
        out: &mut Vec<String>,
    ) {
        out.clear();
        self.decode_upstream_frame_into(frame, decode_buffer);
        if decode_buffer.len() > out.capacity() {
            out.reserve(decode_buffer.len() - out.capacity());
        }
        match self.client_api {
            IngressApi::OpenAiChat => {
                for event in decode_buffer.iter() {
                    if let Some(encoded) = encode_canonical_event_to_openai_sse_with_created(
                        event,
                        &self.model,
                        &self.response_id,
                        self.openai_created_unix_secs,
                    ) {
                        out.push(encoded);
                    }
                }
            }
            IngressApi::Anthropic => {
                let model_len = self.model.len();
                let response_id_len = self.response_id.len();
                let done_sse = self.anthropic_done_sse.as_deref();
                for event in decode_buffer.iter() {
                    if matches!(event, CanonicalStreamEvent::Usage(_)) {
                        continue;
                    }
                    if let CanonicalStreamEvent::Done = event {
                        if let Some(done) = done_sse {
                            out.push(done.to_owned());
                        }
                        continue;
                    }

                    let mut frame = String::with_capacity(estimated_anthropic_frame_capacity(
                        event,
                        model_len,
                        response_id_len,
                    ));
                    if encode_canonical_event_to_anthropic_sse_frame(
                        event,
                        &self.model,
                        &self.response_id,
                        &mut frame,
                    ) {
                        out.push(frame);
                    }
                }
            }
            IngressApi::Gemini => {
                let Some(bindings) = self.gemini_call_name_bindings.as_mut() else {
                    return;
                };
                for event in decode_buffer.iter() {
                    if let Some(encoded) =
                        encode_canonical_event_to_gemini_sse_with_bindings(event, bindings)
                    {
                        out.push(encoded);
                    }
                }
            }
            IngressApi::OpenAiResponses => {
                let model_len = self.model.len();
                let response_id_len = self.response_id.len();
                let done_sse = self.responses_done_sse.as_deref();
                let Some(seq) = self.responses_tool_result_seq.as_mut() else {
                    return;
                };
                for event in decode_buffer.iter() {
                    if matches!(
                        event,
                        CanonicalStreamEvent::Usage(_)
                            | CanonicalStreamEvent::MessageEnd { .. }
                            | CanonicalStreamEvent::ReasoningDelta(_)
                    ) {
                        continue;
                    }
                    if let CanonicalStreamEvent::Done = event {
                        if let Some(done) = done_sse {
                            out.push(done.to_owned());
                        }
                        continue;
                    }

                    let mut frame = String::with_capacity(estimated_responses_frame_capacity(
                        event,
                        model_len,
                        response_id_len,
                    ));
                    if encode_canonical_event_to_responses_sse_frame_with_state(
                        event,
                        &self.model,
                        &self.response_id,
                        seq,
                        &mut frame,
                    ) {
                        out.push(frame);
                    }
                }
            }
        }
    }

    /// Process one SSE frame end-to-end using a caller-provided decode buffer
    /// and emit byte chunks directly.
    pub fn transcode_frame_into_bytes_with_decode_buffer(
        &mut self,
        frame: &SseEvent,
        decode_buffer: &mut Vec<CanonicalStreamEvent>,
        out: &mut Vec<bytes::Bytes>,
    ) {
        out.clear();
        self.decode_upstream_frame_into(frame, decode_buffer);
        if decode_buffer.len() > out.capacity() {
            out.reserve(decode_buffer.len() - out.capacity());
        }
        match self.client_api {
            IngressApi::OpenAiChat => {
                for event in decode_buffer.iter() {
                    if let Some(encoded) = encode_canonical_event_to_openai_sse_with_created(
                        event,
                        &self.model,
                        &self.response_id,
                        self.openai_created_unix_secs,
                    ) {
                        out.push(bytes::Bytes::from(encoded));
                    }
                }
            }
            IngressApi::Anthropic => {
                let model_len = self.model.len();
                let response_id_len = self.response_id.len();
                let done_sse = self.anthropic_done_sse.as_deref();
                for event in decode_buffer.iter() {
                    if matches!(event, CanonicalStreamEvent::Usage(_)) {
                        continue;
                    }
                    if let CanonicalStreamEvent::Done = event {
                        if let Some(done) = done_sse {
                            out.push(bytes::Bytes::copy_from_slice(done.as_bytes()));
                        }
                        continue;
                    }

                    let mut frame = String::with_capacity(estimated_anthropic_frame_capacity(
                        event,
                        model_len,
                        response_id_len,
                    ));
                    if encode_canonical_event_to_anthropic_sse_frame(
                        event,
                        &self.model,
                        &self.response_id,
                        &mut frame,
                    ) {
                        out.push(bytes::Bytes::from(frame));
                    }
                }
            }
            IngressApi::Gemini => {
                let Some(bindings) = self.gemini_call_name_bindings.as_mut() else {
                    return;
                };
                for event in decode_buffer.iter() {
                    if let Some(encoded) =
                        encode_canonical_event_to_gemini_sse_with_bindings(event, bindings)
                    {
                        out.push(bytes::Bytes::from(encoded));
                    }
                }
            }
            IngressApi::OpenAiResponses => {
                let model_len = self.model.len();
                let response_id_len = self.response_id.len();
                let done_sse = self.responses_done_sse.as_deref();
                let Some(seq) = self.responses_tool_result_seq.as_mut() else {
                    return;
                };
                for event in decode_buffer.iter() {
                    if matches!(
                        event,
                        CanonicalStreamEvent::Usage(_)
                            | CanonicalStreamEvent::MessageEnd { .. }
                            | CanonicalStreamEvent::ReasoningDelta(_)
                    ) {
                        continue;
                    }
                    if let CanonicalStreamEvent::Done = event {
                        if let Some(done) = done_sse {
                            out.push(bytes::Bytes::copy_from_slice(done.as_bytes()));
                        }
                        continue;
                    }

                    let mut frame = String::with_capacity(estimated_responses_frame_capacity(
                        event,
                        model_len,
                        response_id_len,
                    ));
                    if encode_canonical_event_to_responses_sse_frame_with_state(
                        event,
                        &self.model,
                        &self.response_id,
                        seq,
                        &mut frame,
                    ) {
                        out.push(bytes::Bytes::from(frame));
                    }
                }
            }
        }
    }

    /// Process one raw SSE frame end-to-end using a caller-provided decode buffer.
    ///
    /// This avoids constructing a temporary `SseEvent` on the hot path.
    pub fn transcode_raw_frame_into_with_decode_buffer(
        &mut self,
        raw_frame: &[u8],
        decode_buffer: &mut Vec<CanonicalStreamEvent>,
        out: &mut Vec<String>,
    ) -> bool {
        out.clear();
        let mut decoded = self.try_decode_upstream_raw_frame_into(raw_frame, decode_buffer);
        if !decoded && !raw_frame_terminated(raw_frame) {
            let mut terminated = Vec::with_capacity(raw_frame.len() + 2);
            terminated.extend_from_slice(raw_frame);
            terminated.extend_from_slice(b"\n\n");
            decoded = self.try_decode_upstream_raw_frame_into(&terminated, decode_buffer);
        }
        if !decoded {
            return false;
        }
        if decode_buffer.len() > out.capacity() {
            out.reserve(decode_buffer.len() - out.capacity());
        }
        match self.client_api {
            IngressApi::OpenAiChat => {
                for event in decode_buffer.iter() {
                    if let Some(encoded) = encode_canonical_event_to_openai_sse_with_created(
                        event,
                        &self.model,
                        &self.response_id,
                        self.openai_created_unix_secs,
                    ) {
                        out.push(encoded);
                    }
                }
            }
            IngressApi::Anthropic => {
                let model_len = self.model.len();
                let response_id_len = self.response_id.len();
                let done_sse = self.anthropic_done_sse.as_deref();
                for event in decode_buffer.iter() {
                    if matches!(event, CanonicalStreamEvent::Usage(_)) {
                        continue;
                    }
                    if let CanonicalStreamEvent::Done = event {
                        if let Some(done) = done_sse {
                            out.push(done.to_owned());
                        }
                        continue;
                    }

                    let mut frame = String::with_capacity(estimated_anthropic_frame_capacity(
                        event,
                        model_len,
                        response_id_len,
                    ));
                    if encode_canonical_event_to_anthropic_sse_frame(
                        event,
                        &self.model,
                        &self.response_id,
                        &mut frame,
                    ) {
                        out.push(frame);
                    }
                }
            }
            IngressApi::Gemini => {
                let Some(bindings) = self.gemini_call_name_bindings.as_mut() else {
                    return true;
                };
                for event in decode_buffer.iter() {
                    if let Some(encoded) =
                        encode_canonical_event_to_gemini_sse_with_bindings(event, bindings)
                    {
                        out.push(encoded);
                    }
                }
            }
            IngressApi::OpenAiResponses => {
                let model_len = self.model.len();
                let response_id_len = self.response_id.len();
                let done_sse = self.responses_done_sse.as_deref();
                let Some(seq) = self.responses_tool_result_seq.as_mut() else {
                    return true;
                };
                for event in decode_buffer.iter() {
                    if matches!(
                        event,
                        CanonicalStreamEvent::Usage(_)
                            | CanonicalStreamEvent::MessageEnd { .. }
                            | CanonicalStreamEvent::ReasoningDelta(_)
                    ) {
                        continue;
                    }
                    if let CanonicalStreamEvent::Done = event {
                        if let Some(done) = done_sse {
                            out.push(done.to_owned());
                        }
                        continue;
                    }

                    let mut frame = String::with_capacity(estimated_responses_frame_capacity(
                        event,
                        model_len,
                        response_id_len,
                    ));
                    if encode_canonical_event_to_responses_sse_frame_with_state(
                        event,
                        &self.model,
                        &self.response_id,
                        seq,
                        &mut frame,
                    ) {
                        out.push(frame);
                    }
                }
            }
        }
        true
    }

    /// Process one raw SSE frame end-to-end using a caller-provided decode
    /// buffer and emit byte chunks directly.
    pub fn transcode_raw_frame_into_bytes_with_decode_buffer(
        &mut self,
        raw_frame: &[u8],
        decode_buffer: &mut Vec<CanonicalStreamEvent>,
        out: &mut Vec<bytes::Bytes>,
    ) -> bool {
        out.clear();
        let mut decoded = self.try_decode_upstream_raw_frame_into(raw_frame, decode_buffer);
        if !decoded && !raw_frame_terminated(raw_frame) {
            let mut terminated = Vec::with_capacity(raw_frame.len() + 2);
            terminated.extend_from_slice(raw_frame);
            terminated.extend_from_slice(b"\n\n");
            decoded = self.try_decode_upstream_raw_frame_into(&terminated, decode_buffer);
        }
        if !decoded {
            return false;
        }
        if decode_buffer.len() > out.capacity() {
            out.reserve(decode_buffer.len() - out.capacity());
        }
        match self.client_api {
            IngressApi::OpenAiChat => {
                for event in decode_buffer.iter() {
                    if let Some(encoded) = encode_canonical_event_to_openai_sse_with_created(
                        event,
                        &self.model,
                        &self.response_id,
                        self.openai_created_unix_secs,
                    ) {
                        out.push(bytes::Bytes::from(encoded));
                    }
                }
            }
            IngressApi::Anthropic => {
                let model_len = self.model.len();
                let response_id_len = self.response_id.len();
                let done_sse = self.anthropic_done_sse.as_deref();
                for event in decode_buffer.iter() {
                    if matches!(event, CanonicalStreamEvent::Usage(_)) {
                        continue;
                    }
                    if let CanonicalStreamEvent::Done = event {
                        if let Some(done) = done_sse {
                            out.push(bytes::Bytes::copy_from_slice(done.as_bytes()));
                        }
                        continue;
                    }

                    let mut frame = String::with_capacity(estimated_anthropic_frame_capacity(
                        event,
                        model_len,
                        response_id_len,
                    ));
                    if encode_canonical_event_to_anthropic_sse_frame(
                        event,
                        &self.model,
                        &self.response_id,
                        &mut frame,
                    ) {
                        out.push(bytes::Bytes::from(frame));
                    }
                }
            }
            IngressApi::Gemini => {
                let Some(bindings) = self.gemini_call_name_bindings.as_mut() else {
                    return true;
                };
                for event in decode_buffer.iter() {
                    if let Some(encoded) =
                        encode_canonical_event_to_gemini_sse_with_bindings(event, bindings)
                    {
                        out.push(bytes::Bytes::from(encoded));
                    }
                }
            }
            IngressApi::OpenAiResponses => {
                let model_len = self.model.len();
                let response_id_len = self.response_id.len();
                let done_sse = self.responses_done_sse.as_deref();
                let Some(seq) = self.responses_tool_result_seq.as_mut() else {
                    return true;
                };
                for event in decode_buffer.iter() {
                    if matches!(
                        event,
                        CanonicalStreamEvent::Usage(_)
                            | CanonicalStreamEvent::MessageEnd { .. }
                            | CanonicalStreamEvent::ReasoningDelta(_)
                    ) {
                        continue;
                    }
                    if let CanonicalStreamEvent::Done = event {
                        if let Some(done) = done_sse {
                            out.push(bytes::Bytes::copy_from_slice(done.as_bytes()));
                        }
                        continue;
                    }

                    let mut frame = String::with_capacity(estimated_responses_frame_capacity(
                        event,
                        model_len,
                        response_id_len,
                    ));
                    if encode_canonical_event_to_responses_sse_frame_with_state(
                        event,
                        &self.model,
                        &self.response_id,
                        seq,
                        &mut frame,
                    ) {
                        out.push(bytes::Bytes::from(frame));
                    }
                }
            }
        }
        true
    }

    /// Process one SSE frame end-to-end and return SSE chunks.
    #[must_use]
    pub fn transcode_frame(&mut self, frame: &SseEvent) -> Vec<String> {
        let mut out = Vec::with_capacity(8);
        self.transcode_frame_into(frame, &mut out);
        out
    }
}

#[inline]
const fn emits_usage_event(client_api: IngressApi) -> bool {
    matches!(client_api, IngressApi::OpenAiChat | IngressApi::Gemini)
}

#[inline]
fn raw_sse_payload_end(raw: &[u8]) -> Option<usize> {
    if raw.ends_with(b"\r\n\r\n") {
        Some(raw.len().saturating_sub(4))
    } else if raw.ends_with(b"\n\n") {
        Some(raw.len().saturating_sub(2))
    } else {
        None
    }
}

#[inline]
fn parse_raw_sse_data_only_frame(raw: &[u8]) -> Option<&[u8]> {
    if !raw.starts_with(b"data:") {
        return None;
    }
    let end = raw_sse_payload_end(raw)?;
    if end < 5 {
        return None;
    }
    let start = 5 + usize::from(raw.get(5) == Some(&b' '));
    let data = raw.get(start..end)?;
    if memchr2(b'\n', b'\r', data).is_some() {
        return None;
    }
    Some(data)
}

#[inline]
fn parse_raw_sse_event_and_data_frame(raw: &[u8]) -> Option<(&str, &[u8])> {
    if !raw.starts_with(b"event:") {
        return None;
    }
    let end = raw_sse_payload_end(raw)?;
    let first_newline = memchr(b'\n', raw)?;
    if first_newline + 1 >= end {
        return None;
    }

    let mut event_line = &raw[..first_newline];
    if event_line.last().copied() == Some(b'\r') {
        event_line = &event_line[..event_line.len().saturating_sub(1)];
    }
    let event_value = event_line.strip_prefix(b"event:")?;
    let event_value = event_value.strip_prefix(b" ").unwrap_or(event_value);
    let event_type = std::str::from_utf8(event_value).ok()?;

    let mut data_line = raw.get(first_newline + 1..end)?;
    if memchr(b'\n', data_line).is_some() {
        return None;
    }
    if data_line.last().copied() == Some(b'\r') {
        data_line = &data_line[..data_line.len().saturating_sub(1)];
    }
    let data_value = data_line.strip_prefix(b"data:")?;
    let data_value = data_value.strip_prefix(b" ").unwrap_or(data_value);

    Some((event_type, data_value))
}

#[inline]
fn estimated_anthropic_frame_capacity(
    event: &CanonicalStreamEvent,
    model_len: usize,
    response_id_len: usize,
) -> usize {
    match event {
        CanonicalStreamEvent::MessageStart { .. } => 152 + model_len + response_id_len,
        CanonicalStreamEvent::TextDelta(text) => 96 + text.len(),
        CanonicalStreamEvent::ReasoningDelta(text) => 112 + text.len(),
        CanonicalStreamEvent::ToolCallStart {
            id: call_id, name, ..
        } => 136 + call_id.len() + name.len(),
        CanonicalStreamEvent::ToolCallArgsDelta { delta, .. } => 120 + delta.len(),
        CanonicalStreamEvent::ToolCallEnd { .. } => 72,
        CanonicalStreamEvent::MessageEnd { .. } => 120,
        CanonicalStreamEvent::Done => 48,
        CanonicalStreamEvent::ToolResult {
            tool_call_id,
            content,
        } => 136 + tool_call_id.len() + content.len(),
        CanonicalStreamEvent::Error { message, .. } => 64 + message.len(),
        CanonicalStreamEvent::Usage(_) => 0,
    }
}

#[inline]
fn estimated_responses_frame_capacity(
    event: &CanonicalStreamEvent,
    model_len: usize,
    response_id_len: usize,
) -> usize {
    match event {
        CanonicalStreamEvent::TextDelta(delta) => 96 + delta.len(),
        CanonicalStreamEvent::ToolCallStart { id, name, .. } => 152 + id.len() + name.len(),
        CanonicalStreamEvent::ToolCallArgsDelta { delta, .. } => 112 + delta.len(),
        CanonicalStreamEvent::ToolCallEnd {
            call_id, call_name, ..
        } => {
            144 + call_id.as_ref().map_or(0, String::len)
                + call_name.as_ref().map_or(0, String::len)
        }
        CanonicalStreamEvent::ToolResult {
            tool_call_id,
            content,
        } => 160 + tool_call_id.len() * 2 + content.len(),
        CanonicalStreamEvent::MessageStart { .. } | CanonicalStreamEvent::Done => {
            152 + model_len + response_id_len
        }
        CanonicalStreamEvent::Error { message, .. } => 56 + message.len(),
        CanonicalStreamEvent::Usage(_)
        | CanonicalStreamEvent::MessageEnd { .. }
        | CanonicalStreamEvent::ReasoningDelta(_) => 0,
    }
}

#[inline]
fn try_fast_decode_openai_stream_chunk(
    bytes: &[u8],
    out: &mut Vec<CanonicalStreamEvent>,
    message_started: &mut bool,
    emit_usage: bool,
) -> bool {
    const OPENAI_CONTENT_KEY_LEN: usize = OPENAI_CONTENT_KEY.len();
    const OPENAI_FINISH_REASON_KEY_LEN: usize = OPENAI_FINISH_REASON_KEY.len();
    const OPENAI_ROLE_KEY_LEN: usize = OPENAI_ROLE_KEY.len();
    const OPENAI_USAGE_KEY_LEN: usize = OPENAI_USAGE_KEY.len();

    let mut produced = false;
    let mut handled = false;
    let key_positions =
        find_openai_chunk_key_positions(bytes, !*message_started, emit_usage && !produced);

    if !*message_started {
        if let Some(role_key_pos) = key_positions.role {
            if let Some(canonical_role) =
                parse_openai_role_after_key_pos(bytes, role_key_pos, OPENAI_ROLE_KEY_LEN)
            {
                out.push(CanonicalStreamEvent::MessageStart {
                    role: canonical_role,
                });
                produced = true;
                *message_started = true;
            }
        }
    }

    if let Some(content_key_pos) = key_positions.content {
        let Some(content) =
            parse_string_after_key_pos_cow(bytes, content_key_pos, OPENAI_CONTENT_KEY_LEN)
        else {
            return false;
        };
        handled = true;
        if !content.is_empty() {
            out.push(CanonicalStreamEvent::TextDelta(content.into_owned()));
            produced = true;
        }
    } else if let Some(tool_calls_key_pos) = key_positions.tool_calls {
        let Some(tool_calls_produced) =
            try_fast_decode_openai_tool_calls_chunk_at(bytes, tool_calls_key_pos, out)
        else {
            return false;
        };
        handled = true;
        if tool_calls_produced {
            produced = true;
        }
    }

    if let Some(finish_reason_key_pos) = key_positions.finish_reason {
        handled = true;
        let value_start = skip_ws(bytes, finish_reason_key_pos + OPENAI_FINISH_REASON_KEY_LEN);
        if bytes.get(value_start) == Some(&b'n') {
            let value_end = value_start.saturating_add(4);
            if value_end > bytes.len() || &bytes[value_start..value_end] != b"null" {
                return false;
            }
        } else if let Some(stop_reason) =
            parse_openai_stop_reason_from_string_value(bytes, value_start)
        {
            out.push(CanonicalStreamEvent::MessageEnd { stop_reason });
            produced = true;
        } else {
            return false;
        }
    }

    if emit_usage && !produced {
        if let Some(usage_key_pos) = key_positions.usage {
            if let Some(usage) =
                parse_openai_usage_after_key_pos(bytes, usage_key_pos, OPENAI_USAGE_KEY_LEN)
            {
                out.push(CanonicalStreamEvent::Usage(usage));
                produced = true;
                handled = true;
            }
        }
    }

    produced || handled
}

const OPENAI_ROLE_KEY: &[u8] = br#""role":"#;
const OPENAI_CONTENT_KEY: &[u8] = br#""content":"#;
const OPENAI_TOOL_CALLS_KEY: &[u8] = br#""tool_calls""#;
const OPENAI_FINISH_REASON_KEY: &[u8] = br#""finish_reason":"#;
const OPENAI_USAGE_KEY: &[u8] = br#""usage""#;
const ANTHROPIC_TYPE_TOOL_USE: &[u8] = br#""type":"tool_use""#;
const ANTHROPIC_TYPE_TEXT: &[u8] = br#""type":"text""#;
const ANTHROPIC_TYPE_THINKING: &[u8] = br#""type":"thinking""#;
const ANTHROPIC_TYPE_TOOL_RESULT: &[u8] = br#""type":"tool_result""#;
const ANTHROPIC_TEXT_DELTA: &[u8] = br#""text_delta""#;
const ANTHROPIC_THINKING_DELTA: &[u8] = br#""thinking_delta""#;
const ANTHROPIC_INPUT_JSON_DELTA: &[u8] = br#""input_json_delta""#;
const ANTHROPIC_STOP_REASON_NULL: &[u8] = br#""stop_reason":null"#;

static ANTHROPIC_TYPE_TOOL_USE_FINDER: LazyLock<memmem::Finder<'static>> =
    LazyLock::new(|| memmem::Finder::new(ANTHROPIC_TYPE_TOOL_USE));
static ANTHROPIC_TYPE_TEXT_FINDER: LazyLock<memmem::Finder<'static>> =
    LazyLock::new(|| memmem::Finder::new(ANTHROPIC_TYPE_TEXT));
static ANTHROPIC_TYPE_THINKING_FINDER: LazyLock<memmem::Finder<'static>> =
    LazyLock::new(|| memmem::Finder::new(ANTHROPIC_TYPE_THINKING));
static ANTHROPIC_TYPE_TOOL_RESULT_FINDER: LazyLock<memmem::Finder<'static>> =
    LazyLock::new(|| memmem::Finder::new(ANTHROPIC_TYPE_TOOL_RESULT));
static ANTHROPIC_TEXT_DELTA_FINDER: LazyLock<memmem::Finder<'static>> =
    LazyLock::new(|| memmem::Finder::new(ANTHROPIC_TEXT_DELTA));
static ANTHROPIC_THINKING_DELTA_FINDER: LazyLock<memmem::Finder<'static>> =
    LazyLock::new(|| memmem::Finder::new(ANTHROPIC_THINKING_DELTA));
static ANTHROPIC_INPUT_JSON_DELTA_FINDER: LazyLock<memmem::Finder<'static>> =
    LazyLock::new(|| memmem::Finder::new(ANTHROPIC_INPUT_JSON_DELTA));
static ANTHROPIC_STOP_REASON_NULL_FINDER: LazyLock<memmem::Finder<'static>> =
    LazyLock::new(|| memmem::Finder::new(ANTHROPIC_STOP_REASON_NULL));

#[derive(Default)]
struct OpenAiChunkKeyPositions {
    role: Option<usize>,
    content: Option<usize>,
    tool_calls: Option<usize>,
    finish_reason: Option<usize>,
    usage: Option<usize>,
}

#[inline]
fn find_openai_chunk_key_positions(
    bytes: &[u8],
    need_role: bool,
    need_usage: bool,
) -> OpenAiChunkKeyPositions {
    let mut positions = OpenAiChunkKeyPositions::default();
    let mut cursor = 0usize;

    while let Some(rel) = memchr(b'"', &bytes[cursor..]) {
        let key_pos = cursor + rel;
        let tail = &bytes[key_pos..];

        if positions.content.is_none() && tail.starts_with(OPENAI_CONTENT_KEY) {
            positions.content = Some(key_pos);
        } else if positions.tool_calls.is_none() && tail.starts_with(OPENAI_TOOL_CALLS_KEY) {
            positions.tool_calls = Some(key_pos);
        } else if positions.finish_reason.is_none() && tail.starts_with(OPENAI_FINISH_REASON_KEY) {
            positions.finish_reason = Some(key_pos);
        } else if need_role && positions.role.is_none() && tail.starts_with(OPENAI_ROLE_KEY) {
            positions.role = Some(key_pos);
        } else if need_usage && positions.usage.is_none() && tail.starts_with(OPENAI_USAGE_KEY) {
            positions.usage = Some(key_pos);
        }

        let delta_key_found = positions.content.is_some() || positions.tool_calls.is_some();
        if delta_key_found
            && positions.finish_reason.is_some()
            && (!need_role || positions.role.is_some())
            && (!need_usage || positions.usage.is_some())
        {
            break;
        }

        cursor = key_pos + 1;
    }

    positions
}

#[inline]
fn try_fast_decode_anthropic_stream_event(
    event_type: &str,
    data: &[u8],
    mut decoder: Option<&mut StatefulAnthropicStreamDecoder>,
    out: &mut Vec<CanonicalStreamEvent>,
    emit_usage: bool,
) -> bool {
    let bytes = data;
    match event_type {
        "message_start" => {
            out.push(CanonicalStreamEvent::MessageStart {
                role: CanonicalRole::Assistant,
            });
            if emit_usage {
                let input_tokens = parse_u64_after_key(bytes, br#""input_tokens":"#);
                let output_tokens = parse_u64_after_key(bytes, br#""output_tokens":"#);
                if input_tokens.unwrap_or(0) > 0 || output_tokens.unwrap_or(0) > 0 {
                    out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
                        input_tokens,
                        output_tokens,
                        total_tokens: match (input_tokens, output_tokens) {
                            (Some(input), Some(output)) => input.checked_add(output),
                            _ => None,
                        },
                    }));
                }
            }
            true
        }
        "content_block_start" => {
            let Some(index_u64) = parse_u64_after_key(bytes, br#""index":"#) else {
                return false;
            };
            let Ok(index) = usize::try_from(index_u64) else {
                return false;
            };
            let Some(content_block_range) =
                parse_json_object_value_range_after_key(bytes, br#""content_block""#)
            else {
                return false;
            };
            let content_block = &bytes[content_block_range.start..content_block_range.end];

            if ANTHROPIC_TYPE_TOOL_USE_FINDER.find(content_block).is_some() {
                let Some(id) = parse_string_after_key_in(
                    bytes,
                    br#""id":"#,
                    content_block_range.start,
                    content_block_range.end,
                ) else {
                    return false;
                };
                let Some(name) = parse_string_after_key_in(
                    bytes,
                    br#""name":"#,
                    content_block_range.start,
                    content_block_range.end,
                ) else {
                    return false;
                };
                if let Some(decoder) = decoder.as_deref_mut() {
                    decoder.set_block_type_for_fast_path(index, true);
                }
                out.push(CanonicalStreamEvent::ToolCallStart { index, id, name });
                return true;
            }

            if let Some(decoder) = decoder.as_deref_mut() {
                decoder.set_block_type_for_fast_path(index, false);
            }

            if ANTHROPIC_TYPE_TEXT_FINDER.find(content_block).is_some() {
                if let Some(text) = parse_string_after_key_in_cow(
                    bytes,
                    br#""text":"#,
                    content_block_range.start,
                    content_block_range.end,
                ) {
                    if !text.is_empty() {
                        out.push(CanonicalStreamEvent::TextDelta(text.into_owned()));
                    }
                }
                return true;
            }

            if ANTHROPIC_TYPE_THINKING_FINDER.find(content_block).is_some() {
                if let Some(thinking) = parse_string_after_key_in_cow(
                    bytes,
                    br#""thinking":"#,
                    content_block_range.start,
                    content_block_range.end,
                ) {
                    if !thinking.is_empty() {
                        out.push(CanonicalStreamEvent::ReasoningDelta(thinking.into_owned()));
                    }
                }
                return true;
            }

            // tool_result and other non-canonical block starts have no output event.
            ANTHROPIC_TYPE_TOOL_RESULT_FINDER
                .find(content_block)
                .is_some()
        }
        "content_block_delta" => {
            if ANTHROPIC_TEXT_DELTA_FINDER.find(bytes).is_some() {
                if let Some(text) = parse_string_after_key(bytes, br#""text":"#) {
                    out.push(CanonicalStreamEvent::TextDelta(text));
                    return true;
                }
            }
            if ANTHROPIC_THINKING_DELTA_FINDER.find(bytes).is_some() {
                if let Some(thinking) = parse_string_after_key(bytes, br#""thinking":"#) {
                    out.push(CanonicalStreamEvent::ReasoningDelta(thinking));
                    return true;
                }
            }
            if ANTHROPIC_INPUT_JSON_DELTA_FINDER.find(bytes).is_some() {
                let Some(index_u64) = parse_u64_after_key(bytes, br#""index":"#) else {
                    return false;
                };
                let Ok(index) = usize::try_from(index_u64) else {
                    return false;
                };
                let Some(delta) = parse_string_after_key(bytes, br#""partial_json":"#) else {
                    return false;
                };
                out.push(CanonicalStreamEvent::ToolCallArgsDelta { index, delta });
                return true;
            }
            false
        }
        "message_stop" => {
            out.push(CanonicalStreamEvent::Done);
            true
        }
        "error" => {
            let Some(message) = parse_string_after_key(bytes, br#""message":"#) else {
                return false;
            };
            out.push(CanonicalStreamEvent::Error {
                status: 500,
                message,
            });
            true
        }
        "ping" => true,
        "message_delta" => {
            let mut produced = false;
            if emit_usage {
                let input_tokens = parse_u64_after_key(bytes, br#""input_tokens":"#);
                let output_tokens = parse_u64_after_key(bytes, br#""output_tokens":"#);
                if input_tokens.is_some() || output_tokens.is_some() {
                    out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
                        input_tokens,
                        output_tokens,
                        total_tokens: match (input_tokens, output_tokens) {
                            (Some(input), Some(output)) => input.checked_add(output),
                            _ => None,
                        },
                    }));
                    produced = true;
                }
            }
            if ANTHROPIC_STOP_REASON_NULL_FINDER.find(bytes).is_none() {
                if let Some(stop_reason) =
                    parse_anthropic_stop_reason_after_key(bytes, br#""stop_reason":"#)
                {
                    out.push(CanonicalStreamEvent::MessageEnd { stop_reason });
                    produced = true;
                }
            }
            produced
        }
        "content_block_stop" => {
            let Some(decoder) = decoder else {
                return false;
            };
            let Some(index_u64) = parse_u64_after_key(bytes, br#""index":"#) else {
                return false;
            };
            let Ok(index) = usize::try_from(index_u64) else {
                return false;
            };
            let event = AnthropicStreamEvent::ContentBlockStop { index };
            decoder.decode_into(&event, out);
            true
        }
        _ => false,
    }
}

#[inline]
fn try_fast_decode_responses_stream_event(
    event_type_hint: Option<&str>,
    data: &[u8],
    out: &mut Vec<CanonicalStreamEvent>,
    emit_usage: bool,
) -> bool {
    let bytes = data;
    let event_type = if let Some(hint) = event_type_hint {
        if hint.is_empty() {
            parse_responses_event_type_from_data(bytes)
        } else {
            parse_responses_event_type_from_str(hint)
        }
    } else {
        parse_responses_event_type_from_data(bytes)
    };

    match event_type {
        ResponsesEventType::Created => {
            out.push(CanonicalStreamEvent::MessageStart {
                role: CanonicalRole::Assistant,
            });
            true
        }
        ResponsesEventType::InProgress
        | ResponsesEventType::ContentPartAdded
        | ResponsesEventType::ContentPartDone
        | ResponsesEventType::OutputTextDone
        | ResponsesEventType::FunctionCallArgumentsDone => {
            // Known non-canonical events: short-circuit without serde fallback.
            true
        }
        ResponsesEventType::OutputTextDelta => {
            if let Some(delta) = parse_string_after_key_cow(bytes, br#""delta":"#) {
                if !delta.is_empty() {
                    out.push(CanonicalStreamEvent::TextDelta(delta.into_owned()));
                    return true;
                }
            }
            false
        }
        ResponsesEventType::FunctionCallArgumentsDelta => {
            let Some(index) = parse_u64_after_key(bytes, br#""output_index":"#) else {
                return false;
            };
            let Ok(index) = usize::try_from(index) else {
                return false;
            };
            let Some(delta) = parse_string_after_key(bytes, br#""delta":"#) else {
                return false;
            };
            out.push(CanonicalStreamEvent::ToolCallArgsDelta { index, delta });
            true
        }
        ResponsesEventType::OutputItemAdded => decode_responses_output_item_added(bytes, out),
        ResponsesEventType::OutputItemDone => decode_responses_output_item_done(bytes, out),
        ResponsesEventType::Completed => decode_responses_completed(bytes, out, emit_usage),
        ResponsesEventType::Error => {
            let Some(message) = parse_string_after_key(bytes, br#""message":"#) else {
                return false;
            };
            out.push(CanonicalStreamEvent::Error {
                status: 500,
                message,
            });
            true
        }
        ResponsesEventType::Unknown => false,
    }
}

#[inline]
fn decode_responses_output_item_added(bytes: &[u8], out: &mut Vec<CanonicalStreamEvent>) -> bool {
    const RESP_TYPE_FUNCTION_CALL: &[u8] = br#""type":"function_call""#;
    static RESP_TYPE_FUNCTION_CALL_FINDER: LazyLock<memmem::Finder<'static>> =
        LazyLock::new(|| memmem::Finder::new(RESP_TYPE_FUNCTION_CALL));
    let Some(global_type_pos) = RESP_TYPE_FUNCTION_CALL_FINDER.find(bytes) else {
        // Non-function-call items have no canonical stream equivalent.
        return true;
    };

    let Some(item_range) = parse_json_object_value_range_after_key(bytes, br#""item""#) else {
        return false;
    };
    let item_bytes = &bytes[item_range.start..item_range.end];
    if global_type_pos < item_range.start
        || global_type_pos >= item_range.end
            && RESP_TYPE_FUNCTION_CALL_FINDER.find(item_bytes).is_none()
    {
        // Non-function-call items have no canonical stream equivalent.
        return true;
    }

    let Some(index_u64) = parse_u64_after_key_in(bytes, br#""output_index":"#, 0, item_range.start)
        .or_else(|| parse_u64_after_key(bytes, br#""output_index":"#))
    else {
        return false;
    };
    let Ok(index) = usize::try_from(index_u64) else {
        return false;
    };
    let Some(id) =
        parse_string_after_key_in(bytes, br#""call_id":"#, item_range.start, item_range.end)
    else {
        return false;
    };
    let Some(name) =
        parse_string_after_key_in(bytes, br#""name":"#, item_range.start, item_range.end)
    else {
        return false;
    };
    out.push(CanonicalStreamEvent::ToolCallStart { index, id, name });
    true
}

#[inline]
fn decode_responses_output_item_done(bytes: &[u8], out: &mut Vec<CanonicalStreamEvent>) -> bool {
    const RESP_TYPE_FUNCTION_CALL_PREFIX: &[u8] = br#""type":"function_call"#;
    static RESP_TYPE_FUNCTION_CALL_PREFIX_FINDER: LazyLock<memmem::Finder<'static>> =
        LazyLock::new(|| memmem::Finder::new(RESP_TYPE_FUNCTION_CALL_PREFIX));
    let Some(global_type_pos) = RESP_TYPE_FUNCTION_CALL_PREFIX_FINDER.find(bytes) else {
        // Message item done has no canonical stream equivalent.
        return true;
    };

    let Some(item_range) = parse_json_object_value_range_after_key(bytes, br#""item""#) else {
        return false;
    };
    let item_bytes = &bytes[item_range.start..item_range.end];
    let type_pos = if global_type_pos >= item_range.start && global_type_pos < item_range.end {
        global_type_pos - item_range.start
    } else if let Some(local_pos) = RESP_TYPE_FUNCTION_CALL_PREFIX_FINDER.find(item_bytes) {
        local_pos
    } else {
        // Message item done has no canonical stream equivalent.
        return true;
    };
    let type_suffix_pos = type_pos + RESP_TYPE_FUNCTION_CALL_PREFIX.len();
    let is_function_call_output = item_bytes
        .get(type_suffix_pos..)
        .is_some_and(|suffix| suffix.starts_with(b"_output\""));
    let is_function_call = item_bytes.get(type_suffix_pos) == Some(&b'"');

    if is_function_call_output {
        let Some(tool_call_id) =
            parse_string_after_key_in(bytes, br#""call_id":"#, item_range.start, item_range.end)
        else {
            return false;
        };
        let Some(content) =
            parse_string_after_key_in(bytes, br#""output":"#, item_range.start, item_range.end)
        else {
            return false;
        };
        out.push(CanonicalStreamEvent::ToolResult {
            tool_call_id,
            content,
        });
        return true;
    }
    if !is_function_call {
        // Message item done has no canonical stream equivalent.
        return true;
    }

    let Some(index_u64) = parse_u64_after_key_in(bytes, br#""output_index":"#, 0, item_range.start)
        .or_else(|| parse_u64_after_key(bytes, br#""output_index":"#))
    else {
        return false;
    };
    let Ok(index) = usize::try_from(index_u64) else {
        return false;
    };
    let Some(call_id) =
        parse_string_after_key_in(bytes, br#""call_id":"#, item_range.start, item_range.end)
    else {
        return false;
    };
    let Some(call_name) =
        parse_string_after_key_in(bytes, br#""name":"#, item_range.start, item_range.end)
    else {
        return false;
    };
    out.push(CanonicalStreamEvent::ToolCallEnd {
        index,
        call_id: Some(call_id),
        call_name: Some(call_name),
    });
    true
}

#[inline]
fn decode_responses_completed(
    bytes: &[u8],
    out: &mut Vec<CanonicalStreamEvent>,
    emit_usage: bool,
) -> bool {
    static RESP_TYPE_FUNCTION_CALL_ANY_FINDER: LazyLock<memmem::Finder<'static>> =
        LazyLock::new(|| memmem::Finder::new(br#""type":"function_call"#));

    let (input_tokens, output_tokens, total_tokens) = if emit_usage {
        let usage_range = parse_json_object_value_range_after_key(bytes, br#""usage""#);
        if let Some(range) = usage_range {
            let input_tokens =
                parse_u64_after_key_in(bytes, br#""input_tokens":"#, range.start, range.end);
            let output_tokens =
                parse_u64_after_key_in(bytes, br#""output_tokens":"#, range.start, range.end);
            let total_tokens =
                parse_u64_after_key_in(bytes, br#""total_tokens":"#, range.start, range.end)
                    .or_else(|| match (input_tokens, output_tokens) {
                        (Some(input), Some(output)) => input.checked_add(output),
                        _ => None,
                    });
            (input_tokens, output_tokens, total_tokens)
        } else {
            (None, None, None)
        }
    } else {
        (None, None, None)
    };
    if emit_usage && (input_tokens.is_some() || output_tokens.is_some() || total_tokens.is_some()) {
        out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
            input_tokens,
            output_tokens,
            total_tokens,
        }));
    }

    let stop_reason = if RESP_TYPE_FUNCTION_CALL_ANY_FINDER.find(bytes).is_some() {
        CanonicalStopReason::ToolCalls
    } else {
        CanonicalStopReason::EndOfTurn
    };
    out.push(CanonicalStreamEvent::MessageEnd { stop_reason });
    out.push(CanonicalStreamEvent::Done);
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResponsesEventType {
    Created,
    InProgress,
    ContentPartAdded,
    ContentPartDone,
    OutputTextDone,
    FunctionCallArgumentsDone,
    OutputTextDelta,
    FunctionCallArgumentsDelta,
    OutputItemAdded,
    OutputItemDone,
    Completed,
    Error,
    Unknown,
}

#[inline]
fn parse_responses_event_type_from_str(event_type: &str) -> ResponsesEventType {
    match event_type {
        "response.created" => ResponsesEventType::Created,
        "response.in_progress" => ResponsesEventType::InProgress,
        "response.content_part.added" => ResponsesEventType::ContentPartAdded,
        "response.content_part.done" => ResponsesEventType::ContentPartDone,
        "response.output_text.done" => ResponsesEventType::OutputTextDone,
        "response.function_call_arguments.done" => ResponsesEventType::FunctionCallArgumentsDone,
        "response.output_text.delta" => ResponsesEventType::OutputTextDelta,
        "response.function_call_arguments.delta" => ResponsesEventType::FunctionCallArgumentsDelta,
        "response.output_item.added" => ResponsesEventType::OutputItemAdded,
        "response.output_item.done" => ResponsesEventType::OutputItemDone,
        "response.completed" => ResponsesEventType::Completed,
        "error" => ResponsesEventType::Error,
        _ => ResponsesEventType::Unknown,
    }
}

#[inline]
fn parse_responses_event_type_from_data(bytes: &[u8]) -> ResponsesEventType {
    let Some(event_type) = parse_unescaped_string_slice_after_key(bytes, br#""type":"#) else {
        return ResponsesEventType::Unknown;
    };
    match event_type {
        b"response.created" => ResponsesEventType::Created,
        b"response.in_progress" => ResponsesEventType::InProgress,
        b"response.content_part.added" => ResponsesEventType::ContentPartAdded,
        b"response.content_part.done" => ResponsesEventType::ContentPartDone,
        b"response.output_text.done" => ResponsesEventType::OutputTextDone,
        b"response.function_call_arguments.done" => ResponsesEventType::FunctionCallArgumentsDone,
        b"response.output_text.delta" => ResponsesEventType::OutputTextDelta,
        b"response.function_call_arguments.delta" => ResponsesEventType::FunctionCallArgumentsDelta,
        b"response.output_item.added" => ResponsesEventType::OutputItemAdded,
        b"response.output_item.done" => ResponsesEventType::OutputItemDone,
        b"response.completed" => ResponsesEventType::Completed,
        b"error" => ResponsesEventType::Error,
        _ => ResponsesEventType::Unknown,
    }
}

#[inline]
fn try_fast_decode_gemini_stream_chunk(
    data: &[u8],
    out: &mut Vec<CanonicalStreamEvent>,
    emit_usage: bool,
) -> bool {
    static GEMINI_CANDIDATES_FINDER: LazyLock<memmem::Finder<'static>> =
        LazyLock::new(|| memmem::Finder::new(br#""candidates""#));
    static GEMINI_FUNCTION_CALL_FINDER: LazyLock<memmem::Finder<'static>> =
        LazyLock::new(|| memmem::Finder::new(br#""functionCall""#));

    let bytes = data;
    if GEMINI_CANDIDATES_FINDER.find(bytes).is_none() {
        return false;
    }

    let mut produced = false;
    let mut has_tool_calls = false;

    if GEMINI_FUNCTION_CALL_FINDER.find(bytes).is_some() {
        let Some(fc_range) = parse_json_object_value_range_after_key(bytes, br#""functionCall""#)
        else {
            return false;
        };
        let Some(name) =
            parse_string_after_key_in(bytes, br#""name":"#, fc_range.start, fc_range.end)
        else {
            return false;
        };
        let Some(args_json) = parse_json_value_string_after_key_in(
            bytes,
            br#""args":"#,
            fc_range.start,
            fc_range.end,
        ) else {
            return false;
        };
        out.push(CanonicalStreamEvent::ToolCallStart {
            index: 0,
            id: next_call_id(),
            name,
        });
        out.push(CanonicalStreamEvent::ToolCallArgsDelta {
            index: 0,
            delta: args_json,
        });
        out.push(CanonicalStreamEvent::ToolCallEnd {
            index: 0,
            call_id: None,
            call_name: None,
        });
        produced = true;
        has_tool_calls = true;
    } else if let Some(text) = parse_string_after_key(bytes, br#""text":"#) {
        if !text.is_empty() {
            out.push(CanonicalStreamEvent::TextDelta(text));
            produced = true;
        }
    }

    if let Some(stop_reason) = parse_gemini_stop_reason_after_key(bytes, br#""finishReason":"#) {
        let stop_reason = if has_tool_calls && stop_reason == CanonicalStopReason::EndOfTurn {
            CanonicalStopReason::ToolCalls
        } else {
            stop_reason
        };
        out.push(CanonicalStreamEvent::MessageEnd { stop_reason });
        produced = true;
    }
    if emit_usage {
        if let Some(usage_range) =
            parse_json_object_value_range_after_key(bytes, br#""usageMetadata""#)
        {
            let input_tokens = parse_u64_after_key_in(
                bytes,
                br#""promptTokenCount":"#,
                usage_range.start,
                usage_range.end,
            );
            let output_tokens = parse_u64_after_key_in(
                bytes,
                br#""candidatesTokenCount":"#,
                usage_range.start,
                usage_range.end,
            );
            let total_tokens = parse_u64_after_key_in(
                bytes,
                br#""totalTokenCount":"#,
                usage_range.start,
                usage_range.end,
            );
            if input_tokens.is_some() || output_tokens.is_some() || total_tokens.is_some() {
                out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
                    input_tokens,
                    output_tokens,
                    total_tokens,
                }));
                produced = true;
            }
        }
    }

    produced
}

#[inline]
fn parse_openai_role_after_key_pos(
    bytes: &[u8],
    key_pos: usize,
    key_len: usize,
) -> Option<CanonicalRole> {
    let value_start = skip_ws(bytes, key_pos + key_len);
    if bytes.get(value_start) != Some(&b'"') {
        return None;
    }
    let value_end = parse_json_string_end(bytes, value_start).ok()?;
    let inner = &bytes[value_start + 1..value_end - 1];
    if memchr(b'\\', inner).is_none() {
        return Some(match inner {
            b"user" => CanonicalRole::User,
            b"system" => CanonicalRole::System,
            b"tool" => CanonicalRole::Tool,
            _ => CanonicalRole::Assistant,
        });
    }

    let decoded = serde_json::from_slice::<String>(&bytes[value_start..value_end]).ok()?;
    Some(match decoded.as_str() {
        "user" => CanonicalRole::User,
        "system" => CanonicalRole::System,
        "tool" => CanonicalRole::Tool,
        _ => CanonicalRole::Assistant,
    })
}

#[inline]
fn parse_openai_stop_reason_from_string_value(
    bytes: &[u8],
    value_start: usize,
) -> Option<CanonicalStopReason> {
    if bytes.get(value_start) != Some(&b'"') {
        return None;
    }
    let value_end = parse_json_string_end(bytes, value_start).ok()?;
    let inner = &bytes[value_start + 1..value_end - 1];
    if memchr(b'\\', inner).is_none() {
        return Some(match inner {
            b"tool_calls" => CanonicalStopReason::ToolCalls,
            b"length" => CanonicalStopReason::MaxTokens,
            b"content_filter" => CanonicalStopReason::ContentFilter,
            _ => CanonicalStopReason::EndOfTurn,
        });
    }

    let decoded = serde_json::from_slice::<String>(&bytes[value_start..value_end]).ok()?;
    Some(openai_stop_to_canonical(&decoded))
}

#[inline]
fn parse_gemini_stop_reason_after_key(
    bytes: &[u8],
    key_pattern: &[u8],
) -> Option<CanonicalStopReason> {
    let key_pos = memmem::find(bytes, key_pattern)?;
    let value_start = skip_ws(bytes, key_pos + key_pattern.len());
    if bytes.get(value_start) != Some(&b'"') {
        return None;
    }
    let value_end = parse_json_string_end(bytes, value_start).ok()?;
    let inner = &bytes[value_start + 1..value_end - 1];
    if memchr(b'\\', inner).is_none() {
        return Some(match inner {
            b"MAX_TOKENS" => CanonicalStopReason::MaxTokens,
            b"SAFETY" | b"RECITATION" => CanonicalStopReason::ContentFilter,
            _ => CanonicalStopReason::EndOfTurn,
        });
    }

    let decoded = serde_json::from_slice::<String>(&bytes[value_start..value_end]).ok()?;
    Some(gemini_stop_to_canonical(&decoded))
}

#[inline]
fn parse_anthropic_stop_reason_after_key(
    bytes: &[u8],
    key_pattern: &[u8],
) -> Option<CanonicalStopReason> {
    let key_pos = memmem::find(bytes, key_pattern)?;
    let value_start = skip_ws(bytes, key_pos + key_pattern.len());
    if bytes.get(value_start) != Some(&b'"') {
        return None;
    }
    let value_end = parse_json_string_end(bytes, value_start).ok()?;
    let inner = &bytes[value_start + 1..value_end - 1];
    if memchr(b'\\', inner).is_none() {
        return Some(match inner {
            b"tool_use" => CanonicalStopReason::ToolCalls,
            b"max_tokens" | b"model_context_window_exceeded" => CanonicalStopReason::MaxTokens,
            b"refusal" => CanonicalStopReason::ContentFilter,
            _ => CanonicalStopReason::EndOfTurn,
        });
    }

    let decoded = serde_json::from_slice::<String>(&bytes[value_start..value_end]).ok()?;
    Some(anthropic_stop_to_canonical(&decoded))
}

#[inline]
fn try_fast_decode_openai_tool_calls_chunk_at(
    bytes: &[u8],
    key_pos: usize,
    out: &mut Vec<CanonicalStreamEvent>,
) -> Option<bool> {
    let colon_pos = skip_ws(bytes, key_pos + br#""tool_calls""#.len());
    if bytes.get(colon_pos) != Some(&b':') {
        return None;
    }
    let array_start = skip_ws(bytes, colon_pos + 1);
    if bytes.get(array_start) != Some(&b'[') {
        return None;
    }

    let mut cursor = skip_ws(bytes, array_start + 1);
    if bytes.get(cursor) == Some(&b']') {
        return Some(false);
    }
    if bytes.get(cursor) != Some(&b'{') {
        return None;
    }
    let call_obj_start = cursor;
    let call_obj_end = parse_json_value_end(bytes, call_obj_start).ok()?;
    cursor = skip_ws(bytes, call_obj_end);
    if bytes.get(cursor) != Some(&b']') {
        return None;
    }

    let index_u64 = parse_u64_after_key_in(bytes, br#""index":"#, call_obj_start, call_obj_end)?;
    let index = usize::try_from(index_u64).ok()?;
    let call_id = parse_string_after_key_in(bytes, br#""id":"#, call_obj_start, call_obj_end);
    let call_name = if call_id.is_some() {
        parse_string_after_key_in(bytes, br#""name":"#, call_obj_start, call_obj_end)
    } else {
        None
    };
    let arguments =
        parse_string_after_key_in(bytes, br#""arguments":"#, call_obj_start, call_obj_end);

    let mut produced = false;
    if let Some(id) = call_id {
        out.push(CanonicalStreamEvent::ToolCallStart {
            index,
            id,
            name: call_name.unwrap_or_default(),
        });
        produced = true;
    }
    if let Some(delta) = arguments {
        if !delta.is_empty() {
            out.push(CanonicalStreamEvent::ToolCallArgsDelta { index, delta });
            produced = true;
        }
    }

    Some(produced)
}

#[inline]
fn parse_openai_usage_after_key_pos(
    bytes: &[u8],
    key_pos: usize,
    key_len: usize,
) -> Option<CanonicalUsage> {
    let usage_range = parse_json_object_value_range_after_key_pos(bytes, key_pos, key_len)?;
    let input_tokens = parse_u64_after_key_in(
        bytes,
        br#""prompt_tokens":"#,
        usage_range.start,
        usage_range.end,
    )?;
    let output_tokens = parse_u64_after_key_in(
        bytes,
        br#""completion_tokens":"#,
        usage_range.start,
        usage_range.end,
    )?;
    let total_tokens = parse_u64_after_key_in(
        bytes,
        br#""total_tokens":"#,
        usage_range.start,
        usage_range.end,
    )?;
    Some(CanonicalUsage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
        total_tokens: Some(total_tokens),
    })
}

#[inline]
fn parse_json_object_value_range_after_key(
    bytes: &[u8],
    key_pattern: &[u8],
) -> Option<std::ops::Range<usize>> {
    let key_pos = memmem::find(bytes, key_pattern)?;
    parse_json_object_value_range_after_key_pos(bytes, key_pos, key_pattern.len())
}

#[inline]
fn parse_json_object_value_range_after_key_pos(
    bytes: &[u8],
    key_pos: usize,
    key_len: usize,
) -> Option<std::ops::Range<usize>> {
    let colon_pos = skip_ws(bytes, key_pos + key_len);
    if bytes.get(colon_pos) != Some(&b':') {
        return None;
    }
    let value_start = skip_ws(bytes, colon_pos + 1);
    if bytes.get(value_start) != Some(&b'{') {
        return None;
    }
    let value_end = parse_json_value_end(bytes, value_start).ok()?;
    Some(value_start..value_end)
}

fn parse_string_after_key_in(
    bytes: &[u8],
    key_pattern: &[u8],
    search_start: usize,
    search_end: usize,
) -> Option<String> {
    parse_string_after_key_in_cow(bytes, key_pattern, search_start, search_end).map(Cow::into_owned)
}

#[inline]
fn parse_string_after_key_in_cow<'a>(
    bytes: &'a [u8],
    key_pattern: &[u8],
    search_start: usize,
    search_end: usize,
) -> Option<Cow<'a, str>> {
    let key_pos = search_start + memmem::find(&bytes[search_start..search_end], key_pattern)?;
    parse_string_after_key_pos_bounded_cow(bytes, key_pos, key_pattern.len(), search_end)
}

#[inline]
fn parse_u64_after_key_in(
    bytes: &[u8],
    key_pattern: &[u8],
    search_start: usize,
    search_end: usize,
) -> Option<u64> {
    const U64_MAX_DIV10: u64 = u64::MAX / 10;
    const U64_MAX_MOD10: u8 = (u64::MAX % 10) as u8;

    let key_pos = search_start + memmem::find(&bytes[search_start..search_end], key_pattern)?;
    let mut i = skip_ws(bytes, key_pos + key_pattern.len());
    let mut value = 0_u64;
    let mut saw_digit = false;
    while i < search_end {
        let ch = bytes[i];
        if !ch.is_ascii_digit() {
            break;
        }
        saw_digit = true;
        let digit = ch - b'0';
        if value > U64_MAX_DIV10 || (value == U64_MAX_DIV10 && digit > U64_MAX_MOD10) {
            return None;
        }
        value = value * 10 + u64::from(digit);
        i += 1;
    }
    saw_digit.then_some(value)
}

#[inline]
fn raw_frame_terminated(raw_frame: &[u8]) -> bool {
    raw_frame.ends_with(b"\n\n") || raw_frame.ends_with(b"\r\n\r\n")
}

#[inline]
fn parse_json_value_string_after_key_in(
    bytes: &[u8],
    key_pattern: &[u8],
    search_start: usize,
    search_end: usize,
) -> Option<String> {
    let key_pos = search_start + memmem::find(&bytes[search_start..search_end], key_pattern)?;
    let colon_pos = skip_ws(bytes, key_pos + key_pattern.len());
    if colon_pos >= search_end || bytes.get(colon_pos) != Some(&b':') {
        return None;
    }
    let value_start = skip_ws(bytes, colon_pos + 1);
    if value_start >= search_end {
        return None;
    }
    let value_end = parse_json_value_end(bytes, value_start).ok()?;
    if value_end > search_end {
        return None;
    }
    std::str::from_utf8(&bytes[value_start..value_end])
        .ok()
        .map(ToOwned::to_owned)
}

#[inline]
fn parse_string_after_key(bytes: &[u8], key_pattern: &[u8]) -> Option<String> {
    parse_string_after_key_cow(bytes, key_pattern).map(Cow::into_owned)
}

#[inline]
fn parse_string_after_key_cow<'a>(bytes: &'a [u8], key_pattern: &[u8]) -> Option<Cow<'a, str>> {
    let key_pos = memmem::find(bytes, key_pattern)?;
    parse_string_after_key_pos_cow(bytes, key_pos, key_pattern.len())
}

#[inline]
fn parse_string_after_key_pos_cow(
    bytes: &[u8],
    key_pos: usize,
    key_len: usize,
) -> Option<Cow<'_, str>> {
    parse_string_after_key_pos_bounded_cow(bytes, key_pos, key_len, bytes.len())
}

#[inline]
fn parse_string_after_key_pos_bounded_cow(
    bytes: &[u8],
    key_pos: usize,
    key_len: usize,
    search_end: usize,
) -> Option<Cow<'_, str>> {
    let value_start = skip_ws(bytes, key_pos + key_len);
    if value_start >= search_end || bytes.get(value_start) != Some(&b'"') {
        return None;
    }
    let value_end = parse_json_string_end(bytes, value_start).ok()?;
    if value_end > search_end {
        return None;
    }
    let inner = &bytes[value_start + 1..value_end - 1];
    if memchr(b'\\', inner).is_none() {
        return std::str::from_utf8(inner).ok().map(Cow::Borrowed);
    }
    serde_json::from_slice::<String>(&bytes[value_start..value_end])
        .ok()
        .map(Cow::Owned)
}

#[inline]
fn parse_unescaped_string_slice_after_key<'a>(
    bytes: &'a [u8],
    key_pattern: &[u8],
) -> Option<&'a [u8]> {
    let key_pos = memmem::find(bytes, key_pattern)?;
    let value_start = skip_ws(bytes, key_pos + key_pattern.len());
    if bytes.get(value_start) != Some(&b'"') {
        return None;
    }
    let value_end = parse_json_string_end(bytes, value_start).ok()?;
    let inner = &bytes[value_start + 1..value_end - 1];
    if memchr(b'\\', inner).is_some() {
        return None;
    }
    Some(inner)
}

#[inline]
fn parse_u64_after_key(bytes: &[u8], key_pattern: &[u8]) -> Option<u64> {
    const U64_MAX_DIV10: u64 = u64::MAX / 10;
    const U64_MAX_MOD10: u8 = (u64::MAX % 10) as u8;

    let key_pos = memmem::find(bytes, key_pattern)?;
    let mut i = skip_ws(bytes, key_pos + key_pattern.len());
    let mut value = 0_u64;
    let mut saw_digit = false;
    while let Some(ch) = bytes.get(i) {
        if !ch.is_ascii_digit() {
            break;
        }
        saw_digit = true;
        let digit = ch - b'0';
        if value > U64_MAX_DIV10 || (value == U64_MAX_DIV10 && digit > U64_MAX_MOD10) {
            return None;
        }
        value = value * 10 + u64::from(digit);
        i += 1;
    }
    saw_digit.then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::CanonicalStopReason;

    fn providers() -> [ProviderKind; 5] {
        [
            ProviderKind::OpenAi,
            ProviderKind::OpenAiResponses,
            ProviderKind::Anthropic,
            ProviderKind::Gemini,
            ProviderKind::GeminiOpenAi,
        ]
    }

    fn ingress_apis() -> [IngressApi; 4] {
        [
            IngressApi::OpenAiChat,
            IngressApi::OpenAiResponses,
            IngressApi::Anthropic,
            IngressApi::Gemini,
        ]
    }

    fn sample_text_delta_frame(provider: ProviderKind) -> SseEvent {
        match provider {
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => SseEvent {
                event: None,
                data: serde_json::json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "model": "m1",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "matrix"},
                        "finish_reason": null
                    }]
                })
                .to_string(),
                id: None,
                retry: None,
            },
            ProviderKind::OpenAiResponses => SseEvent {
                event: Some("response.output_text.delta".into()),
                data: serde_json::json!({
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "matrix"
                })
                .to_string(),
                id: None,
                retry: None,
            },
            ProviderKind::Anthropic => SseEvent {
                event: Some("content_block_delta".into()),
                data: serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "matrix"}
                })
                .to_string(),
                id: None,
                retry: None,
            },
            ProviderKind::Gemini => SseEvent {
                event: None,
                data: serde_json::json!({
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{"text": "matrix"}]
                        },
                        "index": 0
                    }]
                })
                .to_string(),
                id: None,
                retry: None,
            },
        }
    }

    fn sample_done_frame(provider: ProviderKind) -> SseEvent {
        match provider {
            ProviderKind::Anthropic => SseEvent {
                event: Some("message_stop".into()),
                data: serde_json::json!({
                    "type": "message_stop"
                })
                .to_string(),
                id: None,
                retry: None,
            },
            ProviderKind::OpenAi
            | ProviderKind::OpenAiResponses
            | ProviderKind::Gemini
            | ProviderKind::GeminiOpenAi => SseEvent {
                event: None,
                data: "[DONE]".into(),
                id: None,
                retry: None,
            },
        }
    }

    fn sample_tool_call_frame(provider: ProviderKind) -> SseEvent {
        match provider {
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => SseEvent {
                event: None,
                data: serde_json::json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "model": "m1",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}
                            }]
                        },
                        "finish_reason": null
                    }]
                })
                .to_string(),
                id: None,
                retry: None,
            },
            ProviderKind::OpenAiResponses => SseEvent {
                event: Some("response.output_item.added".into()),
                data: serde_json::json!({
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_0",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": "{\"q\":\"x\"}"
                    }
                })
                .to_string(),
                id: None,
                retry: None,
            },
            ProviderKind::Anthropic => SseEvent {
                event: Some("content_block_start".into()),
                data: serde_json::json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "lookup",
                        "input": {"q": "x"}
                    }
                })
                .to_string(),
                id: None,
                retry: None,
            },
            ProviderKind::Gemini => SseEvent {
                event: None,
                data: serde_json::json!({
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{
                                "functionCall": {
                                    "name": "lookup",
                                    "args": {"q": "x"}
                                }
                            }]
                        },
                        "index": 0
                    }]
                })
                .to_string(),
                id: None,
                retry: None,
            },
        }
    }

    fn sample_usage_frame(provider: ProviderKind) -> Option<SseEvent> {
        match provider {
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => Some(SseEvent {
                event: None,
                data: serde_json::json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "model": "m1",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15
                    }
                })
                .to_string(),
                id: None,
                retry: None,
            }),
            ProviderKind::Anthropic => Some(SseEvent {
                event: Some("message_delta".into()),
                data: serde_json::json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": null,
                        "stop_sequence": null
                    },
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5
                    }
                })
                .to_string(),
                id: None,
                retry: None,
            }),
            ProviderKind::Gemini => Some(SseEvent {
                event: None,
                data: serde_json::json!({
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15
                    }
                })
                .to_string(),
                id: None,
                retry: None,
            }),
            ProviderKind::OpenAiResponses => None,
        }
    }

    fn sample_reasoning_frame() -> SseEvent {
        SseEvent {
            event: Some("content_block_delta".into()),
            data: serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "reason"}
            })
            .to_string(),
            id: None,
            retry: None,
        }
    }

    fn sample_error_frame(provider: ProviderKind) -> Option<SseEvent> {
        match provider {
            ProviderKind::Anthropic => Some(SseEvent {
                event: Some("error".into()),
                data: serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "boom"
                    }
                })
                .to_string(),
                id: None,
                retry: None,
            }),
            ProviderKind::OpenAiResponses => Some(SseEvent {
                event: Some("error".into()),
                data: serde_json::json!({
                    "type": "error",
                    "message": "boom"
                })
                .to_string(),
                id: None,
                retry: None,
            }),
            ProviderKind::OpenAi | ProviderKind::Gemini | ProviderKind::GeminiOpenAi => None,
        }
    }

    fn sample_success_combo_frames(provider: ProviderKind) -> Vec<SseEvent> {
        let mut frames = Vec::with_capacity(5);
        frames.push(sample_text_delta_frame(provider));
        if provider == ProviderKind::Anthropic {
            frames.push(sample_reasoning_frame());
        }
        frames.push(sample_tool_call_frame(provider));
        if let Some(usage) = sample_usage_frame(provider) {
            frames.push(usage);
        }
        frames.push(sample_done_frame(provider));
        frames
    }

    fn sample_error_combo_frames(provider: ProviderKind) -> Option<Vec<SseEvent>> {
        let error = sample_error_frame(provider)?;
        let mut frames = Vec::with_capacity(4);
        frames.push(sample_text_delta_frame(provider));
        if provider == ProviderKind::Anthropic {
            frames.push(sample_reasoning_frame());
        }
        frames.push(sample_tool_call_frame(provider));
        frames.push(error);
        Some(frames)
    }

    fn expected_marker(api: IngressApi) -> &'static str {
        match api {
            IngressApi::OpenAiChat => "\"chat.completion.chunk\"",
            IngressApi::OpenAiResponses => "event: response.output_text.delta",
            IngressApi::Anthropic => "event: content_block_delta",
            IngressApi::Gemini => "\"candidates\"",
        }
    }

    fn as_raw_sse_frame(frame: &SseEvent) -> String {
        let mut raw = String::new();
        if let Some(event) = frame.event.as_deref() {
            raw.push_str("event: ");
            raw.push_str(event);
            raw.push('\n');
        }
        raw.push_str("data: ");
        raw.push_str(&frame.data);
        raw.push_str("\n\n");
        raw
    }

    fn normalize_generated_call_ids(input: &str) -> String {
        let mut out = String::with_capacity(input.len());
        let mut rest = input;
        while let Some(pos) = rest.find("call_") {
            out.push_str(&rest[..pos]);
            let tail = &rest[pos + 5..];
            let hex_len = tail
                .as_bytes()
                .iter()
                .take_while(|byte| byte.is_ascii_hexdigit())
                .count();
            if hex_len == 0 {
                out.push_str("call_");
                rest = tail;
                continue;
            }
            out.push_str("call_<id>");
            rest = &tail[hex_len..];
        }
        out.push_str(rest);
        out
    }

    #[test]
    fn test_stream_text_delta_transcode_matrix_5x4() {
        for provider in providers() {
            let frame = sample_text_delta_frame(provider);
            for api in ingress_apis() {
                let mut t = StreamTranscoder::new(provider, api, "m1".into(), "id-1".into());
                let chunks = t.transcode_frame(&frame);
                assert!(
                    !chunks.is_empty(),
                    "expected non-empty transcode output for provider={provider:?} api={api:?}"
                );
                assert!(
                    chunks[0].contains(expected_marker(api)),
                    "unexpected transcode marker for provider={provider:?} api={api:?}: {}",
                    chunks[0]
                );
            }
        }
    }

    #[test]
    fn test_stream_done_transcode_matrix_5x4() {
        for provider in providers() {
            let frame = sample_done_frame(provider);
            for api in ingress_apis() {
                let mut t = StreamTranscoder::new(provider, api, "m1".into(), "id-1".into());
                let chunks = t.transcode_frame(&frame);
                match api {
                    IngressApi::OpenAiChat => {
                        assert_eq!(
                            chunks,
                            vec!["data: [DONE]\n\n".to_string()],
                            "unexpected done output for provider={provider:?} api={api:?}"
                        );
                    }
                    IngressApi::OpenAiResponses => {
                        assert!(
                            chunks
                                .iter()
                                .any(|chunk| chunk.contains("event: response.completed")),
                            "missing responses done output for provider={provider:?} api={api:?}"
                        );
                    }
                    IngressApi::Anthropic => {
                        assert!(
                            chunks
                                .iter()
                                .any(|chunk| chunk.contains("event: message_stop")),
                            "missing anthropic done output for provider={provider:?} api={api:?}"
                        );
                    }
                    IngressApi::Gemini => {
                        assert!(
                            chunks.is_empty(),
                            "gemini target should not emit explicit done for provider={provider:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_stream_tool_call_transcode_matrix_5x4() {
        for provider in providers() {
            let frame = sample_tool_call_frame(provider);
            for api in ingress_apis() {
                let mut t = StreamTranscoder::new(provider, api, "m1".into(), "id-1".into());
                let chunks = t.transcode_frame(&frame);
                match api {
                    IngressApi::OpenAiChat => {
                        assert!(
                            chunks.iter().any(|chunk| chunk.contains("\"tool_calls\"")),
                            "missing openai tool call output for provider={provider:?}"
                        );
                    }
                    IngressApi::OpenAiResponses => {
                        assert!(
                            chunks
                                .iter()
                                .any(|chunk| chunk.contains("event: response.output_item.added")),
                            "missing responses tool call output for provider={provider:?}"
                        );
                    }
                    IngressApi::Anthropic => {
                        assert!(
                            chunks
                                .iter()
                                .any(|chunk| chunk.contains("event: content_block_start")),
                            "missing anthropic tool call output for provider={provider:?}"
                        );
                    }
                    IngressApi::Gemini => {
                        assert!(
                            chunks
                                .iter()
                                .any(|chunk| chunk.contains("\"functionCall\"")),
                            "missing gemini tool call output for provider={provider:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_stream_usage_transcode_matrix() {
        for provider in providers() {
            let Some(frame) = sample_usage_frame(provider) else {
                continue;
            };
            for api in ingress_apis() {
                let mut t = StreamTranscoder::new(provider, api, "m1".into(), "id-1".into());
                let chunks = t.transcode_frame(&frame);
                match api {
                    IngressApi::OpenAiChat => {
                        assert!(
                            chunks.iter().any(|chunk| chunk.contains("\"usage\"")),
                            "missing openai usage output for provider={provider:?}"
                        );
                    }
                    IngressApi::OpenAiResponses => {
                        assert!(
                            chunks.is_empty(),
                            "responses target should suppress standalone usage for provider={provider:?}"
                        );
                    }
                    IngressApi::Anthropic => {
                        assert!(
                            chunks.is_empty(),
                            "anthropic target should suppress standalone usage for provider={provider:?}"
                        );
                    }
                    IngressApi::Gemini => {
                        assert!(
                            chunks
                                .iter()
                                .any(|chunk| chunk.contains("\"usageMetadata\"")),
                            "missing gemini usage output for provider={provider:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_stream_reasoning_transcode_matrix_from_anthropic() {
        let frame = sample_reasoning_frame();
        for api in ingress_apis() {
            let mut t =
                StreamTranscoder::new(ProviderKind::Anthropic, api, "m1".into(), "id-1".into());
            let chunks = t.transcode_frame(&frame);
            match api {
                IngressApi::OpenAiChat | IngressApi::OpenAiResponses => {
                    assert!(
                        chunks.is_empty(),
                        "reasoning should not emit for api={api:?}: {chunks:?}"
                    );
                }
                IngressApi::Anthropic => {
                    assert!(
                        chunks
                            .iter()
                            .any(|chunk| chunk.contains("\"thinking_delta\"")),
                        "missing anthropic reasoning output"
                    );
                }
                IngressApi::Gemini => {
                    assert!(
                        chunks
                            .iter()
                            .any(|chunk| chunk.contains("\"text\":\"reason\"")),
                        "missing gemini reasoning output"
                    );
                }
            }
        }
    }

    #[test]
    fn test_stream_error_transcode_matrix() {
        for provider in [ProviderKind::Anthropic, ProviderKind::OpenAiResponses] {
            let frame = sample_error_frame(provider).expect("error frame");
            for api in ingress_apis() {
                let mut t = StreamTranscoder::new(provider, api, "m1".into(), "id-1".into());
                let chunks = t.transcode_frame(&frame);
                assert!(
                    !chunks.is_empty(),
                    "error should always produce output for provider={provider:?} api={api:?}"
                );
                match api {
                    IngressApi::OpenAiResponses => {
                        assert!(
                            chunks.iter().any(|chunk| chunk.contains("event: error")),
                            "missing responses error output"
                        );
                    }
                    IngressApi::Anthropic => {
                        assert!(
                            chunks.iter().any(|chunk| chunk.contains("event: error")),
                            "missing anthropic error output"
                        );
                    }
                    IngressApi::OpenAiChat | IngressApi::Gemini => {
                        assert!(
                            chunks.iter().any(|chunk| chunk.contains("\"error\"")),
                            "missing json error output"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_stream_success_combo_sequence_matrix_5x4() {
        for provider in providers() {
            let frames = sample_success_combo_frames(provider);
            for api in ingress_apis() {
                let mut transcoder =
                    StreamTranscoder::new(provider, api, "m1".to_string(), "id-1".to_string());
                let mut chunks: Vec<String> = Vec::new();
                for frame in &frames {
                    chunks.extend(transcoder.transcode_frame(frame));
                }
                assert!(
                    !chunks.is_empty(),
                    "expected non-empty combo output for provider={provider:?} api={api:?}"
                );
                match api {
                    IngressApi::OpenAiChat => {
                        assert!(
                            chunks.iter().any(|chunk| chunk.contains("data: [DONE]")),
                            "missing openai done in combo output for provider={provider:?}"
                        );
                    }
                    IngressApi::OpenAiResponses => {
                        assert!(
                            chunks
                                .iter()
                                .any(|chunk| chunk.contains("event: response.completed")),
                            "missing responses done in combo output for provider={provider:?}"
                        );
                    }
                    IngressApi::Anthropic => {
                        assert!(
                            chunks
                                .iter()
                                .any(|chunk| chunk.contains("event: message_stop")),
                            "missing anthropic done in combo output for provider={provider:?}"
                        );
                    }
                    IngressApi::Gemini => {
                        assert!(
                            chunks.iter().any(|chunk| chunk.contains("\"candidates\"")),
                            "missing gemini content in combo output for provider={provider:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_stream_error_combo_sequence_matrix() {
        for provider in [ProviderKind::Anthropic, ProviderKind::OpenAiResponses] {
            let frames = sample_error_combo_frames(provider).expect("error combo");
            for api in ingress_apis() {
                let mut transcoder =
                    StreamTranscoder::new(provider, api, "m1".to_string(), "id-1".to_string());
                let mut chunks: Vec<String> = Vec::new();
                for frame in &frames {
                    chunks.extend(transcoder.transcode_frame(frame));
                }
                assert!(
                    chunks.iter().any(|chunk| chunk.contains("error")),
                    "missing error marker in combo output for provider={provider:?} api={api:?}"
                );
            }
        }
    }

    #[test]
    fn test_raw_frame_transcode_matches_parsed_frame_matrix_5x4() {
        for provider in providers() {
            let frames = [
                sample_text_delta_frame(provider),
                sample_tool_call_frame(provider),
                sample_done_frame(provider),
            ];
            for frame in frames {
                let raw = as_raw_sse_frame(&frame);
                for api in ingress_apis() {
                    let mut parsed_t =
                        StreamTranscoder::new(provider, api, "m1".into(), "id-1".into());
                    let expected = parsed_t.transcode_frame(&frame);

                    let mut raw_t =
                        StreamTranscoder::new(provider, api, "m1".into(), "id-1".into());
                    let mut decode_buffer = Vec::with_capacity(8);
                    let mut out = Vec::with_capacity(8);
                    let _ = raw_t.transcode_raw_frame_into_with_decode_buffer(
                        raw.as_bytes(),
                        &mut decode_buffer,
                        &mut out,
                    );

                    let expected: Vec<String> = expected
                        .iter()
                        .map(|chunk| normalize_generated_call_ids(chunk))
                        .collect();
                    let out: Vec<String> = out
                        .iter()
                        .map(|chunk| normalize_generated_call_ids(chunk))
                        .collect();

                    assert_eq!(
                        out, expected,
                        "raw vs parsed mismatch for provider={provider:?} api={api:?} raw={raw}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_try_decode_raw_frame_rejects_non_sse_payload() {
        let mut t = StreamTranscoder::new(
            ProviderKind::OpenAi,
            IngressApi::OpenAiChat,
            "m1".into(),
            "id-1".into(),
        );
        let mut out = Vec::new();
        assert!(!t.try_decode_upstream_raw_frame_into(b"{\"not\":\"sse\"}", &mut out));
        assert!(out.is_empty());
    }

    #[test]
    fn test_passthrough_openai_to_openai_chat() {
        let t = StreamTranscoder::new(
            ProviderKind::OpenAi,
            IngressApi::OpenAiChat,
            "gpt-4".into(),
            "id-1".into(),
        );
        assert!(t.is_passthrough());
    }

    #[test]
    fn test_passthrough_anthropic_to_anthropic() {
        let t = StreamTranscoder::new(
            ProviderKind::Anthropic,
            IngressApi::Anthropic,
            "claude-3".into(),
            "id-1".into(),
        );
        assert!(t.is_passthrough());
    }

    #[test]
    fn test_passthrough_gemini_to_gemini() {
        let t = StreamTranscoder::new(
            ProviderKind::Gemini,
            IngressApi::Gemini,
            "gemini-pro".into(),
            "id-1".into(),
        );
        assert!(t.is_passthrough());
    }

    #[test]
    fn test_passthrough_gemini_openai_to_openai_chat() {
        let t = StreamTranscoder::new(
            ProviderKind::GeminiOpenAi,
            IngressApi::OpenAiChat,
            "gemini-pro".into(),
            "id-1".into(),
        );
        assert!(t.is_passthrough());
    }

    #[test]
    fn test_passthrough_responses_to_responses() {
        let t = StreamTranscoder::new(
            ProviderKind::OpenAiResponses,
            IngressApi::OpenAiResponses,
            "gpt-4o".into(),
            "id-1".into(),
        );
        assert!(t.is_passthrough());
    }

    #[test]
    fn test_not_passthrough_cross_protocol() {
        let t = StreamTranscoder::new(
            ProviderKind::OpenAi,
            IngressApi::Anthropic,
            "gpt-4".into(),
            "id-1".into(),
        );
        assert!(!t.is_passthrough());
    }

    #[test]
    fn test_decode_openai_text_delta() {
        let mut t = StreamTranscoder::new(
            ProviderKind::OpenAi,
            IngressApi::Anthropic,
            "gpt-4".into(),
            "id-1".into(),
        );
        let frame = SseEvent {
            event: None,
            data: serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": null
                }]
            })
            .to_string(),
            id: None,
            retry: None,
        };
        let events = t.decode_upstream_frame(&frame);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], CanonicalStreamEvent::TextDelta(t) if t == "Hello"));
    }

    #[test]
    fn test_decode_openai_noop_null_finish_reason_stays_fast_path() {
        let mut t = StreamTranscoder::new(
            ProviderKind::OpenAi,
            IngressApi::OpenAiChat,
            "gpt-4".into(),
            "id-1".into(),
        );
        let frame = SseEvent {
            event: None,
            data:
                "{\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null}],\"usage\":null}"
                    .into(),
            id: None,
            retry: None,
        };
        let events = t.decode_upstream_frame(&frame);
        assert!(events.is_empty());
    }

    #[test]
    fn test_decode_openai_done() {
        let mut t = StreamTranscoder::new(
            ProviderKind::OpenAi,
            IngressApi::OpenAiChat,
            "gpt-4".into(),
            "id-1".into(),
        );
        let frame = SseEvent {
            event: None,
            data: "[DONE]".into(),
            id: None,
            retry: None,
        };
        let events = t.decode_upstream_frame(&frame);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], CanonicalStreamEvent::Done));
    }

    #[test]
    fn test_decode_anthropic_text_delta() {
        let mut t = StreamTranscoder::new(
            ProviderKind::Anthropic,
            IngressApi::OpenAiChat,
            "claude-3".into(),
            "id-1".into(),
        );
        let frame = SseEvent {
            event: Some("content_block_delta".into()),
            data: serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "World"}
            })
            .to_string(),
            id: None,
            retry: None,
        };
        let events = t.decode_upstream_frame(&frame);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], CanonicalStreamEvent::TextDelta(t) if t == "World"));
    }

    #[test]
    fn test_decode_anthropic_message_start_fast_path() {
        let mut t = StreamTranscoder::new(
            ProviderKind::Anthropic,
            IngressApi::OpenAiChat,
            "claude-3".into(),
            "id-1".into(),
        );
        let frame = SseEvent {
            event: Some("message_start".into()),
            data: serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": "msg_mock",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3",
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            })
            .to_string(),
            id: None,
            retry: None,
        };
        let events = t.decode_upstream_frame(&frame);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events.first(),
            Some(CanonicalStreamEvent::MessageStart {
                role: CanonicalRole::Assistant
            })
        ));
    }

    #[test]
    fn test_decode_anthropic_error_fast_path() {
        let mut t = StreamTranscoder::new(
            ProviderKind::Anthropic,
            IngressApi::OpenAiChat,
            "claude-3".into(),
            "id-1".into(),
        );
        let frame = SseEvent {
            event: Some("error".into()),
            data: serde_json::json!({
                "type": "error",
                "error": { "type": "api_error", "message": "overloaded" }
            })
            .to_string(),
            id: None,
            retry: None,
        };
        let events = t.decode_upstream_frame(&frame);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events.first(),
            Some(CanonicalStreamEvent::Error { status: 500, message }) if message == "overloaded"
        ));
    }

    #[test]
    fn test_decode_anthropic_tool_use_start_then_stop() {
        let mut t = StreamTranscoder::new(
            ProviderKind::Anthropic,
            IngressApi::OpenAiChat,
            "claude-3".into(),
            "id-1".into(),
        );

        let start_frame = SseEvent {
            event: Some("content_block_start".into()),
            data: serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "lookup",
                    "input": {"q": "x"}
                }
            })
            .to_string(),
            id: None,
            retry: None,
        };
        let start_events = t.decode_upstream_frame(&start_frame);
        assert!(matches!(
            start_events.first(),
            Some(CanonicalStreamEvent::ToolCallStart { index, id, name })
                if *index == 0 && id == "call_1" && name == "lookup"
        ));

        let stop_frame = SseEvent {
            event: Some("content_block_stop".into()),
            data: serde_json::json!({
                "type": "content_block_stop",
                "index": 0
            })
            .to_string(),
            id: None,
            retry: None,
        };
        let stop_events = t.decode_upstream_frame(&stop_frame);
        assert_eq!(stop_events.len(), 1);
        assert!(matches!(
            stop_events.first(),
            Some(CanonicalStreamEvent::ToolCallEnd {
                index,
                call_id: None,
                call_name: None
            }) if *index == 0
        ));
    }

    #[test]
    fn test_decode_gemini_text_delta() {
        let mut t = StreamTranscoder::new(
            ProviderKind::Gemini,
            IngressApi::OpenAiChat,
            "gemini-pro".into(),
            "id-1".into(),
        );
        let frame = SseEvent {
            event: None,
            data: serde_json::json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hola"}]
                    },
                    "index": 0
                }]
            })
            .to_string(),
            id: None,
            retry: None,
        };
        let events = t.decode_upstream_frame(&frame);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], CanonicalStreamEvent::TextDelta(t) if t == "Hola"));
    }

    #[test]
    fn test_decode_gemini_function_call_fast_path() {
        let mut t = StreamTranscoder::new(
            ProviderKind::Gemini,
            IngressApi::OpenAiChat,
            "gemini-pro".into(),
            "id-1".into(),
        );
        let frame = SseEvent {
            event: None,
            data: serde_json::json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{
                            "functionCall": {
                                "name": "lookup",
                                "args": {"q": "x"}
                            }
                        }]
                    },
                    "finishReason": "STOP",
                    "index": 0
                }]
            })
            .to_string(),
            id: None,
            retry: None,
        };
        let events = t.decode_upstream_frame(&frame);
        assert!(matches!(
            events.first(),
            Some(CanonicalStreamEvent::ToolCallStart { name, .. }) if name == "lookup"
        ));
        assert!(events.iter().any(|event| matches!(
            event,
            CanonicalStreamEvent::ToolCallArgsDelta { delta, .. } if delta.contains("\"q\":\"x\"")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            CanonicalStreamEvent::MessageEnd {
                stop_reason: CanonicalStopReason::ToolCalls
            }
        )));
    }

    #[test]
    fn test_decode_responses_text_delta() {
        let mut t = StreamTranscoder::new(
            ProviderKind::OpenAiResponses,
            IngressApi::OpenAiChat,
            "gpt-4o".into(),
            "id-1".into(),
        );
        let frame = SseEvent {
            event: Some("response.output_text.delta".into()),
            data: serde_json::json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Bonjour"
            })
            .to_string(),
            id: None,
            retry: None,
        };
        let events = t.decode_upstream_frame(&frame);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], CanonicalStreamEvent::TextDelta(t) if t == "Bonjour"));
    }

    #[test]
    fn test_decode_responses_uses_event_hint_without_type_field() {
        let mut t = StreamTranscoder::new(
            ProviderKind::OpenAiResponses,
            IngressApi::OpenAiChat,
            "gpt-4o".into(),
            "id-1".into(),
        );
        let frame = SseEvent {
            event: Some("response.output_text.delta".into()),
            data: serde_json::json!({
                "output_index": 0,
                "content_index": 0,
                "delta": "Bonjour"
            })
            .to_string(),
            id: None,
            retry: None,
        };
        let events = t.decode_upstream_frame(&frame);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], CanonicalStreamEvent::TextDelta(t) if t == "Bonjour"));
    }

    #[test]
    fn test_decode_responses_noncanonical_events_fast_skip() {
        let mut t = StreamTranscoder::new(
            ProviderKind::OpenAiResponses,
            IngressApi::OpenAiChat,
            "gpt-4o".into(),
            "id-1".into(),
        );
        let frames = [
            SseEvent {
                event: Some("response.in_progress".into()),
                data: serde_json::json!({
                    "type": "response.in_progress",
                    "response": {
                        "id": "resp_1",
                        "object": "response",
                        "model": "gpt-4o",
                        "output": [],
                        "status": "in_progress"
                    }
                })
                .to_string(),
                id: None,
                retry: None,
            },
            SseEvent {
                event: Some("response.output_text.done".into()),
                data: serde_json::json!({
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "text": "done"
                })
                .to_string(),
                id: None,
                retry: None,
            },
        ];
        for frame in frames {
            let events = t.decode_upstream_frame(&frame);
            assert!(events.is_empty());
        }
    }

    #[test]
    fn test_transcode_openai_to_anthropic() {
        let mut t = StreamTranscoder::new(
            ProviderKind::OpenAi,
            IngressApi::Anthropic,
            "gpt-4".into(),
            "id-1".into(),
        );
        let frame = SseEvent {
            event: None,
            data: serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "test"},
                    "finish_reason": null
                }]
            })
            .to_string(),
            id: None,
            retry: None,
        };
        let results = t.transcode_frame(&frame);
        assert_eq!(results.len(), 1);
        assert!(results[0].contains("event: content_block_delta"));
        assert!(results[0].contains("test"));
    }

    #[test]
    fn test_encode_done_to_openai() {
        let mut t = StreamTranscoder::new(
            ProviderKind::Anthropic,
            IngressApi::OpenAiChat,
            "claude-3".into(),
            "id-1".into(),
        );
        let encoded = t.encode_client_event(&CanonicalStreamEvent::Done);
        assert_eq!(encoded, Some("data: [DONE]\n\n".to_string()));
    }

    #[test]
    fn test_encode_message_end_to_openai() {
        let mut t = StreamTranscoder::new(
            ProviderKind::Anthropic,
            IngressApi::OpenAiChat,
            "claude-3".into(),
            "id-1".into(),
        );
        let encoded = t.encode_client_event(&CanonicalStreamEvent::MessageEnd {
            stop_reason: CanonicalStopReason::EndOfTurn,
        });
        assert!(encoded.is_some());
        let s = encoded.unwrap();
        assert!(s.contains("stop"));
    }

    #[test]
    fn test_transcode_preserves_event_ordering() {
        let mut t = StreamTranscoder::new(
            ProviderKind::OpenAi,
            IngressApi::OpenAiChat,
            "gpt-4".into(),
            "id-1".into(),
        );
        // Message with role + content in same chunk
        let frame = SseEvent {
            event: None,
            data: serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hi"},
                    "finish_reason": null
                }]
            })
            .to_string(),
            id: None,
            retry: None,
        };
        let results = t.transcode_frame(&frame);
        // Should get MessageStart then TextDelta, in order
        assert_eq!(results.len(), 2);
        assert!(results[0].contains("assistant"));
        assert!(results[1].contains("Hi"));
    }

    #[test]
    fn test_decode_invalid_json_returns_empty() {
        let mut t = StreamTranscoder::new(
            ProviderKind::OpenAi,
            IngressApi::OpenAiChat,
            "gpt-4".into(),
            "id-1".into(),
        );
        let frame = SseEvent {
            event: None,
            data: "not valid json".into(),
            id: None,
            retry: None,
        };
        let events = t.decode_upstream_frame(&frame);
        assert!(events.is_empty());
    }
}
