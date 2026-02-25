/// SSE (Server-Sent Events) frame parser, encoder, and stream utilities.
///
/// Handles the low-level parsing of SSE frames from a byte stream,
/// including buffering partial lines and handling field semantics per the
/// [SSE specification](https://html.spec.whatwg.org/multipage/server-sent-events.html).
use super::SseEvent;
use bytes::BytesMut;
use futures_util::Stream;
use memchr::{memchr_iter, memmem};
use smallvec::SmallVec;
use std::sync::LazyLock;

struct PendingEvents {
    events: SmallVec<[SseEvent; 8]>,
    head: usize,
}

impl PendingEvents {
    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        let mut events = SmallVec::new();
        events.reserve(capacity);
        Self { events, head: 0 }
    }

    #[inline]
    fn pop_front(&mut self) -> Option<SseEvent> {
        if self.head >= self.events.len() {
            return None;
        }
        let event = std::mem::take(&mut self.events[self.head]);
        self.head += 1;
        if self.head == self.events.len() {
            self.events.clear();
            self.head = 0;
        }
        Some(event)
    }

    #[inline]
    fn extend_from_vec(&mut self, parsed: &mut Vec<SseEvent>) {
        if parsed.is_empty() {
            return;
        }
        self.events.reserve(parsed.len());
        self.events.extend(parsed.drain(..));
    }
}

// ---------------------------------------------------------------------------
// SseFrame — lightweight frame used by helper functions
// ---------------------------------------------------------------------------

/// A lightweight SSE frame with optional event type and data payload.
#[derive(Debug, Clone)]
pub struct SseFrame {
    pub event: Option<String>,
    pub data: String,
}

// ---------------------------------------------------------------------------
// SseParser — incremental SSE line parser
// ---------------------------------------------------------------------------

/// Incremental SSE line parser.
///
/// Feed it raw text chunks (potentially arriving in arbitrary byte
/// boundaries) and it yields fully-assembled [`SseEvent`] frames.
pub struct SseParser {
    buffer: String,
    read_offset: usize,
    event_type: Option<String>,
    data_buffer: String,
    has_data: bool,
    last_event_id: Option<String>,
}

impl SseParser {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            read_offset: 0,
            event_type: None,
            data_buffer: String::new(),
            has_data: false,
            last_event_id: None,
        }
    }

    /// Feed raw text and return any complete events parsed.
    ///
    /// SSE spec rules:
    /// - Lines starting with `event:` set the event type for the next frame
    /// - Lines starting with `data:` append to the data buffer (strip one
    ///   leading space after the colon per spec)
    /// - Empty lines (`\n\n`) terminate a frame — emit it and reset
    /// - Lines starting with `:` are comments, ignored
    /// - `id:` sets the last event ID
    /// - `retry:` is parsed but not surfaced on the event
    /// - Handle multi-line data (multiple `data:` lines joined with `\n`)
    pub fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        let mut out = Vec::new();
        self.feed_into(chunk, &mut out);
        out
    }

    /// Feed raw text and append complete events into a caller-provided buffer.
    pub fn feed_into(&mut self, chunk: &str, out: &mut Vec<SseEvent>) {
        self.buffer.push_str(chunk);
        let mut processed_up_to = self.read_offset;
        let bytes = self.buffer.as_bytes();
        let scan_start = processed_up_to;
        for rel_pos in memchr_iter(b'\n', &bytes[scan_start..]) {
            let line_end = scan_start + rel_pos;
            let mut line = &self.buffer[processed_up_to..line_end];
            if let Some(stripped) = line.strip_suffix('\r') {
                line = stripped;
            }
            Self::process_line(
                line,
                &mut self.event_type,
                &mut self.data_buffer,
                &mut self.has_data,
                &mut self.last_event_id,
                out,
            );
            processed_up_to = line_end + 1;
        }

        self.read_offset = processed_up_to;
        if self.read_offset == self.buffer.len() {
            self.buffer.clear();
            self.read_offset = 0;
            return;
        }
        let should_compact = self.read_offset > 0
            && (self.read_offset >= self.buffer.len() / 2 || self.read_offset >= 8 * 1024);
        if should_compact {
            self.buffer.drain(..self.read_offset);
            self.read_offset = 0;
        }
    }

    fn process_line(
        line: &str,
        event_type: &mut Option<String>,
        data_buffer: &mut String,
        has_data: &mut bool,
        last_event_id: &mut Option<String>,
        events: &mut Vec<SseEvent>,
    ) {
        if line.is_empty() {
            // Empty line = dispatch event
            if *has_data {
                events.push(SseEvent {
                    event: event_type.take(),
                    data: std::mem::take(data_buffer),
                    id: last_event_id.clone(),
                    retry: None,
                });
                *has_data = false;
            }
            return;
        }

        // Comment line — ignore
        if line.starts_with(':') {
            return;
        }

        if let Some(value) = line.strip_prefix("data:") {
            // Per spec: strip exactly one leading space after "data:"
            let value = value.strip_prefix(' ').unwrap_or(value);
            if *has_data {
                data_buffer.push('\n');
            } else {
                *has_data = true;
            }
            data_buffer.push_str(value);
        } else if let Some(value) = line.strip_prefix("event:") {
            let value = value.strip_prefix(' ').unwrap_or(value);
            *event_type = Some(value.to_string());
        } else if let Some(value) = line.strip_prefix("id:") {
            let value = value.strip_prefix(' ').unwrap_or(value);
            *last_event_id = Some(value.to_string());
        } else if let Some(value) = line.strip_prefix("retry:") {
            if let Ok(_ms) = value.trim().parse::<u64>() {
                // retry field is noted but we don't surface it on events
            }
        }
        // Unknown field names are ignored per spec
    }
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

/// Encode an [`SseFrame`] into SSE wire text.
///
/// - If `event` is `Some`: `event: {event}\ndata: {data}\n\n`
/// - If `event` is `None`: `data: {data}\n\n`
#[must_use]
pub fn encode_sse_frame(frame: &SseFrame) -> String {
    if let Some(event) = frame.event.as_deref() {
        let mut out = String::with_capacity(16 + event.len() + frame.data.len());
        out.push_str("event: ");
        out.push_str(event);
        out.push('\n');
        out.push_str("data: ");
        out.push_str(&frame.data);
        out.push_str("\n\n");
        out
    } else {
        openai_sse_frame(&frame.data)
    }
}

/// Encode an [`SseEvent`] (the richer type from the stream module) into SSE
/// wire text.
#[must_use]
pub fn encode_sse_event(event: &SseEvent) -> String {
    // Fast path for the dominant shape: unnamed, single-line `data` event.
    if event.event.is_none() && event.id.is_none() && !event.data.contains('\n') {
        return openai_sse_frame(&event.data);
    }

    let mut out = String::with_capacity(16 + event.data.len());
    if let Some(ev) = event.event.as_deref() {
        out.push_str("event: ");
        out.push_str(ev);
        out.push('\n');
    }
    for line in event.data.split('\n') {
        out.push_str("data: ");
        out.push_str(line);
        out.push('\n');
    }
    if let Some(id) = event.id.as_deref() {
        out.push_str("id: ");
        out.push_str(id);
        out.push('\n');
    }
    out.push('\n');
    out
}

// ---------------------------------------------------------------------------
// Provider-specific helpers
// ---------------------------------------------------------------------------

/// Check if this is a terminal frame (`OpenAI` `[DONE]` or equivalent).
#[must_use]
pub fn is_done_frame(frame: &SseFrame) -> bool {
    frame.data.trim() == "[DONE]"
}

/// Check if an [`SseEvent`] is a terminal `[DONE]` event.
#[must_use]
pub fn is_done_event(event: &SseEvent) -> bool {
    event.data.trim() == "[DONE]"
}

/// Format a `[DONE]` frame as SSE text.
#[must_use]
pub fn done_frame() -> String {
    DONE_FRAME.to_owned()
}

/// Format an OpenAI-style SSE frame (no event type, just data).
#[must_use]
pub fn openai_sse_frame(json: &str) -> String {
    let mut out = String::with_capacity(10 + json.len());
    out.push_str("data: ");
    out.push_str(json);
    out.push_str("\n\n");
    out
}

/// Format an Anthropic-style SSE frame (with named event type).
#[must_use]
pub fn anthropic_sse_frame(event_type: &str, json: &str) -> String {
    let mut out = String::with_capacity(18 + event_type.len() + json.len());
    out.push_str("event: ");
    out.push_str(event_type);
    out.push('\n');
    out.push_str("data: ");
    out.push_str(json);
    out.push_str("\n\n");
    out
}

/// Format a Gemini-style SSE frame (same shape as `OpenAI` — just data).
#[must_use]
pub fn gemini_sse_frame(json: &str) -> String {
    openai_sse_frame(json)
}

// ---------------------------------------------------------------------------
// Stream utility
// ---------------------------------------------------------------------------

/// Split a byte stream into SSE events using [`SseParser`].
///
/// Bytes arriving from an HTTP response body are decoded as UTF-8,
/// fed into the parser, and complete [`SseEvent`] frames are yielded.
///
/// This is the primary entry point for converting an HTTP response body
/// stream into a stream of parsed SSE events.
pub fn sse_frame_stream<S, E>(byte_stream: S) -> impl Stream<Item = SseEvent> + Send
where
    S: Stream<Item = Result<bytes::Bytes, E>> + Send + 'static,
    E: std::fmt::Debug + Send + 'static,
{
    use futures_util::StreamExt;

    futures_util::stream::unfold(
        (
            Box::pin(byte_stream),
            SseParser::new(),
            Vec::<u8>::new(),
            Vec::<SseEvent>::with_capacity(8),
            PendingEvents::with_capacity(8),
        ),
        |(mut stream, mut parser, mut remainder, mut parsed, mut pending)| async move {
            loop {
                if let Some(event) = pending.pop_front() {
                    return Some((event, (stream, parser, remainder, parsed, pending)));
                }

                let chunk = stream.as_mut().next().await?;
                if let Ok(bytes) = chunk {
                    if remainder.is_empty() {
                        match std::str::from_utf8(&bytes) {
                            Ok(text) => parser.feed_into(text, &mut parsed),
                            Err(e) => {
                                let valid_up_to = e.valid_up_to();
                                // Safety: valid_up_to is guaranteed to be a valid UTF-8 boundary.
                                let text =
                                    unsafe { std::str::from_utf8_unchecked(&bytes[..valid_up_to]) };
                                parser.feed_into(text, &mut parsed);
                                remainder.extend_from_slice(&bytes[valid_up_to..]);
                            }
                        }
                    } else {
                        remainder.extend_from_slice(&bytes);
                        match std::str::from_utf8(remainder.as_slice()) {
                            Ok(text) => {
                                parser.feed_into(text, &mut parsed);
                                remainder.clear();
                            }
                            Err(e) => {
                                let valid_up_to = e.valid_up_to();
                                // Safety: valid_up_to is guaranteed to be a valid UTF-8 boundary.
                                let text = unsafe {
                                    std::str::from_utf8_unchecked(&remainder[..valid_up_to])
                                };
                                parser.feed_into(text, &mut parsed);
                                if valid_up_to > 0 {
                                    if valid_up_to == remainder.len() {
                                        remainder.clear();
                                    } else {
                                        let remain_len = remainder.len() - valid_up_to;
                                        remainder.copy_within(valid_up_to.., 0);
                                        remainder.truncate(remain_len);
                                    }
                                }
                            }
                        }
                    }
                    if !parsed.is_empty() {
                        pending.extend_from_vec(&mut parsed);
                        if let Some(first) = pending.pop_front() {
                            return Some((first, (stream, parser, remainder, parsed, pending)));
                        }
                    }
                }
            }
        },
    )
}

#[inline]
fn find_sse_frame_terminator(buffer: &[u8]) -> Option<(usize, usize)> {
    find_sse_frame_terminator_from(buffer, 0)
}

#[inline]
fn find_sse_frame_terminator_from(buffer: &[u8], scan_from: usize) -> Option<(usize, usize)> {
    static LF_LF_FINDER: LazyLock<memmem::Finder<'static>> =
        LazyLock::new(|| memmem::Finder::new(b"\n\n"));
    static CRLF_CRLF_FINDER: LazyLock<memmem::Finder<'static>> =
        LazyLock::new(|| memmem::Finder::new(b"\r\n\r\n"));

    let scan_from = scan_from.min(buffer.len());
    let haystack = &buffer[scan_from..];
    let lf_lf_pos = LF_LF_FINDER.find(haystack).map(|rel| scan_from + rel);
    let crlf_crlf_pos = CRLF_CRLF_FINDER.find(haystack).map(|rel| scan_from + rel);

    match (lf_lf_pos, crlf_crlf_pos) {
        (Some(lf_pos), Some(crlf_pos)) => {
            if lf_pos <= crlf_pos {
                Some((lf_pos, 2))
            } else {
                Some((crlf_pos, 4))
            }
        }
        (Some(lf_pos), None) => Some((lf_pos, 2)),
        (None, Some(crlf_pos)) => Some((crlf_pos, 4)),
        (None, None) => None,
    }
}

/// Split a byte stream into raw SSE frame text chunks.
///
/// Each yielded item contains one complete SSE frame, including the trailing
/// blank-line separator. This is useful for fast passthrough paths that need
/// frame boundaries but can defer full SSE field parsing.
pub fn sse_raw_frame_stream<S, E>(byte_stream: S) -> impl Stream<Item = bytes::Bytes> + Send
where
    S: Stream<Item = Result<bytes::Bytes, E>> + Send + 'static,
    E: std::fmt::Debug + Send + 'static,
{
    use futures_util::StreamExt;

    futures_util::stream::unfold(
        (Box::pin(byte_stream), BytesMut::with_capacity(4096), 0usize),
        |(mut stream, mut buffer, mut scan_from)| async move {
            loop {
                if let Some((frame_start, frame_len)) =
                    find_sse_frame_terminator_from(&buffer, scan_from)
                {
                    let split = frame_start + frame_len;
                    let frame = buffer.split_to(split).freeze();
                    scan_from = 0;
                    return Some((frame, (stream, buffer, scan_from)));
                }

                // On no-match, keep only a tiny overlap so next scan can catch
                // frame terminators spanning chunk boundaries.
                scan_from = buffer.len().saturating_sub(3);
                if let Some(Ok(bytes)) = stream.as_mut().next().await {
                    if buffer.is_empty() {
                        let chunk = bytes.as_ref();
                        if let Some((frame_start, frame_len)) = find_sse_frame_terminator(chunk) {
                            let split = frame_start + frame_len;
                            if split == chunk.len() {
                                scan_from = 0;
                                return Some((bytes, (stream, buffer, scan_from)));
                            }

                            let frame = bytes.slice(..split);
                            buffer.extend_from_slice(&chunk[split..]);
                            scan_from = buffer.len().saturating_sub(3);
                            return Some((frame, (stream, buffer, scan_from)));
                        }
                    }

                    buffer.extend_from_slice(bytes.as_ref());
                } else {
                    if !buffer.is_empty() {
                        let frame = buffer.split().freeze();
                        scan_from = 0;
                        return Some((frame, (stream, buffer, scan_from)));
                    }
                    return None;
                }
            }
        },
    )
}

const DONE_FRAME: &str = "data: [DONE]\n\n";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures_util::StreamExt;

    // -- SseParser tests --

    #[test]
    fn test_parse_simple_data_frame() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hello world\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello world");
        assert!(events[0].event.is_none());
    }

    #[test]
    fn test_parse_named_event() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: message_start\ndata: {\"type\":\"message_start\"}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("message_start"));
        assert_eq!(events[0].data, "{\"type\":\"message_start\"}");
    }

    #[test]
    fn test_parse_multiline_data() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: line1\ndata: line2\ndata: line3\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2\nline3");
    }

    #[test]
    fn test_parse_multiple_frames() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: first\n\ndata: second\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "first");
        assert_eq!(events[1].data, "second");
    }

    #[test]
    fn test_parse_ignores_comments() {
        let mut parser = SseParser::new();
        let events = parser.feed(": this is a comment\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_parse_done_frame() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: [DONE]\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "[DONE]");
        assert!(is_done_event(&events[0]));
    }

    #[test]
    fn test_parse_incremental_chunks() {
        let mut parser = SseParser::new();

        // First chunk — partial line
        let events = parser.feed("data: hel");
        assert!(events.is_empty());

        // Second chunk — completes the line but no blank line yet
        let events = parser.feed("lo\n");
        assert!(events.is_empty());

        // Third chunk — blank line terminates the frame
        let events = parser.feed("\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_parse_data_no_space_after_colon() {
        let mut parser = SseParser::new();
        let events = parser.feed("data:nospace\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "nospace");
    }

    #[test]
    fn test_parse_empty_data() {
        let mut parser = SseParser::new();
        let events = parser.feed("data:\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    #[test]
    fn test_parse_id_field() {
        let mut parser = SseParser::new();
        let events = parser.feed("id: 42\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id.as_deref(), Some("42"));
    }

    #[test]
    fn test_parse_crlf_line_endings() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hello\r\n\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_parse_empty_lines_without_data_dont_emit() {
        let mut parser = SseParser::new();
        let events = parser.feed("\n\n\n");
        assert!(events.is_empty());
    }

    #[test]
    fn test_parse_anthropic_sequence() {
        let mut parser = SseParser::new();
        let input = "\
event: message_start\n\
data: {\"type\":\"message_start\"}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"Hi\"}}\n\
\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\
\n";
        let events = parser.feed(input);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event.as_deref(), Some("message_start"));
        assert_eq!(events[1].event.as_deref(), Some("content_block_delta"));
        assert_eq!(events[2].event.as_deref(), Some("message_stop"));
    }

    #[test]
    fn test_parse_openai_sequence() {
        let mut parser = SseParser::new();
        let input = "\
data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\
\n\
data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\
\n\
data: [DONE]\n\
\n";
        let events = parser.feed(input);
        assert_eq!(events.len(), 3);
        assert!(events[0].event.is_none());
        assert!(events[1].event.is_none());
        assert!(is_done_event(&events[2]));
    }

    #[test]
    fn test_feed_into_appends_without_clearing_output() {
        let mut parser = SseParser::new();
        let mut out = vec![SseEvent {
            event: None,
            data: "seed".to_string(),
            id: None,
            retry: None,
        }];
        parser.feed_into("data: a\n\n", &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].data, "seed");
        assert_eq!(out[1].data, "a");
    }

    // -- Encoding tests --

    #[test]
    fn test_encode_frame_without_event() {
        let frame = SseFrame {
            event: None,
            data: "{\"key\":\"value\"}".to_string(),
        };
        assert_eq!(encode_sse_frame(&frame), "data: {\"key\":\"value\"}\n\n");
    }

    #[test]
    fn test_encode_frame_with_event() {
        let frame = SseFrame {
            event: Some("message_start".to_string()),
            data: "{\"type\":\"message_start\"}".to_string(),
        };
        assert_eq!(
            encode_sse_frame(&frame),
            "event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
        );
    }

    #[test]
    fn test_encode_sse_event_multiline() {
        let event = SseEvent {
            event: None,
            data: "line1\nline2".to_string(),
            id: None,
            retry: None,
        };
        assert_eq!(encode_sse_event(&event), "data: line1\ndata: line2\n\n");
    }

    #[test]
    fn test_encode_sse_event_with_id() {
        let event = SseEvent {
            event: Some("ping".to_string()),
            data: "{}".to_string(),
            id: Some("42".to_string()),
            retry: None,
        };
        assert_eq!(
            encode_sse_event(&event),
            "event: ping\ndata: {}\nid: 42\n\n"
        );
    }

    // -- Helper function tests --

    #[test]
    fn test_is_done_frame() {
        let frame = SseFrame {
            event: None,
            data: "[DONE]".to_string(),
        };
        assert!(is_done_frame(&frame));

        let frame = SseFrame {
            event: None,
            data: " [DONE] ".to_string(),
        };
        assert!(is_done_frame(&frame));

        let frame = SseFrame {
            event: None,
            data: "{\"content\":\"hello\"}".to_string(),
        };
        assert!(!is_done_frame(&frame));
    }

    #[test]
    fn test_done_frame_string() {
        assert_eq!(done_frame(), "data: [DONE]\n\n");
    }

    #[test]
    fn test_openai_sse_frame_helper() {
        let json = r#"{"id":"chatcmpl-1"}"#;
        assert_eq!(openai_sse_frame(json), "data: {\"id\":\"chatcmpl-1\"}\n\n");
    }

    #[test]
    fn test_anthropic_sse_frame_helper() {
        let json = r#"{"type":"message_start"}"#;
        assert_eq!(
            anthropic_sse_frame("message_start", json),
            "event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
        );
    }

    #[test]
    fn test_gemini_sse_frame_helper() {
        let json = r#"{"candidates":[]}"#;
        assert_eq!(gemini_sse_frame(json), "data: {\"candidates\":[]}\n\n");
    }

    #[tokio::test]
    async fn test_sse_raw_frame_stream_single_frame_chunk() {
        let source = futures_util::stream::iter(vec![Ok::<Bytes, std::convert::Infallible>(
            Bytes::from_static(b"data: hello\n\n"),
        )]);
        let frames: Vec<Bytes> = sse_raw_frame_stream(source).collect().await;
        assert_eq!(frames, vec![Bytes::from_static(b"data: hello\n\n")]);
    }

    #[tokio::test]
    async fn test_sse_raw_frame_stream_multiple_frames_same_chunk_order() {
        let source = futures_util::stream::iter(vec![Ok::<Bytes, std::convert::Infallible>(
            Bytes::from_static(b"data: first\n\ndata: second\n\n"),
        )]);
        let frames: Vec<Bytes> = sse_raw_frame_stream(source).collect().await;
        assert_eq!(
            frames,
            vec![
                Bytes::from_static(b"data: first\n\n"),
                Bytes::from_static(b"data: second\n\n")
            ]
        );
    }

    #[tokio::test]
    async fn test_sse_raw_frame_stream_crlf_and_partial_tail() {
        let source = futures_util::stream::iter(vec![
            Ok::<Bytes, std::convert::Infallible>(Bytes::from_static(b"data: a\r\n\r\n")),
            Ok::<Bytes, std::convert::Infallible>(Bytes::from_static(b"data: tail")),
        ]);
        let frames: Vec<Bytes> = sse_raw_frame_stream(source).collect().await;
        assert_eq!(
            frames,
            vec![
                Bytes::from_static(b"data: a\r\n\r\n"),
                Bytes::from_static(b"data: tail")
            ]
        );
    }

    #[tokio::test]
    async fn test_sse_raw_frame_stream_split_terminator_across_chunks() {
        let source = futures_util::stream::iter(vec![
            Ok::<Bytes, std::convert::Infallible>(Bytes::from_static(b"data: a\n")),
            Ok::<Bytes, std::convert::Infallible>(Bytes::from_static(b"\ndata: b\n\n")),
        ]);
        let frames: Vec<Bytes> = sse_raw_frame_stream(source).collect().await;
        assert_eq!(
            frames,
            vec![
                Bytes::from_static(b"data: a\n\n"),
                Bytes::from_static(b"data: b\n\n")
            ]
        );
    }
}
