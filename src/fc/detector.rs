use std::sync::LazyLock;

use memchr::{memchr, memmem};

// Streaming function-call trigger detector.
//
// Ports the Python `StreamingFunctionCallDetector` to Rust. The detector is a
// small state machine that scans text deltas arriving from an upstream LLM and
// decides whether the model is emitting a tool-call block.
//
// Key invariants:
// - S3-I2: Trigger detection ignores occurrences inside reasoning blocks, such
//   as `<think>…</think>`, `<thinking>…</thinking>`,
//   `<reasoning>…</reasoning>`, and `<analysis>…</analysis>`.
// - S9-I2: Internal buffer is capped at 512 KB; overflow falls back to passthrough.
// - Detection works correctly across arbitrary chunk boundaries.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// State of the detector state-machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectorState {
    /// Scanning incoming text for the trigger signal.
    Detecting,
    /// Trigger was found – buffering subsequent text for XML parsing.
    ToolParsing,
    /// Terminal state – a complete `</function_calls>` closing tag was received.
    Completed,
}

/// Action returned by [`StreamingFcDetector::feed`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectorAction {
    /// Forward this text to the client unchanged.
    PassThrough(String),
    /// Text has been buffered internally; do not forward anything yet.
    Buffer,
    /// The trigger signal was detected.
    ///
    /// `text_before` is content that should still be forwarded to the client.
    TriggerFound { text_before: String },
    /// The internal buffer exceeded the maximum size – flush everything as
    /// plain text so the client is not starved.
    BufferOverflow(String),
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Streaming function-call trigger detector.
pub struct StreamingFcDetector {
    trigger_signal: &'static str,
    trigger_starts_with_lt: bool,
    buffer: String,
    state: DetectorState,
    /// Nesting depth of reasoning tags currently open.
    think_depth: usize,
    /// Maximum buffer size in bytes (default 512 KB).
    max_buffer_size: usize,
    /// Whether `<function_calls>` opening tag has appeared after trigger.
    saw_function_calls_open: bool,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const THINKING_OPEN: &str = "<thinking>";
const THINKING_CLOSE: &str = "</thinking>";
const REASONING_OPEN: &str = "<reasoning>";
const REASONING_CLOSE: &str = "</reasoning>";
const ANALYSIS_OPEN: &str = "<analysis>";
const ANALYSIS_CLOSE: &str = "</analysis>";
const FC_OPEN: &str = "<function_calls>";
const FC_CLOSE: &str = "</function_calls>";
static FC_OPEN_FINDER: LazyLock<memmem::Finder<'static>> =
    LazyLock::new(|| memmem::Finder::new(FC_OPEN.as_bytes()));
static FC_CLOSE_FINDER: LazyLock<memmem::Finder<'static>> =
    LazyLock::new(|| memmem::Finder::new(FC_CLOSE.as_bytes()));
const fn max_usize(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}
const MAX_REASONING_TAG_LEN: usize = max_usize(
    max_usize(THINK_CLOSE.len(), THINKING_CLOSE.len()),
    max_usize(REASONING_CLOSE.len(), ANALYSIS_CLOSE.len()),
);
const DEFAULT_MAX_BUFFER: usize = 512 * 1024;
const MAX_TRIGGER_PREAMBLE_WITHOUT_FC_OPEN: usize = 4096;

#[inline]
fn contains_fc_open_since(buffer: &str, previous_len: usize) -> bool {
    let scan_start = previous_len.saturating_sub(FC_OPEN.len().saturating_sub(1));
    FC_OPEN_FINDER
        .find(buffer.as_bytes().get(scan_start..).unwrap_or_default())
        .is_some()
}

#[inline]
fn reasoning_open_tag_len_at(bytes: &[u8]) -> Option<usize> {
    if bytes.first().copied()? != b'<' {
        return None;
    }
    match bytes.get(1).copied() {
        Some(b't') => {
            if bytes.starts_with(THINK_OPEN.as_bytes()) {
                Some(THINK_OPEN.len())
            } else if bytes.starts_with(THINKING_OPEN.as_bytes()) {
                Some(THINKING_OPEN.len())
            } else {
                None
            }
        }
        Some(b'r') => bytes
            .starts_with(REASONING_OPEN.as_bytes())
            .then_some(REASONING_OPEN.len()),
        Some(b'a') => bytes
            .starts_with(ANALYSIS_OPEN.as_bytes())
            .then_some(ANALYSIS_OPEN.len()),
        _ => None,
    }
}

#[inline]
fn reasoning_close_tag_len_at(bytes: &[u8]) -> Option<usize> {
    if bytes.first().copied()? != b'<' || bytes.get(1).copied()? != b'/' {
        return None;
    }
    match bytes.get(2).copied() {
        Some(b't') => {
            if bytes.starts_with(THINK_CLOSE.as_bytes()) {
                Some(THINK_CLOSE.len())
            } else if bytes.starts_with(THINKING_CLOSE.as_bytes()) {
                Some(THINKING_CLOSE.len())
            } else {
                None
            }
        }
        Some(b'r') => bytes
            .starts_with(REASONING_CLOSE.as_bytes())
            .then_some(REASONING_CLOSE.len()),
        Some(b'a') => bytes
            .starts_with(ANALYSIS_CLOSE.as_bytes())
            .then_some(ANALYSIS_CLOSE.len()),
        _ => None,
    }
}

#[inline]
fn next_utf8_char_boundary(bytes: &[u8], mut i: usize) -> usize {
    i = i.saturating_add(1);
    while i < bytes.len() && (bytes[i] & 0b1100_0000) == 0b1000_0000 {
        i += 1;
    }
    i
}

impl StreamingFcDetector {
    /// Create a new detector that looks for `trigger_signal` in the text
    /// stream.
    #[must_use]
    pub fn new(trigger_signal: &'static str) -> Self {
        Self {
            trigger_signal,
            trigger_starts_with_lt: trigger_signal.as_bytes().first() == Some(&b'<'),
            buffer: String::new(),
            state: DetectorState::Detecting,
            think_depth: 0,
            max_buffer_size: DEFAULT_MAX_BUFFER,
            saw_function_calls_open: false,
        }
    }

    /// Return a reference to the current state.
    #[must_use]
    pub fn state(&self) -> &DetectorState {
        &self.state
    }

    #[must_use]
    pub fn trigger_signal(&self) -> &'static str {
        self.trigger_signal
    }

    // -- public API ---------------------------------------------------------

    /// Feed a new text delta into the detector and obtain the resulting action.
    pub fn feed(&mut self, text: &str) -> DetectorAction {
        if text.is_empty() {
            return DetectorAction::Buffer;
        }

        match self.state {
            DetectorState::Detecting => self.feed_detecting(text),
            DetectorState::ToolParsing => self.feed_tool_parsing(text),
            DetectorState::Completed => {
                // After completion, pass everything through.
                DetectorAction::PassThrough(text.to_string())
            }
        }
    }

    /// Feed an owned text delta into the detector.
    ///
    /// This enables a zero-copy passthrough fast path when the detector can
    /// prove the chunk cannot contain a trigger.
    pub fn feed_owned(&mut self, text: String) -> DetectorAction {
        if text.is_empty() {
            return DetectorAction::Buffer;
        }

        if matches!(self.state, DetectorState::Detecting)
            && self.trigger_starts_with_lt
            && self.buffer.is_empty()
            && self.think_depth == 0
            && text.len() <= self.max_buffer_size
            && !text.as_bytes().contains(&b'<')
        {
            return DetectorAction::PassThrough(text);
        }

        match self.state {
            DetectorState::Detecting => self.feed_detecting(&text),
            DetectorState::ToolParsing => self.feed_tool_parsing(&text),
            DetectorState::Completed => DetectorAction::PassThrough(text),
        }
    }

    /// Call when the stream ends. Returns any remaining buffered content.
    pub fn finalize(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        let remaining = std::mem::take(&mut self.buffer);
        Some(remaining)
    }

    // -- Detecting state ----------------------------------------------------

    fn feed_detecting(&mut self, text: &str) -> DetectorAction {
        let sig_len = self.trigger_signal.len();
        let min_keep = std::cmp::max(sig_len.saturating_sub(1), MAX_REASONING_TAG_LEN);

        // Fast path: when the trigger starts with '<' and this chunk has no '<',
        // trigger detection cannot start in this chunk.
        if self.trigger_starts_with_lt
            && self.buffer.is_empty()
            && self.think_depth == 0
            && text.len() <= self.max_buffer_size
            && !text.as_bytes().contains(&b'<')
        {
            return DetectorAction::PassThrough(text.to_string());
        }

        self.buffer.push_str(text);

        // Overflow guard.
        if self.buffer.len() > self.max_buffer_size {
            let flushed = std::mem::take(&mut self.buffer);
            self.think_depth = 0;
            self.saw_function_calls_open = false;
            return DetectorAction::BufferOverflow(flushed);
        }

        // Walk the buffer while tracking reasoning depth and looking for the
        // trigger signal outside of reasoning blocks.
        let mut i: usize = 0;
        let mut trigger_at: Option<usize> = None;
        {
            let buffer_bytes = self.buffer.as_bytes();
            let trigger_bytes = self.trigger_signal.as_bytes();
            if self.trigger_starts_with_lt {
                let scan_limit = buffer_bytes.len().saturating_sub(min_keep);
                while i < scan_limit {
                    let Some(rel_lt) = memchr(b'<', &buffer_bytes[i..scan_limit]) else {
                        i = scan_limit;
                        break;
                    };
                    i += rel_lt;

                    if let Some(open_len) = reasoning_open_tag_len_at(&buffer_bytes[i..]) {
                        self.think_depth += 1;
                        i += open_len;
                        continue;
                    }

                    if let Some(close_len) = reasoning_close_tag_len_at(&buffer_bytes[i..]) {
                        self.think_depth = self.think_depth.saturating_sub(1);
                        i += close_len;
                        continue;
                    }

                    if self.think_depth == 0
                        && sig_len > 0
                        && i + sig_len <= buffer_bytes.len()
                        && buffer_bytes[i..].starts_with(trigger_bytes)
                    {
                        trigger_at = Some(i);
                        break;
                    }

                    i += 1;
                }
            } else {
                while i < buffer_bytes.len() {
                    if let Some(open_len) = reasoning_open_tag_len_at(&buffer_bytes[i..]) {
                        self.think_depth += 1;
                        i += open_len;
                        continue;
                    }

                    if let Some(close_len) = reasoning_close_tag_len_at(&buffer_bytes[i..]) {
                        self.think_depth = self.think_depth.saturating_sub(1);
                        i += close_len;
                        continue;
                    }

                    if self.think_depth == 0
                        && sig_len > 0
                        && i + sig_len <= buffer_bytes.len()
                        && buffer_bytes[i..].starts_with(trigger_bytes)
                    {
                        trigger_at = Some(i);
                        break;
                    }

                    let remaining = buffer_bytes.len() - i;
                    if remaining <= min_keep {
                        break;
                    }

                    // Advance to the next UTF-8 boundary so later split_off(i) stays valid.
                    i = next_utf8_char_boundary(buffer_bytes, i);
                }
            }
        }

        if let Some(trigger_index) = trigger_at {
            self.state = DetectorState::ToolParsing;
            if trigger_index == 0 {
                self.saw_function_calls_open = self.buffer.contains(FC_OPEN);
                return DetectorAction::TriggerFound {
                    text_before: String::new(),
                };
            }
            let tail = self.buffer.split_off(trigger_index);
            let text_before = std::mem::replace(&mut self.buffer, tail);
            self.saw_function_calls_open = self.buffer.contains(FC_OPEN);
            return DetectorAction::TriggerFound { text_before };
        }

        if i == 0 {
            DetectorAction::Buffer
        } else {
            let tail = self.buffer.split_off(i);
            let pass_through = std::mem::replace(&mut self.buffer, tail);
            DetectorAction::PassThrough(pass_through)
        }
    }

    // -- ToolParsing state --------------------------------------------------

    fn feed_tool_parsing(&mut self, text: &str) -> DetectorAction {
        let previous_len = self.buffer.len();
        let search_from = previous_len.saturating_sub(FC_CLOSE.len().saturating_sub(1));
        self.buffer.push_str(text);

        if !self.saw_function_calls_open {
            self.saw_function_calls_open = contains_fc_open_since(&self.buffer, previous_len);
            if !self.saw_function_calls_open
                && self.buffer.len() > MAX_TRIGGER_PREAMBLE_WITHOUT_FC_OPEN
            {
                let flushed = std::mem::take(&mut self.buffer);
                self.state = DetectorState::Completed;
                return DetectorAction::BufferOverflow(flushed);
            }
        }

        // Overflow guard.
        if self.buffer.len() > self.max_buffer_size {
            let flushed = std::mem::take(&mut self.buffer);
            self.state = DetectorState::Completed;
            self.saw_function_calls_open = false;
            return DetectorAction::BufferOverflow(flushed);
        }

        // Check for the closing </function_calls> tag.
        if FC_CLOSE_FINDER
            .find(
                self.buffer
                    .as_bytes()
                    .get(search_from..)
                    .unwrap_or_default(),
            )
            .is_some()
        {
            self.state = DetectorState::Completed;
        }

        // In ToolParsing we always buffer – the caller retrieves the XML via
        // `finalize()` or inspects state to know when parsing is done.
        DetectorAction::Buffer
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TRIGGER: &str = "<Function_AB12_Start/>";

    fn new_detector() -> StreamingFcDetector {
        StreamingFcDetector::new(TRIGGER)
    }

    #[test]
    fn plain_text_passes_through() {
        let mut d = new_detector();
        let action = d.feed("Hello, world! This is a normal response.");
        match action {
            DetectorAction::PassThrough(t) => {
                assert!(!t.is_empty(), "should yield some text");
            }
            DetectorAction::Buffer => { /* acceptable if text is shorter than trigger */ }
            other => panic!("unexpected action: {other:?}"),
        }
    }

    #[test]
    fn trigger_detected_in_single_chunk() {
        let mut d = new_detector();
        let input = format!("Some preamble text{TRIGGER}<function_calls><function_call>");
        let action = d.feed(&input);
        match action {
            DetectorAction::TriggerFound { text_before } => {
                assert_eq!(text_before, "Some preamble text");
            }
            other => panic!("expected TriggerFound, got {other:?}"),
        }
    }

    #[test]
    fn trigger_split_across_chunks() {
        let mut d = new_detector();
        let (first, second) = TRIGGER.split_at(TRIGGER.len() / 2);
        let input1 = format!("Hello {first}");
        let input2 = format!("{second}<function_calls>");

        let a1 = d.feed(&input1);
        // Should pass through safe text or buffer.
        match &a1 {
            DetectorAction::PassThrough(t) => assert!(t.starts_with("Hello")),
            DetectorAction::Buffer => {}
            other => panic!("unexpected: {other:?}"),
        }

        let a2 = d.feed(&input2);
        match a2 {
            DetectorAction::TriggerFound { text_before } => {
                // text_before may include "Hello " portion depending on flush
                let _ = text_before;
            }
            other => panic!("expected TriggerFound, got {other:?}"),
        }
    }

    #[test]
    fn trigger_inside_think_block_is_ignored() {
        let mut d = new_detector();
        let input = format!("<think>Reasoning about {TRIGGER}</think>");
        let action = d.feed(&input);
        // The trigger is inside <think>, so it must not be detected.
        match action {
            DetectorAction::PassThrough(_) | DetectorAction::Buffer => {}
            DetectorAction::TriggerFound { text_before: _ } => {
                panic!("trigger inside <think> must be ignored");
            }
            DetectorAction::BufferOverflow(_) => panic!("unexpected overflow"),
        }
        assert_eq!(*d.state(), DetectorState::Detecting);
    }

    #[test]
    fn trigger_inside_reasoning_block_is_ignored() {
        let mut d = new_detector();
        let input = format!("<reasoning>Reasoning about {TRIGGER}</reasoning>");
        let action = d.feed(&input);
        match action {
            DetectorAction::PassThrough(_) | DetectorAction::Buffer => {}
            DetectorAction::TriggerFound { text_before: _ } => {
                panic!("trigger inside <reasoning> must be ignored");
            }
            DetectorAction::BufferOverflow(_) => panic!("unexpected overflow"),
        }
        assert_eq!(*d.state(), DetectorState::Detecting);
    }

    #[test]
    fn trigger_inside_thinking_block_is_ignored() {
        let mut d = new_detector();
        let input = format!("<thinking>Reasoning about {TRIGGER}</thinking>");
        let action = d.feed(&input);
        match action {
            DetectorAction::PassThrough(_) | DetectorAction::Buffer => {}
            DetectorAction::TriggerFound { text_before: _ } => {
                panic!("trigger inside <thinking> must be ignored");
            }
            DetectorAction::BufferOverflow(_) => panic!("unexpected overflow"),
        }
        assert_eq!(*d.state(), DetectorState::Detecting);
    }

    #[test]
    fn trigger_inside_analysis_block_is_ignored() {
        let mut d = new_detector();
        let input = format!("<analysis>Reasoning about {TRIGGER}</analysis>");
        let action = d.feed(&input);
        match action {
            DetectorAction::PassThrough(_) | DetectorAction::Buffer => {}
            DetectorAction::TriggerFound { text_before: _ } => {
                panic!("trigger inside <analysis> must be ignored");
            }
            DetectorAction::BufferOverflow(_) => panic!("unexpected overflow"),
        }
        assert_eq!(*d.state(), DetectorState::Detecting);
    }

    #[test]
    fn nested_think_blocks() {
        let mut d = new_detector();
        let input =
            format!("<think>Outer <think>Inner {TRIGGER}</think></think>After think {TRIGGER}more");
        let action = d.feed(&input);
        match action {
            DetectorAction::TriggerFound { text_before } => {
                // The trigger after the think block should be found.
                assert!(text_before.contains("After think"));
            }
            other => panic!("expected TriggerFound after think blocks, got {other:?}"),
        }
    }

    #[test]
    fn tool_parsing_detects_closing_tag() {
        let mut d = new_detector();
        let input1 = format!("{TRIGGER}<function_calls><function_call>");
        let _ = d.feed(&input1);
        assert_eq!(*d.state(), DetectorState::ToolParsing);

        let action = d.feed("</function_call></function_calls>");
        assert_eq!(action, DetectorAction::Buffer);
        assert_eq!(*d.state(), DetectorState::Completed);
    }

    #[test]
    fn tool_parsing_without_function_calls_open_falls_back_early() {
        let mut d = new_detector();
        let first = format!(
            "{TRIGGER}{}",
            "x".repeat(MAX_TRIGGER_PREAMBLE_WITHOUT_FC_OPEN + 32)
        );
        let trigger_action = d.feed(&first);
        assert!(matches!(
            trigger_action,
            DetectorAction::TriggerFound { .. }
        ));
        assert_eq!(*d.state(), DetectorState::ToolParsing);

        let action = d.feed("tail");
        match action {
            DetectorAction::BufferOverflow(flushed) => {
                assert!(flushed.contains(TRIGGER));
            }
            other => panic!("expected BufferOverflow, got {other:?}"),
        }
        assert_eq!(*d.state(), DetectorState::Completed);
    }

    #[test]
    fn tool_parsing_with_function_calls_open_allows_large_payload() {
        let mut d = new_detector();
        let first = format!("{TRIGGER}<function_calls><function_call><tool>x</tool><args_json>");
        let trigger_action = d.feed(&first);
        assert!(matches!(
            trigger_action,
            DetectorAction::TriggerFound { .. }
        ));
        assert_eq!(*d.state(), DetectorState::ToolParsing);

        let action = d.feed(&"x".repeat(MAX_TRIGGER_PREAMBLE_WITHOUT_FC_OPEN + 64));
        assert_eq!(action, DetectorAction::Buffer);
        assert_eq!(*d.state(), DetectorState::ToolParsing);
    }

    #[test]
    fn buffer_overflow_in_detecting() {
        let mut d = StreamingFcDetector::new(TRIGGER);
        d.max_buffer_size = 100;

        let big = "A".repeat(200);
        let action = d.feed(&big);
        match action {
            DetectorAction::BufferOverflow(s) => {
                assert!(!s.is_empty());
            }
            other => panic!("expected BufferOverflow, got {other:?}"),
        }
    }

    #[test]
    fn buffer_overflow_in_tool_parsing() {
        let mut d = StreamingFcDetector::new(TRIGGER);
        d.max_buffer_size = 100;

        let input1 = format!("{TRIGGER}<function_calls>");
        let _ = d.feed(&input1);
        assert_eq!(*d.state(), DetectorState::ToolParsing);

        let big = "X".repeat(200);
        let action = d.feed(&big);
        match action {
            DetectorAction::BufferOverflow(_) => {}
            other => panic!("expected BufferOverflow, got {other:?}"),
        }
        assert_eq!(*d.state(), DetectorState::Completed);
    }

    #[test]
    fn finalize_returns_remaining_buffer() {
        let mut d = new_detector();
        let input = format!("{TRIGGER}<function_calls><fc>");
        let _ = d.feed(&input);
        let remaining = d.finalize();
        assert!(remaining.is_some());
        assert!(remaining.unwrap().contains("<function_calls>"));
    }

    #[test]
    fn feed_owned_completed_is_zero_copy_passthrough() {
        let mut d = StreamingFcDetector::new(TRIGGER);
        let input1 = format!("{TRIGGER}<function_calls><function_call>");
        let _ = d.feed(&input1);
        let _ = d.feed("</function_call></function_calls>");
        assert_eq!(*d.state(), DetectorState::Completed);

        let tail = String::from("tail");
        let action = d.feed_owned(tail);
        assert_eq!(action, DetectorAction::PassThrough(String::from("tail")));
    }

    #[test]
    fn empty_feed_returns_buffer() {
        let mut d = new_detector();
        assert_eq!(d.feed(""), DetectorAction::Buffer);
    }

    #[test]
    fn completed_state_passes_through() {
        let mut d = new_detector();
        d.state = DetectorState::Completed;
        let action = d.feed("trailing text");
        assert_eq!(
            action,
            DetectorAction::PassThrough("trailing text".to_string())
        );
    }
}
