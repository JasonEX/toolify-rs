/// FC XML parser — multi-tier parsing of function calls from LLM output.
///
/// Ports the Python `parse_function_calls_xml` flow to Rust. The parser
/// extracts tool calls from text that contains a trigger signal followed by
/// an XML `<function_calls>` block.
///
/// Key invariants:
/// - S3-I3: Parser finds the LAST trigger signal outside think blocks.
/// - S3-I4: Multi-tier parsing — strict quick-xml first, regex/permissive fallback.
/// - S3-I5: CDATA unwrapping before value extraction.
/// - Parse determinism: same input always produces same output.
use crate::error::CanonicalError;
use memchr::{memchr, memchr2, memmem};
use std::borrow::Cow;

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const THINKING_OPEN: &str = "<thinking>";
const THINKING_CLOSE: &str = "</thinking>";
const REASONING_OPEN: &str = "<reasoning>";
const REASONING_CLOSE: &str = "</reasoning>";
const ANALYSIS_OPEN: &str = "<analysis>";
const ANALYSIS_CLOSE: &str = "</analysis>";
const FUNCTION_CALLS_OPEN: &[u8] = b"<function_calls>";
const FUNCTION_CALLS_CLOSE: &[u8] = b"</function_calls>";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A parsed tool call extracted from the model's XML output.
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    /// Optional tool-call id when provided by the upstream payload.
    pub id: Option<Box<str>>,
    /// The tool/function name (from the `name` attribute of `<invoke>`).
    pub name: String,
    /// The arguments as a JSON object.
    pub arguments: serde_json::Value,
    /// Optional raw JSON text for arguments, reused by streaming emitters.
    ///
    /// This is populated for `<function_call>` formats that already provide
    /// JSON text in `<args_json>`/`<arguments>`/`<parameters>` tags.
    pub arguments_json: Option<Box<str>>,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse function calls from the model's text output.
///
/// 1. Remove reasoning blocks (for parsing only), including
///    `<think>…</think>`, `<thinking>…</thinking>`,
///    `<reasoning>…</reasoning>`, and `<analysis>…</analysis>`.
/// 2. Find the **last** occurrence of `trigger_signal` and parse function-call
///    payload from there (prefer `<function_calls>` wrapper when present).
/// 3. Try strict XML parsing (Tier 1, quick-xml).
/// 4. On failure, try regex parsing (Tier 2), then permissive parsing (Tier 3).
/// 5. Validate that every call has a non-empty name and object arguments.
///
/// # Errors
///
/// Returns [`CanonicalError::FcParse`] when trigger signal/payload is missing,
/// XML is malformed, or decoded tool calls are invalid.
pub fn parse_function_calls(
    text: &str,
    trigger_signal: &str,
) -> Result<Vec<ParsedToolCall>, CanonicalError> {
    if text.is_empty() || trigger_signal.is_empty() {
        return Err(CanonicalError::FcParse(
            "empty input or trigger signal".into(),
        ));
    }

    // Step 1 — strip reasoning blocks for parsing purposes.
    let cleaned = remove_think_blocks(text);

    // Step 2 — find the last trigger occurrence that is followed by a
    // <function_calls> block.
    let cleaned_bytes = cleaned.as_bytes();
    let trigger_bytes = trigger_signal.as_bytes();
    let mut search_end = cleaned_bytes.len();
    let mut found: Option<(&str, &str)> = None;
    let mut trigger_tail: Option<&str> = None;
    while let Some(pos) = memmem::rfind(&cleaned_bytes[..search_end], trigger_bytes) {
        let sub = &cleaned[pos..];
        if trigger_tail.is_none() {
            trigger_tail = Some(sub);
        }
        if let Some((calls_xml, calls_content)) = find_function_calls_block(sub) {
            found = Some((calls_xml, calls_content));
            break;
        }
        if pos == 0 {
            break;
        }
        search_end = pos;
    }

    let tail = trigger_tail.ok_or_else(|| {
        CanonicalError::FcParse("trigger signal not followed by function-call payload".into())
    })?;

    // Step 3 & 4 — zero-allocation-ish fast path, strict parse, regex fallback,
    // then permissive fallback.
    let results = if let Some((calls_xml, calls_content)) = found {
        parse_xml_fast_function_calls(calls_content)
            .or_else(|_| parse_xml_strict(calls_xml))
            .or_else(|_| parse_xml_regex(calls_content))
            .or_else(|_| parse_xml_permissive(calls_content))
    } else {
        parse_xml_permissive(tail)
    }?;

    if results.is_empty() {
        return Err(CanonicalError::FcParse(
            "no valid tool calls found in function_calls block".into(),
        ));
    }

    // Step 5 — validate each call.
    for call in &results {
        if call.name.is_empty() {
            return Err(CanonicalError::FcParse("tool call has empty name".into()));
        }
        if !call.arguments.is_object() {
            return Err(CanonicalError::FcParse(format!(
                "tool call '{}' arguments must be a JSON object, got {}",
                call.name,
                kind_label(&call.arguments),
            )));
        }
    }

    Ok(results)
}

#[inline]
fn find_function_calls_block(text: &str) -> Option<(&str, &str)> {
    let bytes = text.as_bytes();
    let open_start = memmem::find(bytes, FUNCTION_CALLS_OPEN)?;
    let content_start = open_start + FUNCTION_CALLS_OPEN.len();
    let close_rel = memmem::find(&bytes[content_start..], FUNCTION_CALLS_CLOSE)?;
    let content_end = content_start + close_rel;
    let close_end = content_end + FUNCTION_CALLS_CLOSE.len();
    Some((
        &text[open_start..close_end],
        &text[content_start..content_end],
    ))
}

#[inline]
fn parse_xml_fast_function_calls(
    calls_content: &str,
) -> Result<Vec<ParsedToolCall>, CanonicalError> {
    const FUNCTION_CALL_OPEN: &[u8] = b"<function_call>";
    const FUNCTION_CALL_CLOSE: &[u8] = b"</function_call>";
    const INVOKE_OPEN: &[u8] = b"<invoke";
    const TOOL_OPEN: &[u8] = b"<tool>";
    const TOOL_CLOSE: &[u8] = b"</tool>";

    let bytes = calls_content.as_bytes();
    let mut first_non_ws = 0usize;
    while bytes.get(first_non_ws).is_some_and(u8::is_ascii_whitespace) {
        first_non_ws += 1;
    }

    // Pure <function_call> payloads are the hottest path. Avoid a full-buffer
    // <invoke ...> pre-scan and only fall back to mixed parsing when needed.
    if bytes[first_non_ws..].starts_with(FUNCTION_CALL_OPEN) {
        let mut cursor = first_non_ws;
        let mut results = Vec::with_capacity(2);

        while let Some(open_rel) = memmem::find(&bytes[cursor..], FUNCTION_CALL_OPEN) {
            let content_start = cursor + open_rel + FUNCTION_CALL_OPEN.len();
            let Some(close_rel) = memmem::find(&bytes[content_start..], FUNCTION_CALL_CLOSE) else {
                return Err(CanonicalError::FcParse(
                    "malformed <function_call> block".to_string(),
                ));
            };
            let content_end = content_start + close_rel;
            let block = &calls_content[content_start..content_end];

            let tool_name = extract_xml_tag_text(block, TOOL_OPEN, TOOL_CLOSE)
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| {
                    CanonicalError::FcParse("missing <tool> in function_call".to_string())
                })?
                .to_string();

            let args_text = extract_first_args_tag_text(block);
            let (arguments, arguments_json) =
                args_text.map_or_else(empty_args_json_pair, parse_args_json_with_delta_or_empty);
            let call_id = extract_first_call_id(block);

            results.push(ParsedToolCall {
                id: call_id,
                name: tool_name,
                arguments,
                arguments_json,
            });

            cursor = content_end + FUNCTION_CALL_CLOSE.len();
        }

        if !results.is_empty() && memmem::find(&bytes[cursor..], INVOKE_OPEN).is_none() {
            return Ok(results);
        }
    }

    parse_xml_fast_mixed(calls_content)
}

fn parse_xml_fast_mixed(calls_content: &str) -> Result<Vec<ParsedToolCall>, CanonicalError> {
    const FUNCTION_CALL_OPEN: &[u8] = b"<function_call>";
    const FUNCTION_CALL_CLOSE: &[u8] = b"</function_call>";
    const INVOKE_OPEN: &[u8] = b"<invoke";
    const INVOKE_CLOSE: &[u8] = b"</invoke>";
    const PARAMETER_OPEN: &[u8] = b"<parameter";
    const PARAMETER_CLOSE: &[u8] = b"</parameter>";
    const TOOL_OPEN: &[u8] = b"<tool>";
    const TOOL_CLOSE: &[u8] = b"</tool>";

    let bytes = calls_content.as_bytes();
    let mut cursor = 0usize;
    let mut results = Vec::with_capacity(2);
    while let Some(rel_lt) = memchr(b'<', &bytes[cursor..]) {
        let block_start = cursor + rel_lt;
        if bytes[block_start..].starts_with(FUNCTION_CALL_OPEN) {
            let content_start = block_start + FUNCTION_CALL_OPEN.len();
            let Some(close_rel) = memmem::find(&bytes[content_start..], FUNCTION_CALL_CLOSE) else {
                return Err(CanonicalError::FcParse(
                    "malformed <function_call> block".to_string(),
                ));
            };
            let content_end = content_start + close_rel;
            let block = &calls_content[content_start..content_end];

            let tool_name = extract_xml_tag_text(block, TOOL_OPEN, TOOL_CLOSE)
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| {
                    CanonicalError::FcParse("missing <tool> in function_call".to_string())
                })?
                .to_string();
            let args_text = extract_first_args_tag_text(block);
            let (arguments, arguments_json) =
                args_text.map_or_else(empty_args_json_pair, parse_args_json_with_delta_or_empty);
            let call_id = extract_first_call_id(block);
            results.push(ParsedToolCall {
                id: call_id,
                name: tool_name,
                arguments,
                arguments_json,
            });
            cursor = content_end + FUNCTION_CALL_CLOSE.len();
            continue;
        }
        if !bytes[block_start..].starts_with(INVOKE_OPEN) {
            cursor = block_start + 1;
            continue;
        }

        let Some(invoke_tag_end_rel) = memchr(b'>', &bytes[block_start..]) else {
            return Err(CanonicalError::FcParse(
                "malformed <invoke> start tag".to_string(),
            ));
        };
        let invoke_tag_end = block_start + invoke_tag_end_rel;
        let Some(start_tag) = calls_content.get(block_start..=invoke_tag_end) else {
            return Err(CanonicalError::FcParse(
                "invalid utf-8 boundary in <invoke> start tag".to_string(),
            ));
        };
        let Some(tool_name) = extract_name_attr_ascii(start_tag).map(ToOwned::to_owned) else {
            return Err(CanonicalError::FcParse(
                "missing name attribute on <invoke>".to_string(),
            ));
        };
        let call_id = extract_id_attr_ascii(start_tag).and_then(normalize_call_id);
        let invoke_content_start = invoke_tag_end + 1;
        let Some(close_rel) = memmem::find(&bytes[invoke_content_start..], INVOKE_CLOSE) else {
            return Err(CanonicalError::FcParse(
                "malformed <invoke> block".to_string(),
            ));
        };
        let invoke_content_end = invoke_content_start + close_rel;
        let Some(invoke_body) = calls_content.get(invoke_content_start..invoke_content_end) else {
            return Err(CanonicalError::FcParse(
                "invalid utf-8 boundary in <invoke> block".to_string(),
            ));
        };
        let invoke_bytes = invoke_body.as_bytes();
        let mut params = serde_json::Map::with_capacity(2);
        let mut invoke_cursor = 0usize;
        while let Some(param_open_rel) =
            memmem::find(&invoke_bytes[invoke_cursor..], PARAMETER_OPEN)
        {
            let param_tag_start = invoke_cursor + param_open_rel;
            let Some(param_tag_end_rel) = memchr(b'>', &invoke_bytes[param_tag_start..]) else {
                return Err(CanonicalError::FcParse(
                    "malformed <parameter> start tag".to_string(),
                ));
            };
            let param_tag_end = param_tag_start + param_tag_end_rel;
            let Some(param_tag) = invoke_body.get(param_tag_start..=param_tag_end) else {
                return Err(CanonicalError::FcParse(
                    "invalid utf-8 boundary in <parameter> start tag".to_string(),
                ));
            };
            let Some(param_name) = extract_name_attr_ascii(param_tag).map(ToOwned::to_owned) else {
                invoke_cursor = param_tag_end + 1;
                continue;
            };
            let param_value_start = param_tag_end + 1;
            let Some(param_close_rel) =
                memmem::find(&invoke_bytes[param_value_start..], PARAMETER_CLOSE)
            else {
                return Err(CanonicalError::FcParse(
                    "malformed <parameter> block".to_string(),
                ));
            };
            let param_value_end = param_value_start + param_close_rel;
            let Some(raw_value_text) = invoke_body.get(param_value_start..param_value_end) else {
                return Err(CanonicalError::FcParse(
                    "invalid utf-8 boundary in <parameter> value".to_string(),
                ));
            };
            let raw_value = unwrap_cdata(raw_value_text);
            let decoded = decode_xml_entities(raw_value.trim());
            let value = coerce_json_value(decoded.as_ref());
            params.insert(param_name, value);
            invoke_cursor = param_value_end + PARAMETER_CLOSE.len();
        }
        results.push(ParsedToolCall {
            id: call_id,
            name: tool_name,
            arguments: serde_json::Value::Object(params),
            arguments_json: None,
        });
        cursor = invoke_content_end + INVOKE_CLOSE.len();
    }

    if results.is_empty() {
        return Err(CanonicalError::FcParse(
            "fast XML parse found no function_call blocks".to_string(),
        ));
    }

    Ok(results)
}
#[inline]
fn extract_name_attr_ascii(tag: &str) -> Option<&str> {
    extract_attr_ascii(tag, b"name")
}

#[inline]
fn extract_id_attr_ascii(tag: &str) -> Option<&str> {
    extract_attr_ascii(tag, b"id")
}

#[inline]
fn extract_attr_ascii<'a>(tag: &'a str, attr: &[u8]) -> Option<&'a str> {
    let bytes = tag.as_bytes();
    let mut search_from = 0usize;
    while let Some(rel_name) = memmem::find(&bytes[search_from..], attr) {
        let mut idx = search_from + rel_name;
        let name_start = idx;
        let name_end = name_start + attr.len();
        if name_start > 0 && is_ascii_attr_char(bytes[name_start - 1]) {
            search_from = name_start + 1;
            continue;
        }
        if bytes.get(name_end).copied().is_some_and(is_ascii_attr_char) {
            search_from = name_start + 1;
            continue;
        }
        idx = name_end;
        while bytes.get(idx).is_some_and(u8::is_ascii_whitespace) {
            idx += 1;
        }
        if bytes.get(idx) != Some(&b'=') {
            search_from = name_start + 1;
            continue;
        }
        idx += 1;
        while bytes.get(idx).is_some_and(u8::is_ascii_whitespace) {
            idx += 1;
        }
        let quote = *bytes.get(idx)?;
        if quote != b'"' && quote != b'\'' {
            search_from = name_start + 1;
            continue;
        }
        let value_start = idx + 1;
        let value_end_rel = memchr(quote, &bytes[value_start..])?;
        let value_end = value_start + value_end_rel;
        return tag.get(value_start..value_end).map(str::trim);
    }
    None
}

#[inline]
const fn is_ascii_attr_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b':')
}

#[inline]
fn extract_first_args_tag_text(block: &str) -> Option<&str> {
    extract_xml_tag_text(block, b"<args_json>", b"</args_json>")
        .or_else(|| extract_xml_tag_text(block, b"<arguments>", b"</arguments>"))
        .or_else(|| extract_xml_tag_text(block, b"<parameters>", b"</parameters>"))
}

#[inline]
fn extract_first_call_id(block: &str) -> Option<Box<str>> {
    let trimmed = block.trim_start();
    if let Some(rest) = trimmed.strip_prefix("<id>") {
        let end = rest.find("</id>")?;
        return normalize_call_id(&rest[..end]);
    }
    if let Some(rest) = trimmed.strip_prefix("<tool_call_id>") {
        let end = rest.find("</tool_call_id>")?;
        return normalize_call_id(&rest[..end]);
    }
    None
}

#[inline]
fn normalize_call_id(raw: &str) -> Option<Box<str>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.len() > 128 {
        return None;
    }
    if !trimmed
        .as_bytes()
        .iter()
        .copied()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return None;
    }
    Some(trimmed.into())
}

#[inline]
fn extract_xml_tag_text<'a>(text: &'a str, open: &[u8], close: &[u8]) -> Option<&'a str> {
    let bytes = text.as_bytes();
    let start = memmem::find(bytes, open)?;
    let content_start = start + open.len();
    let end_rel = memmem::find(&bytes[content_start..], close)?;
    let content_end = content_start + end_rel;
    text.get(content_start..content_end)
}

// ---------------------------------------------------------------------------
// Think-block removal
// ---------------------------------------------------------------------------

/// Remove all reasoning blocks (including nested ones) from `text`.
///
/// Supported wrappers:
/// - `<think>…</think>`
/// - `<thinking>…</thinking>`
/// - `<reasoning>…</reasoning>`
/// - `<analysis>…</analysis>`
///
/// The removal is linear-time and preserves existing behavior for unmatched
/// reasoning blocks (keeps the unmatched portion verbatim).
fn remove_think_blocks(text: &str) -> Cow<'_, str> {
    if !contains_reasoning_open_tag(text) {
        return Cow::Borrowed(text);
    }

    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    let mut depth = 0usize;
    let mut unmatched_start: Option<usize> = None;

    while i < bytes.len() {
        let Some(rel_lt) = memchr(b'<', &bytes[i..]) else {
            if depth == 0 {
                out.push_str(&text[i..]);
            }
            break;
        };
        let abs = i + rel_lt;
        if depth == 0 && abs > i {
            out.push_str(&text[i..abs]);
        }

        if let Some(open_len) = reasoning_open_tag_len_at(&text[abs..]) {
            if depth == 0 {
                unmatched_start = Some(abs);
            }
            depth += 1;
            i = abs + open_len;
            continue;
        }

        if let Some(close_len) = reasoning_close_tag_len_at(&text[abs..]) {
            if depth == 0 {
                out.push('<');
                i = abs + 1;
                continue;
            }
            depth -= 1;
            i = abs + close_len;
            if depth == 0 {
                unmatched_start = None;
            }
            continue;
        }

        if depth == 0 {
            out.push('<');
        }
        i = abs + 1;
    }

    if depth > 0 {
        if let Some(start) = unmatched_start {
            out.push_str(&text[start..]);
        }
    }

    Cow::Owned(out)
}

#[inline]
fn contains_reasoning_open_tag(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut cursor = 0usize;
    while let Some(rel_lt) = memchr(b'<', &bytes[cursor..]) {
        let abs = cursor + rel_lt;
        let Some(first) = bytes.get(abs + 1).copied() else {
            break;
        };
        let matched = match first {
            b't' => {
                bytes[abs..].starts_with(THINK_OPEN.as_bytes())
                    || bytes[abs..].starts_with(THINKING_OPEN.as_bytes())
            }
            b'r' => bytes[abs..].starts_with(REASONING_OPEN.as_bytes()),
            b'a' => bytes[abs..].starts_with(ANALYSIS_OPEN.as_bytes()),
            _ => false,
        };
        if matched {
            return true;
        }
        cursor = abs + 1;
    }
    false
}

#[inline]
fn reasoning_open_tag_len_at(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.first().copied()? != b'<' {
        return None;
    }
    match bytes.get(1).copied() {
        Some(b't') => {
            if text.starts_with(THINK_OPEN) {
                Some(THINK_OPEN.len())
            } else if text.starts_with(THINKING_OPEN) {
                Some(THINKING_OPEN.len())
            } else {
                None
            }
        }
        Some(b'r') => text
            .starts_with(REASONING_OPEN)
            .then_some(REASONING_OPEN.len()),
        Some(b'a') => text
            .starts_with(ANALYSIS_OPEN)
            .then_some(ANALYSIS_OPEN.len()),
        _ => None,
    }
}

#[inline]
fn reasoning_close_tag_len_at(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.first().copied()? != b'<' || bytes.get(1).copied()? != b'/' {
        return None;
    }
    match bytes.get(2).copied() {
        Some(b't') => {
            if text.starts_with(THINK_CLOSE) {
                Some(THINK_CLOSE.len())
            } else if text.starts_with(THINKING_CLOSE) {
                Some(THINKING_CLOSE.len())
            } else {
                None
            }
        }
        Some(b'r') => text
            .starts_with(REASONING_CLOSE)
            .then_some(REASONING_CLOSE.len()),
        Some(b'a') => text
            .starts_with(ANALYSIS_CLOSE)
            .then_some(ANALYSIS_CLOSE.len()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tier 1: strict XML parse via quick-xml
// ---------------------------------------------------------------------------

#[inline]
fn is_args_tag(tag: &[u8]) -> bool {
    matches!(tag, b"args_json" | b"arguments" | b"parameters")
}

#[derive(Debug)]
enum StrictXmlState {
    Outside,
    InFunctionCalls,
    InInvoke {
        name: String,
        params: serde_json::Map<String, serde_json::Value>,
    },
    InParameter {
        invoke_name: String,
        invoke_params: serde_json::Map<String, serde_json::Value>,
        param_name: String,
        param_text: String,
    },
    InFunctionCall {
        tool_name: String,
        arguments: serde_json::Value,
        arguments_json: Option<Box<str>>,
        call_id: Option<Box<str>>,
    },
    InTool {
        fc_arguments: serde_json::Value,
        fc_arguments_json: Option<Box<str>>,
        fc_call_id: Option<Box<str>>,
        text: String,
    },
    InArgsJson {
        fc_tool_name: String,
        fc_call_id: Option<Box<str>>,
        text: String,
    },
}

/// Parse the `<function_calls>…</function_calls>` block using quick-xml.
///
/// Supports two XML formats:
///
/// Format 1 (invoke):
/// ```xml
/// <function_calls>
///   <invoke name="tool_name">
///     <parameter name="param_name">value</parameter>
///   </invoke>
/// </function_calls>
/// ```
///
/// Format 2 (`function_call)`:
/// ```xml
/// <function_calls>
///   <function_call>
///     <tool>tool_name</tool>
///     <args_json><![CDATA[{"key": "value"}]]></args_json>
///   </function_call>
/// </function_calls>
/// ```
fn parse_xml_strict(xml_text: &str) -> Result<Vec<ParsedToolCall>, CanonicalError> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    // State machine:
    // - Outside -> InFunctionCalls (on <function_calls>)
    // - InFunctionCalls -> InInvoke (on <invoke>) OR InFunctionCall (on <function_call>)
    // - InInvoke -> InParameter (on <parameter>)
    // - InParameter -> accumulate text (including CDATA)
    // - InFunctionCall -> InTool (on <tool>) OR InArgsJson (on <args_json>)
    // - InTool -> accumulate tool name text
    // - InArgsJson -> accumulate JSON text (including CDATA)

    let mut reader = Reader::from_str(xml_text);
    let mut results: Vec<ParsedToolCall> = Vec::with_capacity(2);

    let mut state = StrictXmlState::Outside;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let name = e.name();
                let tag = name.as_ref();
                match state {
                    StrictXmlState::Outside if tag == b"function_calls" => {
                        state = StrictXmlState::InFunctionCalls;
                    }
                    StrictXmlState::InFunctionCalls if tag == b"invoke" => {
                        let name = extract_name_attr(e)?;
                        state = StrictXmlState::InInvoke {
                            name,
                            params: serde_json::Map::new(),
                        };
                    }
                    StrictXmlState::InFunctionCalls if tag == b"function_call" => {
                        let name_attr = extract_optional_name_attr(e).unwrap_or_default();
                        let id_attr =
                            extract_optional_id_attr(e).and_then(|id| normalize_call_id(&id));
                        state = StrictXmlState::InFunctionCall {
                            tool_name: name_attr,
                            arguments: serde_json::Value::Object(serde_json::Map::new()),
                            arguments_json: None,
                            call_id: id_attr,
                        };
                    }
                    StrictXmlState::InInvoke {
                        ref name,
                        ref params,
                    } if tag == b"parameter" => {
                        let param_name = extract_name_attr(e)?;
                        state = StrictXmlState::InParameter {
                            invoke_name: name.clone(),
                            invoke_params: params.clone(),
                            param_name,
                            param_text: String::new(),
                        };
                    }
                    StrictXmlState::InFunctionCall {
                        tool_name: _,
                        ref arguments,
                        ref arguments_json,
                        ref call_id,
                    } if tag == b"tool" => {
                        state = StrictXmlState::InTool {
                            fc_arguments: arguments.clone(),
                            fc_arguments_json: arguments_json.clone(),
                            fc_call_id: call_id.clone(),
                            text: String::new(),
                        };
                    }
                    StrictXmlState::InFunctionCall {
                        ref tool_name,
                        ref call_id,
                        arguments: _,
                        arguments_json: _,
                    } if is_args_tag(tag) => {
                        state = StrictXmlState::InArgsJson {
                            fc_tool_name: tool_name.clone(),
                            fc_call_id: call_id.clone(),
                            text: String::new(),
                        };
                    }
                    _ => {
                        // Unknown nested tag — skip.
                    }
                }
            }
            Ok(Event::End(ref e)) => {
                let name = e.name();
                let tag = name.as_ref();
                match state {
                    StrictXmlState::InParameter {
                        ref invoke_name,
                        ref mut invoke_params,
                        ref param_name,
                        ref param_text,
                    } if tag == b"parameter" => {
                        let raw_value = unwrap_cdata(param_text);
                        let value = coerce_json_value(&raw_value);
                        invoke_params.insert(param_name.clone(), value);
                        // Back to InInvoke.
                        let name = invoke_name.clone();
                        let params = invoke_params.clone();
                        state = StrictXmlState::InInvoke { name, params };
                    }
                    StrictXmlState::InInvoke {
                        ref name,
                        ref params,
                    } if tag == b"invoke" => {
                        results.push(ParsedToolCall {
                            id: None,
                            name: name.clone(),
                            arguments: serde_json::Value::Object(params.clone()),
                            arguments_json: None,
                        });
                        state = StrictXmlState::InFunctionCalls;
                    }
                    StrictXmlState::InTool {
                        ref fc_arguments,
                        ref fc_arguments_json,
                        ref fc_call_id,
                        ref text,
                    } if tag == b"tool" => {
                        // Save the accumulated text as the tool name.
                        state = StrictXmlState::InFunctionCall {
                            tool_name: text.trim().to_string(),
                            arguments: fc_arguments.clone(),
                            arguments_json: fc_arguments_json.clone(),
                            call_id: fc_call_id.clone(),
                        };
                    }
                    StrictXmlState::InArgsJson {
                        ref fc_tool_name,
                        ref fc_call_id,
                        ref text,
                    } if is_args_tag(tag) => {
                        // Parse the accumulated text as JSON args.
                        let (parsed, parsed_json) = parse_args_json_with_delta_or_empty(text);
                        state = StrictXmlState::InFunctionCall {
                            tool_name: fc_tool_name.clone(),
                            arguments: parsed,
                            arguments_json: parsed_json,
                            call_id: fc_call_id.clone(),
                        };
                    }
                    StrictXmlState::InFunctionCall {
                        ref tool_name,
                        ref arguments,
                        ref arguments_json,
                        ref call_id,
                    } if tag == b"function_call" => {
                        results.push(ParsedToolCall {
                            id: call_id.clone(),
                            name: tool_name.clone(),
                            arguments: arguments.clone(),
                            arguments_json: arguments_json.clone(),
                        });
                        state = StrictXmlState::InFunctionCalls;
                    }
                    StrictXmlState::InFunctionCalls if tag == b"function_calls" => {
                        state = StrictXmlState::Outside;
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                match state {
                    StrictXmlState::InParameter {
                        ref mut param_text, ..
                    } => {
                        // Unescape XML entities.
                        match e.unescape() {
                            Ok(cow) => param_text.push_str(&cow),
                            Err(_) => {
                                param_text.push_str(&String::from_utf8_lossy(e.as_ref()));
                            }
                        }
                    }
                    StrictXmlState::InTool { ref mut text, .. }
                    | StrictXmlState::InArgsJson { ref mut text, .. } => match e.unescape() {
                        Ok(cow) => text.push_str(&cow),
                        Err(_) => {
                            text.push_str(&String::from_utf8_lossy(e.as_ref()));
                        }
                    },
                    _ => {}
                }
            }
            Ok(Event::CData(ref e)) => {
                match state {
                    StrictXmlState::InParameter {
                        ref mut param_text, ..
                    } => {
                        // quick-xml gives us the raw CDATA content (without the
                        // `<![CDATA[` / `]]>` wrapper).
                        param_text.push_str(&String::from_utf8_lossy(e.as_ref()));
                    }
                    StrictXmlState::InArgsJson { ref mut text, .. } => {
                        text.push_str(&String::from_utf8_lossy(e.as_ref()));
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(CanonicalError::FcParse(format!("XML parse error: {e}")));
            }
            _ => {}
        }
    }

    if results.is_empty() {
        return Err(CanonicalError::FcParse(
            "strict XML parse found no tool call elements".into(),
        ));
    }

    Ok(results)
}

/// Extract the `name="..."` attribute from a quick-xml start element.
fn extract_name_attr(e: &quick_xml::events::BytesStart<'_>) -> Result<String, CanonicalError> {
    for attr in e.attributes().flatten() {
        if attr.key.as_ref() == b"name" {
            return Ok(String::from_utf8_lossy(&attr.value).to_string());
        }
    }
    Err(CanonicalError::FcParse(format!(
        "missing 'name' attribute on <{}>",
        String::from_utf8_lossy(e.name().as_ref()),
    )))
}

fn extract_optional_name_attr(e: &quick_xml::events::BytesStart<'_>) -> Option<String> {
    for attr in e.attributes().flatten() {
        if attr.key.as_ref() == b"name" {
            return Some(String::from_utf8_lossy(&attr.value).to_string());
        }
    }
    None
}

fn extract_optional_id_attr(e: &quick_xml::events::BytesStart<'_>) -> Option<String> {
    for attr in e.attributes().flatten() {
        if attr.key.as_ref() == b"id" {
            return Some(String::from_utf8_lossy(&attr.value).to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tier 2/3: regex & permissive fallback
// ---------------------------------------------------------------------------

/// Parse function calls using regex patterns — more tolerant of malformed XML.
///
/// Supports both `<invoke>` format and `<function_call>` format.
fn parse_xml_regex(calls_content: &str) -> Result<Vec<ParsedToolCall>, CanonicalError> {
    const INVOKE_OPEN: &[u8] = b"<invoke";
    const INVOKE_CLOSE: &[u8] = b"</invoke>";
    const PARAMETER_OPEN: &[u8] = b"<parameter";
    const PARAMETER_CLOSE: &[u8] = b"</parameter>";
    const FUNCTION_CALL_OPEN: &[u8] = b"<function_call>";
    const FUNCTION_CALL_CLOSE: &[u8] = b"</function_call>";
    const TOOL_OPEN: &[u8] = b"<tool>";
    const TOOL_CLOSE: &[u8] = b"</tool>";

    let mut results: Vec<ParsedToolCall> = Vec::with_capacity(2);
    let bytes = calls_content.as_bytes();

    // --- Parse <invoke> blocks (best effort) ---
    let mut invoke_cursor = 0usize;
    while let Some(open_rel) = memmem::find(&bytes[invoke_cursor..], INVOKE_OPEN) {
        let invoke_start = invoke_cursor + open_rel;
        let Some(tag_end_rel) = memchr(b'>', &bytes[invoke_start..]) else {
            break;
        };
        let invoke_tag_end = invoke_start + tag_end_rel;
        let Some(start_tag) = calls_content.get(invoke_start..=invoke_tag_end) else {
            break;
        };
        let Some(name) = extract_name_attr_ascii(start_tag).map(ToOwned::to_owned) else {
            invoke_cursor = invoke_tag_end + 1;
            continue;
        };
        let call_id = extract_id_attr_ascii(start_tag).and_then(normalize_call_id);
        let body_start = invoke_tag_end + 1;
        let body_end = match memmem::find(&bytes[body_start..], INVOKE_CLOSE) {
            Some(close_rel) => body_start + close_rel,
            None => calls_content.len(),
        };
        let Some(invoke_body) = calls_content.get(body_start..body_end) else {
            break;
        };

        // Extract parameters from this invoke block.
        let mut params = serde_json::Map::with_capacity(2);
        let invoke_bytes = invoke_body.as_bytes();
        let mut param_cursor = 0usize;
        while let Some(param_open_rel) = memmem::find(&invoke_bytes[param_cursor..], PARAMETER_OPEN)
        {
            let param_start = param_cursor + param_open_rel;
            let Some(param_tag_end_rel) = memchr(b'>', &invoke_bytes[param_start..]) else {
                break;
            };
            let param_tag_end = param_start + param_tag_end_rel;
            let Some(param_tag) = invoke_body.get(param_start..=param_tag_end) else {
                break;
            };
            let Some(param_name) = extract_name_attr_ascii(param_tag).map(ToOwned::to_owned) else {
                param_cursor = param_tag_end + 1;
                continue;
            };
            let param_value_start = param_tag_end + 1;
            let Some(param_close_rel) =
                memmem::find(&invoke_bytes[param_value_start..], PARAMETER_CLOSE)
            else {
                break;
            };
            let param_value_end = param_value_start + param_close_rel;
            let Some(value_text) = invoke_body.get(param_value_start..param_value_end) else {
                break;
            };
            let value_text = unwrap_cdata(value_text);
            let value = coerce_json_value(&value_text);
            params.insert(param_name, value);
            param_cursor = param_value_end + PARAMETER_CLOSE.len();
        }

        results.push(ParsedToolCall {
            id: call_id,
            name,
            arguments: serde_json::Value::Object(params),
            arguments_json: None,
        });
        invoke_cursor = body_end.saturating_add(INVOKE_CLOSE.len());
    }

    // --- Parse <function_call> blocks ---
    let mut fc_cursor = 0usize;
    while let Some(open_rel) = memmem::find(&bytes[fc_cursor..], FUNCTION_CALL_OPEN) {
        let content_start = fc_cursor + open_rel + FUNCTION_CALL_OPEN.len();
        let Some(close_rel) = memmem::find(&bytes[content_start..], FUNCTION_CALL_CLOSE) else {
            break;
        };
        let content_end = content_start + close_rel;
        let Some(fc_body) = calls_content.get(content_start..content_end) else {
            break;
        };

        let tool_name = if let Some(name) = extract_xml_tag_text(fc_body, TOOL_OPEN, TOOL_CLOSE)
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            name.to_string()
        } else {
            // Skip function_call blocks without a <tool> tag.
            fc_cursor = content_end + FUNCTION_CALL_CLOSE.len();
            continue;
        };

        let (arguments, arguments_json) =
            extract_xml_tag_text(fc_body, b"<args_json>", b"</args_json>")
                .map_or_else(empty_args_json_pair, parse_args_json_with_delta_or_empty);
        let call_id = extract_first_call_id(fc_body);

        results.push(ParsedToolCall {
            id: call_id,
            name: tool_name,
            arguments,
            arguments_json,
        });
        fc_cursor = content_end + FUNCTION_CALL_CLOSE.len();
    }

    if results.is_empty() {
        return Err(CanonicalError::FcParse(
            "regex parse found no tool call elements".into(),
        ));
    }

    Ok(results)
}

/// Parse function calls with permissive patterns for malformed-but-salvageable outputs.
///
/// Handles:
/// - single or double-quoted attributes
/// - `<tool>` / `<name>` for function name
/// - `<args_json>` / `<arguments>` / `<parameters>` for arguments
/// - payloads where `<function_calls>` wrapper is omitted
fn parse_xml_permissive(text: &str) -> Result<Vec<ParsedToolCall>, CanonicalError> {
    let mut results: Vec<ParsedToolCall> = Vec::with_capacity(2);

    let mut fc_cursor = 0usize;
    while let Some(fc_block) = next_tag_block_case_insensitive(text, b"function_call", fc_cursor) {
        let tool_name = extract_first_tag_body_case_insensitive(fc_block.body, &[b"tool", b"name"])
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| extract_name_attr_permissive(fc_block.attrs));

        if let Some(tool_name) = tool_name {
            let call_id =
                extract_first_tag_body_case_insensitive(fc_block.body, &[b"id", b"tool_call_id"])
                    .and_then(normalize_call_id)
                    .or_else(|| {
                        extract_id_attr_permissive(fc_block.attrs)
                            .and_then(|id| normalize_call_id(id.as_str()))
                    });
            let (arguments, arguments_json) = extract_first_tag_body_case_insensitive(
                fc_block.body,
                &[b"args_json", b"arguments", b"parameters"],
            )
            .map_or_else(empty_args_json_pair, parse_args_json_with_delta_or_empty);

            results.push(ParsedToolCall {
                id: call_id,
                name: tool_name,
                arguments,
                arguments_json,
            });
        }

        fc_cursor = fc_block.next_cursor;
    }

    let mut invoke_cursor = 0usize;
    while let Some(invoke_block) = next_tag_block_case_insensitive(text, b"invoke", invoke_cursor) {
        let Some(name) = extract_name_attr_permissive(invoke_block.attrs) else {
            invoke_cursor = invoke_block.next_cursor;
            continue;
        };

        let mut params = serde_json::Map::with_capacity(2);
        let mut param_cursor = 0usize;
        while let Some(param_block) =
            next_tag_block_case_insensitive(invoke_block.body, b"parameter", param_cursor)
        {
            if let Some(param_name) = extract_name_attr_permissive(param_block.attrs) {
                let raw_value = unwrap_cdata(param_block.body);
                let value_text = decode_xml_entities(raw_value.trim());
                let value = coerce_json_value(value_text.as_ref());
                params.insert(param_name, value);
            }
            param_cursor = param_block.next_cursor;
        }

        let call_id = extract_id_attr_permissive(invoke_block.attrs)
            .and_then(|id| normalize_call_id(id.as_str()));
        results.push(ParsedToolCall {
            id: call_id,
            name,
            arguments: serde_json::Value::Object(params),
            arguments_json: None,
        });
        invoke_cursor = invoke_block.next_cursor;
    }

    if results.is_empty() {
        return Err(CanonicalError::FcParse(
            "permissive parse found no tool call elements".into(),
        ));
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct TagBlock<'a> {
    start: usize,
    attrs: &'a str,
    body: &'a str,
    next_cursor: usize,
}

#[inline]
const fn is_ascii_word_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

#[inline]
const fn is_tag_name_boundary(byte: u8) -> bool {
    !is_ascii_word_char(byte)
}

#[inline]
fn find_open_tag_start_case_insensitive(
    bytes: &[u8],
    tag_name: &[u8],
    from: usize,
) -> Option<usize> {
    let mut cursor = from;
    while let Some(rel_lt) = memchr(b'<', &bytes[cursor..]) {
        let start = cursor + rel_lt;
        let name_start = start + 1;
        let name_end = name_start.checked_add(tag_name.len())?;
        let name_slice = bytes.get(name_start..name_end)?;
        if name_slice.eq_ignore_ascii_case(tag_name) {
            let boundary = bytes.get(name_end).copied().unwrap_or(b'>');
            if is_tag_name_boundary(boundary) {
                return Some(start);
            }
        }
        cursor = start + 1;
    }
    None
}

#[inline]
fn find_close_tag_start_case_insensitive(
    bytes: &[u8],
    tag_name: &[u8],
    from: usize,
) -> Option<usize> {
    let mut cursor = from;
    while let Some(rel_lt) = memchr(b'<', &bytes[cursor..]) {
        let start = cursor + rel_lt;
        if bytes.get(start + 1) != Some(&b'/') {
            cursor = start + 1;
            continue;
        }
        let name_start = start + 2;
        let name_end = name_start.checked_add(tag_name.len())?;
        let name_slice = bytes.get(name_start..name_end)?;
        if name_slice.eq_ignore_ascii_case(tag_name) {
            let boundary = bytes.get(name_end).copied().unwrap_or(b'>');
            if is_tag_name_boundary(boundary) {
                return Some(start);
            }
        }
        cursor = start + 1;
    }
    None
}

fn next_tag_block_case_insensitive<'a>(
    text: &'a str,
    tag_name: &[u8],
    from: usize,
) -> Option<TagBlock<'a>> {
    let bytes = text.as_bytes();
    let mut search_from = from;
    while let Some(start) = find_open_tag_start_case_insensitive(bytes, tag_name, search_from) {
        let name_end = start + 1 + tag_name.len();
        let Some(open_gt_rel) = memchr(b'>', &bytes[name_end..]) else {
            search_from = start + 1;
            continue;
        };
        let open_gt = name_end + open_gt_rel;
        let body_start = open_gt + 1;
        let Some(close_start) = find_close_tag_start_case_insensitive(bytes, tag_name, body_start)
        else {
            search_from = start + 1;
            continue;
        };
        let close_name_end = close_start + 2 + tag_name.len();
        let Some(close_gt_rel) = memchr(b'>', &bytes[close_name_end..]) else {
            search_from = start + 1;
            continue;
        };
        let close_gt = close_name_end + close_gt_rel;

        let Some(attrs) = text.get(name_end..open_gt) else {
            search_from = start + 1;
            continue;
        };
        let Some(body) = text.get(body_start..close_start) else {
            search_from = start + 1;
            continue;
        };

        return Some(TagBlock {
            start,
            attrs,
            body,
            next_cursor: close_gt + 1,
        });
    }
    None
}

fn extract_first_tag_body_case_insensitive<'a>(text: &'a str, tags: &[&[u8]]) -> Option<&'a str> {
    let mut best: Option<TagBlock<'_>> = None;
    for &tag in tags {
        if let Some(candidate) = next_tag_block_case_insensitive(text, tag, 0) {
            if best
                .as_ref()
                .is_none_or(|current| candidate.start < current.start)
            {
                best = Some(candidate);
            }
        }
    }
    best.map(|block| block.body)
}

/// Unwrap CDATA sections from a string.
///
/// If the string contains `<![CDATA[...]]>`, extract and concatenate the inner
/// content of all CDATA sections. Otherwise return the string as-is.
fn unwrap_cdata(text: &str) -> Cow<'_, str> {
    const CDATA_OPEN: &[u8] = b"<![CDATA[";
    const CDATA_CLOSE: &[u8] = b"]]>";

    let bytes = text.as_bytes();
    let mut out: Option<String> = None;
    let mut cursor = 0usize;

    while let Some(open_rel) = memmem::find(&bytes[cursor..], CDATA_OPEN) {
        if out.is_none() {
            out = Some(String::with_capacity(text.len()));
        }
        let open = cursor + open_rel;
        let content_start = open + CDATA_OPEN.len();
        let Some(close_rel) = memmem::find(&bytes[content_start..], CDATA_CLOSE) else {
            // Keep behavior conservative on malformed payloads.
            return Cow::Borrowed(text);
        };
        let content_end = content_start + close_rel;
        if let Some(part) = text.get(content_start..content_end) {
            out.get_or_insert_with(String::new).push_str(part);
        }
        cursor = content_end + CDATA_CLOSE.len();
    }

    out.map_or_else(|| Cow::Borrowed(text), Cow::Owned)
}

/// Decode common XML entities in text content.
fn decode_xml_entities(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let Some(first_amp) = memchr(b'&', bytes) else {
        return Cow::Borrowed(text);
    };

    let mut out = String::with_capacity(text.len());
    if first_amp > 0 {
        out.push_str(&text[..first_amp]);
    }
    let mut i = first_amp;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"&amp;") {
            out.push('&');
            i += 5;
        } else if bytes[i..].starts_with(b"&lt;") {
            out.push('<');
            i += 4;
        } else if bytes[i..].starts_with(b"&gt;") {
            out.push('>');
            i += 4;
        } else if bytes[i..].starts_with(b"&quot;") {
            out.push('"');
            i += 6;
        } else if bytes[i..].starts_with(b"&apos;") {
            out.push('\'');
            i += 6;
        } else {
            out.push('&');
            i += 1;
        }

        let Some(next_rel_amp) = memchr(b'&', &bytes[i..]) else {
            out.push_str(&text[i..]);
            break;
        };
        let next_amp = i + next_rel_amp;
        if next_amp > i {
            out.push_str(&text[i..next_amp]);
        }
        i = next_amp;
    }
    Cow::Owned(out)
}

fn extract_name_attr_permissive(attrs: &str) -> Option<String> {
    extract_attr_permissive(attrs, b"name")
}

fn extract_id_attr_permissive(attrs: &str) -> Option<String> {
    extract_attr_permissive(attrs, b"id")
}

fn extract_attr_permissive(attrs: &str, attr: &[u8]) -> Option<String> {
    if attr.is_empty() {
        return None;
    }

    let bytes = attrs.as_bytes();
    let mut search_from = 0usize;
    while let Some(rel_name) = find_ascii_case_insensitive(&bytes[search_from..], attr) {
        let name_start = search_from + rel_name;
        let name_end = name_start + attr.len();

        // Preserve regex word-boundary behavior around the attribute name.
        let left_ok = name_start == 0
            || !matches!(bytes[name_start - 1], b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_');
        let right_ok = name_end >= bytes.len()
            || !matches!(bytes[name_end], b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_');
        if !left_ok || !right_ok {
            search_from = name_start + 1;
            continue;
        }

        let mut idx = name_end;
        while bytes.get(idx).is_some_and(u8::is_ascii_whitespace) {
            idx += 1;
        }
        if bytes.get(idx) != Some(&b'=') {
            search_from = name_start + 1;
            continue;
        }
        idx += 1;
        while bytes.get(idx).is_some_and(u8::is_ascii_whitespace) {
            idx += 1;
        }
        let quote = *bytes.get(idx)?;
        if quote != b'"' && quote != b'\'' {
            search_from = name_start + 1;
            continue;
        }
        let value_start = idx + 1;
        let value_end_rel = memchr(quote, &bytes[value_start..])?;
        let value_end = value_start + value_end_rel;
        return attrs
            .get(value_start..value_end)
            .map(str::trim)
            .map(ToOwned::to_owned);
    }
    None
}

#[inline]
fn find_ascii_case_insensitive(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if haystack.len() < needle.len() {
        return None;
    }

    if let Some((&first, rest)) = needle.split_first() {
        let mut cursor = 0usize;
        while let Some(rel_first) = memchr2(
            first.to_ascii_lowercase(),
            first.to_ascii_uppercase(),
            &haystack[cursor..],
        ) {
            let start = cursor + rel_first;
            let end = start + needle.len();
            if haystack[start..end].eq_ignore_ascii_case(needle) {
                return Some(start);
            }
            if rest.is_empty() {
                return Some(start);
            }
            cursor = start + 1;
            if haystack.len() - cursor < needle.len() {
                break;
            }
        }
        return None;
    }

    haystack
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle))
}

/// If a value looks like JSON, parse it.
/// Otherwise return it as a JSON string.
fn coerce_json_value(s: &str) -> serde_json::Value {
    let trimmed = s.trim();
    if should_attempt_json_parse(trimmed) {
        match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => serde_json::Value::String(s.to_string()),
        }
    } else {
        serde_json::Value::String(s.to_string())
    }
}

#[inline]
fn should_attempt_json_parse(trimmed: &str) -> bool {
    let Some(first) = trimmed.as_bytes().first().copied() else {
        return false;
    };

    matches!(first, b'{' | b'[' | b'-' | b'0'..=b'9' | b't' | b'f' | b'n')
}

#[inline]
fn empty_json_object() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

#[inline]
fn empty_args_json_pair() -> (serde_json::Value, Option<Box<str>>) {
    (empty_json_object(), Some("{}".into()))
}

#[inline]
fn parse_args_json_with_delta_or_empty(args_text: &str) -> (serde_json::Value, Option<Box<str>>) {
    let trimmed = args_text.trim();
    if trimmed.is_empty() {
        return empty_args_json_pair();
    }

    // Fast path: most FC args are already raw JSON in <args_json>.
    if let Ok(parsed) = serde_json::from_str(trimmed) {
        return (parsed, Some(trimmed.into()));
    }

    let bytes = trimmed.as_bytes();
    let needs_cdata_unwrap = memmem::find(bytes, b"<![CDATA[").is_some();
    let needs_entity_decode = memchr(b'&', bytes).is_some();
    if !needs_cdata_unwrap && !needs_entity_decode {
        return empty_args_json_pair();
    }

    let normalized = if needs_cdata_unwrap {
        unwrap_cdata(trimmed)
    } else {
        Cow::Borrowed(trimmed)
    };
    let normalized = if needs_entity_decode {
        decode_xml_entities(normalized.as_ref())
    } else {
        normalized
    };
    let normalized = normalized.trim();
    if normalized.is_empty() {
        return empty_args_json_pair();
    }

    match serde_json::from_str(normalized) {
        Ok(parsed) => (parsed, Some(normalized.into())),
        Err(_) => empty_args_json_pair(),
    }
}

/// Human-readable label for a JSON value kind.
fn kind_label(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "parser_tests.rs"]
mod tests;
