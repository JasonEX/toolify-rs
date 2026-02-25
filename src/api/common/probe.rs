use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};

use memchr::memmem;
use serde::Deserialize;

use crate::error::CanonicalError;
use crate::json_scan::{
    find_top_level_field_value_range, parse_json_string_end, parse_json_value_end, skip_ws,
};
use crate::util::sampled_bytes_hash;

pub(crate) struct CommonRequestProbe<'a> {
    pub model: Cow<'a, str>,
    pub stream: Option<bool>,
    pub has_tools: bool,
    pub ranges: Option<CommonProbeRanges>,
}

const PROBE_RANGES_CACHE_WAYS: usize = 8;
const PROBE_RANGES_CACHE_MAX_BODY_BYTES: usize = 16 * 1024;
static STREAM_FIELD_FINDER: LazyLock<memmem::Finder<'static>> =
    LazyLock::new(|| memmem::Finder::new(br#""stream""#));
static TOOLS_FIELD_FINDER: LazyLock<memmem::Finder<'static>> =
    LazyLock::new(|| memmem::Finder::new(br#""tools""#));

#[derive(Clone)]
struct ProbeRangesCacheEntry {
    key_hash: u64,
    body: Arc<[u8]>,
    ranges: CommonProbeRanges,
}

thread_local! {
    static PROBE_RANGES_CACHE: RefCell<VecDeque<ProbeRangesCacheEntry>> =
        RefCell::new(VecDeque::with_capacity(PROBE_RANGES_CACHE_WAYS));
    static PROBE_RANGES_LAST_HIT: RefCell<Option<ProbeRangesCacheEntry>> =
        const { RefCell::new(None) };
}

#[inline]
fn probe_ranges_cacheable(bytes: &[u8]) -> bool {
    bytes.len() <= PROBE_RANGES_CACHE_MAX_BODY_BYTES
}

#[inline]
fn probe_ranges_key_hash(bytes: &[u8]) -> u64 {
    sampled_bytes_hash(bytes)
}

#[inline]
fn probe_ranges_thread_cache_get(bytes: &[u8], key_hash: u64) -> Option<CommonProbeRanges> {
    PROBE_RANGES_LAST_HIT.with(|slot| {
        let guard = slot.borrow();
        let entry = guard.as_ref()?;
        if entry.key_hash != key_hash || entry.body.as_ref() != bytes {
            return None;
        }
        Some(entry.ranges.clone())
    })
}

#[inline]
fn probe_ranges_thread_cache_set(entry: &ProbeRangesCacheEntry) {
    PROBE_RANGES_LAST_HIT.with(|slot| {
        *slot.borrow_mut() = Some(entry.clone());
    });
}

fn probe_ranges_cache_get(bytes: &[u8], key_hash_hint: Option<u64>) -> Option<CommonProbeRanges> {
    if !probe_ranges_cacheable(bytes) {
        return None;
    }
    let key_hash = key_hash_hint.unwrap_or_else(|| probe_ranges_key_hash(bytes));
    if let Some(hit) = probe_ranges_thread_cache_get(bytes, key_hash) {
        return Some(hit);
    }

    PROBE_RANGES_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let pos = cache
            .iter()
            .rposition(|entry| entry.key_hash == key_hash && entry.body.as_ref() == bytes)?;
        if pos + 1 != cache.len() {
            let entry = cache.remove(pos)?;
            let ranges = entry.ranges.clone();
            probe_ranges_thread_cache_set(&entry);
            cache.push_back(entry);
            return Some(ranges);
        }
        let entry = cache.get(pos)?;
        probe_ranges_thread_cache_set(entry);
        Some(entry.ranges.clone())
    })
}

fn probe_ranges_cache_insert(bytes: &[u8], key_hash_hint: Option<u64>, ranges: &CommonProbeRanges) {
    if !probe_ranges_cacheable(bytes) {
        return;
    }
    let key_hash = key_hash_hint.unwrap_or_else(|| probe_ranges_key_hash(bytes));
    PROBE_RANGES_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(pos) = cache
            .iter()
            .position(|entry| entry.key_hash == key_hash && entry.body.as_ref() == bytes)
        {
            let _ = cache.remove(pos);
        }
        if cache.len() >= PROBE_RANGES_CACHE_WAYS {
            let _ = cache.pop_front();
        }
        let entry = ProbeRangesCacheEntry {
            key_hash,
            body: Arc::from(bytes),
            ranges: ranges.clone(),
        };
        probe_ranges_thread_cache_set(&entry);
        cache.push_back(entry);
    });
}

#[derive(Deserialize)]
struct SlowCommonProbe {
    model: String,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
}

pub(crate) fn parse_common_request_probe<'a>(
    body: &'a bytes::Bytes,
    request_name: &str,
) -> Result<CommonRequestProbe<'a>, CanonicalError> {
    let body_bytes = body.as_ref();
    if let Some(fast_probe) = try_parse_probe_without_stream_and_tools(body_bytes) {
        return Ok(fast_probe);
    }

    let body_key_hash =
        probe_ranges_cacheable(body_bytes).then(|| probe_ranges_key_hash(body_bytes));

    if let Some(ranges) = probe_ranges_cache_get(body_bytes, body_key_hash) {
        if let Some(model_range) = ranges.model.as_ref() {
            let model = parse_model_token(body_bytes, model_range).map_err(|()| {
                CanonicalError::InvalidRequest(format!(
                    "Invalid {request_name} body: model must be a string"
                ))
            })?;

            let stream = match ranges.stream.as_ref() {
                Some(range) => parse_optional_bool_token(body_bytes, range).map_err(|()| {
                    CanonicalError::InvalidRequest(format!(
                        "Invalid {request_name} body: stream must be a boolean or null"
                    ))
                })?,
                None => None,
            };

            let has_tools = ranges
                .tools
                .as_ref()
                .is_some_and(|range| raw_tools_token_has_items(&body[range.start..range.end]));

            return Ok(CommonRequestProbe {
                model,
                stream,
                has_tools,
                ranges: Some(ranges),
            });
        }
    }

    if let Ok(mut ranges) = find_common_probe_field_ranges(body_bytes) {
        if let Some(model_range) = ranges.model.as_ref() {
            ranges.body_key_hash = body_key_hash;
            probe_ranges_cache_insert(body_bytes, body_key_hash, &ranges);
            let model = parse_model_token(body_bytes, model_range).map_err(|()| {
                CanonicalError::InvalidRequest(format!(
                    "Invalid {request_name} body: model must be a string"
                ))
            })?;

            let stream = match ranges.stream.as_ref() {
                Some(range) => parse_optional_bool_token(body_bytes, range).map_err(|()| {
                    CanonicalError::InvalidRequest(format!(
                        "Invalid {request_name} body: stream must be a boolean or null"
                    ))
                })?,
                None => None,
            };

            let has_tools = ranges
                .tools
                .as_ref()
                .is_some_and(|range| raw_tools_token_has_items(&body[range.start..range.end]));

            return Ok(CommonRequestProbe {
                model,
                stream,
                has_tools,
                ranges: Some(ranges),
            });
        }
    }

    let slow: SlowCommonProbe = serde_json::from_slice(body)
        .map_err(|e| CanonicalError::InvalidRequest(format!("Invalid {request_name} body: {e}")))?;

    let has_tools = slow.tools.as_ref().is_some_and(raw_tools_value_has_items);
    Ok(CommonRequestProbe {
        model: Cow::Owned(slow.model),
        stream: slow.stream,
        has_tools,
        ranges: None,
    })
}

fn try_parse_probe_without_stream_and_tools(bytes: &[u8]) -> Option<CommonRequestProbe<'_>> {
    if STREAM_FIELD_FINDER.find(bytes).is_some() || TOOLS_FIELD_FINDER.find(bytes).is_some() {
        return None;
    }

    let mut i = skip_ws(bytes, 0);
    if bytes.get(i) != Some(&b'{') {
        return None;
    }
    i = skip_ws(bytes, i + 1);
    if bytes.get(i) != Some(&b'"') {
        return None;
    }

    let key_end = parse_json_string_end(bytes, i).ok()?;
    if bytes.get(i + 1..key_end - 1)? != b"model" {
        return None;
    }

    i = skip_ws(bytes, key_end);
    if bytes.get(i) != Some(&b':') {
        return None;
    }
    i = skip_ws(bytes, i + 1);
    let model_end = parse_json_value_end(bytes, i).ok()?;
    let model_range = i..model_end;
    let model = parse_model_token(bytes, &model_range).ok()?;

    Some(CommonRequestProbe {
        model,
        stream: None,
        has_tools: false,
        ranges: Some(CommonProbeRanges {
            model: Some(model_range),
            ..CommonProbeRanges::default()
        }),
    })
}

fn raw_tools_value_has_items(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Array(items) => !items.is_empty(),
        _ => true,
    }
}

pub(crate) fn raw_tools_token_has_items(token: &[u8]) -> bool {
    if token.first() != Some(&b'[') || token.last() != Some(&b']') {
        return true;
    }
    !token[1..token.len() - 1]
        .iter()
        .all(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
}

pub(crate) fn raw_tools_field_has_items(tools: Option<&serde_json::value::RawValue>) -> bool {
    tools.is_some_and(|raw| raw_tools_token_has_items(raw.get().as_bytes()))
}

fn parse_model_token<'a>(
    bytes: &'a [u8],
    range: &std::ops::Range<usize>,
) -> Result<Cow<'a, str>, ()> {
    let token = bytes.get(range.start..range.end).ok_or(())?;
    if token.len() < 2 || token.first() != Some(&b'"') || token.last() != Some(&b'"') {
        return Err(());
    }
    let inner = &token[1..token.len() - 1];
    if !inner.contains(&b'\\') {
        return std::str::from_utf8(inner)
            .map(Cow::Borrowed)
            .map_err(|_| ());
    }
    serde_json::from_slice::<String>(token)
        .map(Cow::Owned)
        .map_err(|_| ())
}

pub(crate) fn parse_optional_bool_token(
    bytes: &[u8],
    range: &std::ops::Range<usize>,
) -> Result<Option<bool>, ()> {
    let token = bytes.get(range.start..range.end).ok_or(())?;
    match token {
        b"true" => Ok(Some(true)),
        b"false" => Ok(Some(false)),
        b"null" => Ok(None),
        _ => Err(()),
    }
}

#[derive(Default, Clone)]
pub(crate) struct CommonProbeRanges {
    pub(crate) model: Option<std::ops::Range<usize>>,
    pub(crate) stream: Option<std::ops::Range<usize>>,
    pub(crate) tools: Option<std::ops::Range<usize>>,
    pub(crate) tool_choice: Option<std::ops::Range<usize>>,
    pub(crate) messages: Option<std::ops::Range<usize>>,
    pub(crate) body_key_hash: Option<u64>,
}

impl CommonProbeRanges {
    #[must_use]
    pub(crate) fn messages_range(&self) -> Option<&std::ops::Range<usize>> {
        self.messages.as_ref()
    }
}

pub(crate) fn find_common_probe_field_ranges(bytes: &[u8]) -> Result<CommonProbeRanges, ()> {
    let mut i = skip_ws(bytes, 0);
    if bytes.get(i) != Some(&b'{') {
        return Err(());
    }
    i += 1;

    let mut ranges = CommonProbeRanges::default();
    loop {
        i = skip_ws(bytes, i);
        match bytes.get(i) {
            Some(b'}') => return Ok(ranges),
            Some(b'"') => {}
            _ => return Err(()),
        }

        let key_start = i + 1;
        let key_end = parse_json_string_end(bytes, i)?;
        let key = &bytes[key_start..key_end - 1];

        i = skip_ws(bytes, key_end);
        if bytes.get(i) != Some(&b':') {
            return Err(());
        }
        i = skip_ws(bytes, i + 1);

        let value_start = i;
        let value_end = parse_json_value_end(bytes, i)?;
        let value_range = value_start..value_end;

        match key {
            b"model" => ranges.model = Some(value_range),
            b"stream" => ranges.stream = Some(value_range),
            b"tools" => ranges.tools = Some(value_range),
            b"tool_choice" => ranges.tool_choice = Some(value_range),
            b"messages" => ranges.messages = Some(value_range),
            _ => {}
        }
        i = skip_ws(bytes, value_end);

        match bytes.get(i) {
            Some(b',') => i += 1,
            Some(b'}') => return Ok(ranges),
            _ => return Err(()),
        }
    }
}

#[inline]
fn rewrite_model_token_range(
    body: &bytes::Bytes,
    model_value_range: &std::ops::Range<usize>,
    actual_model: &str,
) -> Result<bytes::Bytes, CanonicalError> {
    let replaced_len = model_value_range.end - model_value_range.start;
    if !model_needs_json_escape(actual_model) {
        let model_bytes = actual_model.as_bytes();
        let mut out = Vec::with_capacity(body.len() - replaced_len + model_bytes.len() + 2);
        out.extend_from_slice(&body[..model_value_range.start]);
        out.push(b'"');
        out.extend_from_slice(model_bytes);
        out.push(b'"');
        out.extend_from_slice(&body[model_value_range.end..]);
        return Ok(out.into());
    }

    let quoted_model = serde_json::to_vec(actual_model)
        .map_err(|e| CanonicalError::Transport(format!("Failed to serialize body: {e}")))?;
    let mut out = Vec::with_capacity(body.len() - replaced_len + quoted_model.len());
    out.extend_from_slice(&body[..model_value_range.start]);
    out.extend_from_slice(&quoted_model);
    out.extend_from_slice(&body[model_value_range.end..]);
    Ok(out.into())
}

#[inline]
fn model_needs_json_escape(actual_model: &str) -> bool {
    actual_model
        .as_bytes()
        .iter()
        .any(|&byte| byte < 0x20 || byte == b'"' || byte == b'\\')
}

pub(crate) fn rewrite_model_field_in_json_body_with_range(
    body: &bytes::Bytes,
    actual_model: &str,
    request_name: &str,
    model_value_range: Option<&std::ops::Range<usize>>,
) -> Result<bytes::Bytes, CanonicalError> {
    if let Some(model_value_range) = model_value_range {
        return rewrite_model_token_range(body, model_value_range, actual_model);
    }
    if let Ok(Some(model_value_range)) = find_top_level_field_value_range(body.as_ref(), b"model") {
        return rewrite_model_token_range(body, &model_value_range, actual_model);
    }

    // Fallback for edge cases (missing model field, escaped key forms, etc.).
    let mut value: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| CanonicalError::InvalidRequest(format!("Invalid {request_name} body: {e}")))?;
    let Some(obj) = value.as_object_mut() else {
        return Err(CanonicalError::InvalidRequest(format!(
            "{request_name} body must be a JSON object"
        )));
    };
    obj.insert(
        "model".to_string(),
        serde_json::Value::String(actual_model.to_string()),
    );
    serde_json::to_vec(&value)
        .map(bytes::Bytes::from)
        .map_err(|e| CanonicalError::Transport(format!("Failed to serialize body: {e}")))
}
