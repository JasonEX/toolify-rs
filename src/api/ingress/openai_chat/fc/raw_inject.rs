use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};

use crate::api::common::{
    find_common_probe_field_ranges, parse_json_string_end, parse_json_value_end,
    parse_optional_bool_token, raw_tools_token_has_items, skip_ws, CommonProbeRanges,
};
use crate::config::FeaturesConfig;
use crate::error::CanonicalError;
use crate::json_scan::find_top_level_field_value_range;
use crate::protocol::canonical::CanonicalToolSpec;
use crate::util::{mix_u64, sampled_bytes_hash};

use super::cache::resolve_simple_inject_artifacts;
use super::response_format_token_is_json_mode;

pub(crate) type OpenAiSimpleInjectBuild = (
    bytes::Bytes,
    Arc<[crate::protocol::canonical::CanonicalToolSpec]>,
    bool,
);

const SIMPLE_INJECT_BODY_CACHE_SET_COUNT: usize = 8;
const SIMPLE_INJECT_BODY_CACHE_SET_WAYS: usize = 4;
const SIMPLE_INJECT_BODY_CACHE_MAX_BODY_BYTES: usize = 16 * 1024;
const SIMPLE_INJECT_BODY_CACHE_MAX_MODEL_BYTES: usize = 256;

#[derive(Clone)]
struct SimpleInjectBodyCacheEntry {
    key_hash: u64,
    source_body: Arc<[u8]>,
    actual_model: Arc<str>,
    inject_body: bytes::Bytes,
    saved_tools: Arc<[CanonicalToolSpec]>,
    inject_stream: bool,
}

static SIMPLE_INJECT_BODY_CACHE: LazyLock<
    [parking_lot::Mutex<VecDeque<SimpleInjectBodyCacheEntry>>; SIMPLE_INJECT_BODY_CACHE_SET_COUNT],
> = LazyLock::new(|| {
    std::array::from_fn(|_| {
        parking_lot::Mutex::new(VecDeque::with_capacity(SIMPLE_INJECT_BODY_CACHE_SET_WAYS))
    })
});

thread_local! {
    static SIMPLE_INJECT_BODY_LAST_HIT: RefCell<Option<SimpleInjectBodyCacheEntry>> =
        const { RefCell::new(None) };
}

#[inline]
fn simple_inject_body_cacheable(source_body: &[u8], actual_model: &str) -> bool {
    source_body.len() <= SIMPLE_INJECT_BODY_CACHE_MAX_BODY_BYTES
        && actual_model.len() <= SIMPLE_INJECT_BODY_CACHE_MAX_MODEL_BYTES
}

#[inline]
fn simple_inject_body_key_hash(source_body: &[u8], actual_model: &str) -> u64 {
    let body_hash = sampled_bytes_hash(source_body);
    let model_hash = sampled_bytes_hash(actual_model.as_bytes());
    mix_u64(body_hash ^ 0xff_u64 ^ model_hash.rotate_left(17))
}

#[inline]
fn simple_inject_body_set_index(key_hash: u64) -> usize {
    usize::try_from(key_hash & u64::try_from(SIMPLE_INJECT_BODY_CACHE_SET_COUNT - 1).unwrap_or(0))
        .unwrap_or(0)
}

fn simple_inject_body_cache_get(
    source_body: &[u8],
    actual_model: &str,
    key_hash_hint: Option<u64>,
) -> Option<OpenAiSimpleInjectBuild> {
    if !simple_inject_body_cacheable(source_body, actual_model) {
        return None;
    }
    let key_hash =
        key_hash_hint.unwrap_or_else(|| simple_inject_body_key_hash(source_body, actual_model));
    if let Some(cached) = simple_inject_body_thread_cache_get(source_body, actual_model, key_hash) {
        return Some(cached);
    }

    let set = &SIMPLE_INJECT_BODY_CACHE[simple_inject_body_set_index(key_hash)];
    let mut guard = set.lock();
    let pos = guard.iter().rposition(|entry| {
        entry.key_hash == key_hash
            && entry.actual_model.as_ref() == actual_model
            && entry.source_body.as_ref() == source_body
    })?;

    let move_to_back = pos + 1 != guard.len();
    let entry = if move_to_back {
        guard.remove(pos)?
    } else {
        guard.get(pos)?.clone()
    };
    if move_to_back {
        guard.push_back(entry.clone());
    }
    drop(guard);

    simple_inject_body_thread_cache_set(&entry);
    let out = (
        entry.inject_body.clone(),
        Arc::clone(&entry.saved_tools),
        entry.inject_stream,
    );
    Some(out)
}

fn simple_inject_body_cache_insert(
    source_body: &[u8],
    actual_model: &str,
    inject_body: &bytes::Bytes,
    saved_tools: &Arc<[CanonicalToolSpec]>,
    inject_stream: bool,
    key_hash_hint: Option<u64>,
) {
    if !simple_inject_body_cacheable(source_body, actual_model) {
        return;
    }
    let key_hash =
        key_hash_hint.unwrap_or_else(|| simple_inject_body_key_hash(source_body, actual_model));
    let set = &SIMPLE_INJECT_BODY_CACHE[simple_inject_body_set_index(key_hash)];
    let mut guard = set.lock();
    if let Some(pos) = guard.iter().position(|entry| {
        entry.key_hash == key_hash
            && entry.actual_model.as_ref() == actual_model
            && entry.source_body.as_ref() == source_body
    }) {
        let _ = guard.remove(pos);
    }
    if guard.len() >= SIMPLE_INJECT_BODY_CACHE_SET_WAYS {
        let _ = guard.pop_front();
    }
    let entry = SimpleInjectBodyCacheEntry {
        key_hash,
        source_body: Arc::from(source_body),
        actual_model: Arc::from(actual_model),
        inject_body: inject_body.clone(),
        saved_tools: Arc::clone(saved_tools),
        inject_stream,
    };
    simple_inject_body_thread_cache_set(&entry);
    guard.push_back(entry);
}

#[inline]
fn simple_inject_body_thread_cache_get(
    source_body: &[u8],
    actual_model: &str,
    key_hash: u64,
) -> Option<OpenAiSimpleInjectBuild> {
    SIMPLE_INJECT_BODY_LAST_HIT.with(|slot| {
        let guard = slot.borrow();
        let entry = guard.as_ref()?;
        if entry.key_hash != key_hash
            || entry.actual_model.as_ref() != actual_model
            || entry.source_body.as_ref() != source_body
        {
            return None;
        }
        Some((
            entry.inject_body.clone(),
            Arc::clone(&entry.saved_tools),
            entry.inject_stream,
        ))
    })
}

#[inline]
fn simple_inject_body_thread_cache_set(entry: &SimpleInjectBodyCacheEntry) {
    SIMPLE_INJECT_BODY_LAST_HIT.with(|slot| {
        *slot.borrow_mut() = Some(entry.clone());
    });
}

pub(crate) fn try_build_openai_simple_fc_inject_body_from_raw(
    body: &bytes::Bytes,
    actual_model: &str,
    features: &FeaturesConfig,
    ranges_hint: Option<&CommonProbeRanges>,
) -> Result<Option<OpenAiSimpleInjectBuild>, CanonicalError> {
    let body_slice = body.as_ref();
    let body_key_hash = ranges_hint
        .and_then(|ranges| ranges.body_key_hash)
        .or_else(|| {
            simple_inject_body_cacheable(body_slice, actual_model)
                .then(|| simple_inject_body_key_hash(body_slice, actual_model))
        });

    if let Some(cached) = simple_inject_body_cache_get(body_slice, actual_model, body_key_hash) {
        return Ok(Some(cached));
    }

    let owned_ranges;
    let ranges = if let Some(ranges) = ranges_hint {
        ranges
    } else {
        let Ok(parsed_ranges) = find_common_probe_field_ranges(body_slice) else {
            return Ok(None);
        };
        owned_ranges = parsed_ranges;
        &owned_ranges
    };

    let Some(messages_range) = ranges.messages.as_ref() else {
        return Ok(None);
    };
    let Some((inner_start_rel, inner_end_rel)) =
        messages_inner_bounds_if_simple(&body_slice[messages_range.clone()])
    else {
        return Ok(None);
    };
    let messages_inner_bounds = (
        messages_range.start + inner_start_rel,
        messages_range.start + inner_end_rel,
    );

    let Some(tools_range) = ranges.tools.as_ref() else {
        return Ok(None);
    };

    if let Ok(Some(response_format_range)) =
        find_top_level_field_value_range(body_slice, b"response_format")
    {
        let response_format_token = &body_slice[response_format_range];
        if response_format_token_is_json_mode(response_format_token) {
            return Ok(None);
        }
    }

    let tools_token = &body_slice[tools_range.clone()];
    let tool_choice_token = ranges
        .tool_choice
        .as_ref()
        .map(|range| &body_slice[range.clone()][..]);
    let Some((saved_tools, fc_prompt_artifacts)) =
        resolve_simple_inject_artifacts(tools_token, tool_choice_token, features)?
    else {
        return Ok(None);
    };
    let inject_stream = match ranges.stream {
        Some(ref range) => parse_optional_bool_token(body_slice, range).map_err(|()| {
            CanonicalError::InvalidRequest(
                "Invalid OpenAI Chat request body: stream must be a boolean or null".to_string(),
            )
        })?,
        None => None,
    }
    .unwrap_or(false);

    let Some(inject_body) = build_openai_simple_inject_json_body(
        body_slice,
        actual_model,
        fc_prompt_artifacts.openai_system_message_json(),
        Some(messages_inner_bounds),
    )?
    else {
        return Ok(None);
    };

    simple_inject_body_cache_insert(
        body_slice,
        actual_model,
        &inject_body,
        &saved_tools,
        inject_stream,
        body_key_hash,
    );

    Ok(Some((inject_body, saved_tools, inject_stream)))
}

pub(crate) fn build_openai_simple_inject_json_body(
    body: &[u8],
    actual_model: &str,
    system_msg_bytes: &[u8],
    prevalidated_messages_inner_bounds: Option<(usize, usize)>,
) -> Result<Option<bytes::Bytes>, CanonicalError> {
    let mut i = skip_ws(body, 0);
    if body.get(i) != Some(&b'{') {
        return Ok(None);
    }
    i += 1;

    let mut out = Vec::with_capacity(body.len() + system_msg_bytes.len() + actual_model.len() + 20);
    out.push(b'{');

    let mut wrote_any = false;
    let mut saw_model = false;
    let mut saw_messages = false;

    loop {
        i = skip_ws(body, i);
        match body.get(i) {
            Some(b'}') => break,
            Some(b'"') => {}
            _ => return Ok(None),
        }

        let key_start = i;
        let key_end = parse_json_string_end(body, i).map_err(|()| {
            CanonicalError::InvalidRequest("Invalid OpenAI Chat request body".to_string())
        })?;
        let key = &body[key_start + 1..key_end - 1];
        if key.contains(&b'\\') {
            // Escaped key names are rare; fall back to typed transform path.
            return Ok(None);
        }

        i = skip_ws(body, key_end);
        if body.get(i) != Some(&b':') {
            return Ok(None);
        }
        i = skip_ws(body, i + 1);

        let value_start = i;
        let value_end = parse_json_value_end(body, i).map_err(|()| {
            CanonicalError::InvalidRequest("Invalid OpenAI Chat request body".to_string())
        })?;

        let drop_field = matches!(key, b"tools" | b"tool_choice");
        if !drop_field {
            if wrote_any {
                out.push(b',');
            }
            out.extend_from_slice(&body[key_start..key_end]);
            out.push(b':');

            if key == b"model" {
                append_json_string_quoted(&mut out, actual_model);
                saw_model = true;
            } else if key == b"messages" {
                if let Some((inner_start, inner_end)) = prevalidated_messages_inner_bounds {
                    if inner_start >= value_start && inner_end <= value_end {
                        inject_system_message_from_inner_bounds(
                            body,
                            inner_start,
                            inner_end,
                            system_msg_bytes,
                            &mut out,
                        );
                    } else if !inject_system_message_into_messages_out(
                        &body[value_start..value_end],
                        system_msg_bytes,
                        &mut out,
                    ) {
                        return Ok(None);
                    }
                } else if !inject_system_message_into_messages_out(
                    &body[value_start..value_end],
                    system_msg_bytes,
                    &mut out,
                ) {
                    return Ok(None);
                }
                saw_messages = true;
            } else {
                out.extend_from_slice(&body[value_start..value_end]);
            }
            wrote_any = true;
        }

        i = skip_ws(body, value_end);
        match body.get(i) {
            Some(b',') => i += 1,
            Some(b'}') => break,
            _ => return Ok(None),
        }
    }

    if !saw_messages {
        return Ok(None);
    }

    if !saw_model {
        if wrote_any {
            out.push(b',');
        }
        out.extend_from_slice(br#""model":"#);
        append_json_string_quoted(&mut out, actual_model);
    }

    out.push(b'}');
    Ok(Some(out.into()))
}

fn inject_system_message_into_messages_out(
    messages_value: &[u8],
    system_msg_bytes: &[u8],
    out: &mut Vec<u8>,
) -> bool {
    let Some((inner_start, inner_end)) = messages_inner_bounds_if_simple(messages_value) else {
        return false;
    };
    inject_system_message_from_inner_bounds(
        messages_value,
        inner_start,
        inner_end,
        system_msg_bytes,
        out,
    );
    true
}

fn inject_system_message_from_inner_bounds(
    source: &[u8],
    inner_start: usize,
    inner_end: usize,
    system_msg_bytes: &[u8],
    out: &mut Vec<u8>,
) {
    out.push(b'[');
    out.extend_from_slice(system_msg_bytes);
    if inner_start < inner_end {
        out.push(b',');
        out.extend_from_slice(&source[inner_start..inner_end]);
    }
    out.push(b']');
}

fn append_json_string_quoted(out: &mut Vec<u8>, value: &str) {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    out.push(b'"');
    for &byte in value.as_bytes() {
        match byte {
            b'"' => out.extend_from_slice(br#"\""#),
            b'\\' => out.extend_from_slice(br"\\"),
            b'\n' => out.extend_from_slice(br"\n"),
            b'\r' => out.extend_from_slice(br"\r"),
            b'\t' => out.extend_from_slice(br"\t"),
            0x08 => out.extend_from_slice(br"\b"),
            0x0c => out.extend_from_slice(br"\f"),
            0x00..=0x1f => {
                out.extend_from_slice(br"\u00");
                out.push(HEX[(byte >> 4) as usize]);
                out.push(HEX[(byte & 0x0f) as usize]);
            }
            _ => out.push(byte),
        }
    }
    out.push(b'"');
}

pub(crate) fn messages_inner_bounds_if_simple(messages_token: &[u8]) -> Option<(usize, usize)> {
    let start = skip_ws(messages_token, 0);
    if messages_token.get(start) != Some(&b'[') {
        return None;
    }
    let mut i = start + 1;
    let payload_start = i;
    let payload_end;

    loop {
        i = skip_ws(messages_token, i);
        match messages_token.get(i) {
            Some(b']') => {
                payload_end = i;
                i += 1;
                break;
            }
            Some(_) => {}
            None => return None,
        }

        let item_start = i;
        let Ok(item_end) = parse_json_value_end(messages_token, i) else {
            return None;
        };
        if !message_token_is_simple_fc_inject(&messages_token[item_start..item_end]) {
            return None;
        }

        i = skip_ws(messages_token, item_end);
        match messages_token.get(i) {
            Some(b',') => i += 1,
            Some(b']') => {
                payload_end = i;
                i += 1;
                break;
            }
            _ => return None,
        }
    }

    if skip_ws(messages_token, i) != messages_token.len() {
        return None;
    }

    let inner_start = skip_ws(messages_token, payload_start);
    let mut inner_end = payload_end;
    while inner_end > inner_start
        && matches!(messages_token[inner_end - 1], b' ' | b'\n' | b'\r' | b'\t')
    {
        inner_end -= 1;
    }

    Some((inner_start, inner_end))
}

fn message_token_is_simple_fc_inject(message_token: &[u8]) -> bool {
    let mut i = skip_ws(message_token, 0);
    if message_token.get(i) != Some(&b'{') {
        return false;
    }
    i += 1;

    let mut saw_role = false;
    let mut role_is_blocked = false;
    let mut tool_calls_has_items = false;

    loop {
        i = skip_ws(message_token, i);
        match message_token.get(i) {
            Some(b'}') => return saw_role && !role_is_blocked && !tool_calls_has_items,
            Some(b'"') => {}
            _ => return false,
        }

        let key_start = i + 1;
        let Ok(key_end) = parse_json_string_end(message_token, i) else {
            return false;
        };
        let key = &message_token[key_start..key_end - 1];
        if key.contains(&b'\\') {
            return false;
        }

        i = skip_ws(message_token, key_end);
        if message_token.get(i) != Some(&b':') {
            return false;
        }
        i = skip_ws(message_token, i + 1);

        let value_start = i;
        let Ok(value_end) = parse_json_value_end(message_token, i) else {
            return false;
        };
        let value = &message_token[value_start..value_end];

        if key == b"role" {
            saw_role = true;
            let is_system = json_string_token_equals(value, b"system");
            let is_tool = json_string_token_equals(value, b"tool");
            match (is_system, is_tool) {
                (Some(true), _) | (_, Some(true)) => role_is_blocked = true,
                (Some(false), Some(false)) => {}
                _ => return false,
            }
        } else if key == b"tool_calls" {
            tool_calls_has_items = raw_tools_token_has_items(value);
        }

        i = skip_ws(message_token, value_end);
        match message_token.get(i) {
            Some(b',') => i += 1,
            Some(b'}') => return saw_role && !role_is_blocked && !tool_calls_has_items,
            _ => return false,
        }
    }
}

fn json_string_token_equals(token: &[u8], expected: &[u8]) -> Option<bool> {
    if token.len() < 2 || token.first() != Some(&b'"') || token.last() != Some(&b'"') {
        return None;
    }
    let inner = &token[1..token.len() - 1];
    if !inner.contains(&b'\\') {
        return Some(inner == expected);
    }
    let parsed: String = serde_json::from_slice(token).ok()?;
    Some(parsed.as_bytes() == expected)
}
