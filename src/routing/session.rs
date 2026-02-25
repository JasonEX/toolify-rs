use crate::json_scan::{find_top_level_field_value_range, skip_ws};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionClass {
    Portable,
    Anchored,
}

/// Extract a compact prompt-like byte prefix for sticky routing.
///
/// Preference order:
/// 1. Parsed top-level `messages` range (from fast probe).
/// 2. Top-level `input` / `messages` / `contents` / `prompt`.
/// 3. Entire body fallback.
#[must_use]
pub fn route_prompt_prefix_bytes<'a>(
    body: &'a [u8],
    messages_range: Option<&std::ops::Range<usize>>,
) -> &'a [u8] {
    const ROUTE_STICKY_PREFIX_MAX_BYTES: usize = 256;

    fn trim_and_cap_prefix(token: &[u8]) -> &[u8] {
        const ROUTE_STICKY_PREFIX_MAX_BYTES: usize = 256;
        let start = skip_ws(token, 0);
        let end = start
            .saturating_add(ROUTE_STICKY_PREFIX_MAX_BYTES)
            .min(token.len());
        &token[start..end]
    }

    if let Some(range) = messages_range {
        if let Some(token) = body.get(range.clone()) {
            return trim_and_cap_prefix(token);
        }
    }

    for field in [
        b"input".as_slice(),
        b"messages".as_slice(),
        b"contents".as_slice(),
        b"prompt".as_slice(),
    ] {
        if let Ok(Some(range)) = find_top_level_field_value_range(body, field) {
            if let Some(token) = body.get(range) {
                return trim_and_cap_prefix(token);
            }
        }
    }

    if body.len() <= ROUTE_STICKY_PREFIX_MAX_BYTES {
        return trim_and_cap_prefix(body);
    }
    trim_and_cap_prefix(&body[..ROUTE_STICKY_PREFIX_MAX_BYTES])
}

#[must_use]
pub fn classify_session_class(body: &[u8], has_messages_range: bool) -> SessionClass {
    for field in [
        b"previous_response_id".as_slice(),
        b"thread_id".as_slice(),
        b"session_id".as_slice(),
        b"conversation_id".as_slice(),
    ] {
        if top_level_field_non_null(body, field) {
            return SessionClass::Anchored;
        }
    }

    if has_messages_range
        || top_level_field_non_null(body, b"messages")
        || top_level_field_non_null(body, b"input")
        || top_level_field_non_null(body, b"contents")
    {
        return SessionClass::Portable;
    }

    SessionClass::Portable
}

fn top_level_field_non_null(body: &[u8], field: &[u8]) -> bool {
    if let Ok(Some(range)) = find_top_level_field_value_range(body, field) {
        return body
            .get(range)
            .is_some_and(|token| !token.eq_ignore_ascii_case(b"null"));
    }
    false
}
