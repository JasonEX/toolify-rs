use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::CanonicalError;

static CALL_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
const HEX: &[u8; 16] = b"0123456789abcdef";
const SAMPLED_HASH_SAMPLE_LEN: usize = 32;

#[inline]
pub(crate) fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[inline]
pub(crate) fn sampled_bytes_hash(bytes: &[u8]) -> u64 {
    const HASH_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

    #[inline]
    fn read_u64_lossy(bytes: &[u8]) -> u64 {
        let mut buf = [0u8; 8];
        let copy_len = bytes.len().min(8);
        buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
        u64::from_le_bytes(buf)
    }

    let len = bytes.len();
    let mut hash = mix_u64((len as u64) ^ HASH_SEED);
    if len == 0 {
        return hash;
    }

    hash ^= mix_u64(read_u64_lossy(bytes));

    if len > 8 {
        hash ^= mix_u64(read_u64_lossy(&bytes[len - 8..]));
    }

    if len > 16 {
        let mid = len / 2;
        let mid_start = mid.saturating_sub(4);
        let mid_end = (mid_start + 8).min(len);
        hash ^= mix_u64(read_u64_lossy(&bytes[mid_start..mid_end]));
    }

    if len > SAMPLED_HASH_SAMPLE_LEN * 2 {
        let quarter = len / 4;
        let q3 = len.saturating_sub(quarter + 8);
        hash ^= mix_u64(read_u64_lossy(&bytes[quarter..(quarter + 8).min(len)]));
        hash ^= mix_u64(read_u64_lossy(&bytes[q3..(q3 + 8).min(len)]));
    }

    mix_u64(hash)
}

#[inline]
pub(crate) fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[inline]
pub(crate) fn next_call_id() -> String {
    let id = CALL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut out = String::with_capacity(21);
    out.push_str("call_");
    push_u64_hex_16(&mut out, id);
    out
}

pub(crate) fn next_generated_id(prefix: &str, counter: &AtomicU64) -> String {
    let id = counter.fetch_add(1, Ordering::Relaxed);
    let mut out = String::with_capacity(prefix.len() + 17);
    out.push_str(prefix);
    out.push('-');
    push_u64_hex_16(&mut out, id);
    out
}

#[inline]
pub(crate) fn format_request_seq_hex(prefix: &str, request_seq: u64) -> String {
    let mut out = String::with_capacity(prefix.len() + 16);
    out.push_str(prefix);
    push_u64_hex_16(&mut out, request_seq);
    out
}

#[inline]
pub(crate) fn extract_sse_data_payload(
    line: &str,
    allow_data_no_space: bool,
    allow_bare_json: bool,
    ignore_event_lines: bool,
) -> Option<&str> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with(':') {
        return None;
    }
    if ignore_event_lines && trimmed.starts_with("event:") {
        return None;
    }

    let payload = if let Some(rest) = trimmed.strip_prefix("data: ") {
        rest
    } else if allow_data_no_space {
        trimmed.strip_prefix("data:")?
    } else if allow_bare_json {
        trimmed
    } else {
        return None;
    };

    let payload = payload.trim();
    if payload == "[DONE]" {
        return None;
    }
    Some(payload)
}

#[inline]
pub(crate) fn parse_sse_data_json_line<T>(
    line: &str,
    allow_data_no_space: bool,
    allow_bare_json: bool,
    ignore_event_lines: bool,
) -> Option<T>
where
    T: serde::de::DeserializeOwned,
{
    let payload = extract_sse_data_payload(
        line,
        allow_data_no_space,
        allow_bare_json,
        ignore_event_lines,
    )?;
    serde_json::from_str(payload).ok()
}

#[inline]
pub(crate) fn push_json_string_escaped(out: &mut String, value: &str) {
    let bytes = value.as_bytes();
    if bytes.iter().all(|&b| b >= 0x20 && b != b'"' && b != b'\\') {
        out.push('"');
        out.push_str(value);
        out.push('"');
        return;
    }

    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if c <= '\u{1f}' => {
                let control = c as u8;
                out.push_str("\\u00");
                out.push(char::from(HEX[(control >> 4) as usize]));
                out.push(char::from(HEX[(control & 0x0f) as usize]));
            }
            _ => out.push(ch),
        }
    }
    out.push('"');
}

#[inline]
pub(crate) fn push_u64_decimal(out: &mut String, mut n: u64) {
    if n == 0 {
        out.push('0');
        return;
    }

    let mut buf = [0u8; 20];
    let mut i = buf.len();
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + ((n % 10) as u8);
        n /= 10;
    }
    let digits = std::str::from_utf8(&buf[i..]).unwrap_or("0");
    out.push_str(digits);
}

#[inline]
pub(crate) fn push_usize_decimal(out: &mut String, n: usize) {
    push_u64_decimal(out, n as u64);
}

#[inline]
fn push_u64_hex_16(out: &mut String, mut value: u64) {
    let mut buf = [b'0'; 16];
    let mut idx = 16;
    while idx > 0 {
        idx -= 1;
        let nibble = usize::try_from(value & 0x0f).unwrap_or(0);
        buf[idx] = HEX[nibble];
        value >>= 4;
    }
    for byte in buf {
        out.push(char::from(byte));
    }
}

pub(crate) fn raw_value_from_string(
    json: String,
    context: &'static str,
) -> Result<Box<serde_json::value::RawValue>, CanonicalError> {
    serde_json::value::RawValue::from_string(json).map_err(|e| {
        CanonicalError::Translation(format!(
            "Failed to convert {context} arguments to RawValue: {e}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{format_request_seq_hex, push_json_string_escaped};

    #[test]
    fn format_request_seq_hex_matches_formatter() {
        assert_eq!(
            format_request_seq_hex("chatcmpl-", 0x1234_abcd_u64),
            "chatcmpl-000000001234abcd"
        );
        assert_eq!(
            format_request_seq_hex("msg_", u64::MAX),
            "msg_ffffffffffffffff"
        );
    }

    #[test]
    fn push_json_string_escaped_matches_serde_json() {
        let inputs = [
            "",
            "plain ascii",
            "quote \" and slash \\",
            "line\nbreak\r\n",
            "\u{08}\u{0c}\t",
            "control \u{001f} tail",
            "emoji 😀 café",
            "mix \"😀\\\n\t\r\u{0000}",
        ];

        for input in inputs {
            let mut out = String::new();
            push_json_string_escaped(&mut out, input);
            let expected = serde_json::to_string(input).expect("serialize");
            assert_eq!(out, expected);
        }
    }
}
