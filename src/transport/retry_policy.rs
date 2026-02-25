use std::time::{Duration, SystemTime};

use http::header::RETRY_AFTER;

pub(crate) const RETRY_MAX_ATTEMPTS: u32 = 2;
pub(crate) const RETRY_BACKOFF_BASE_MS: u64 = 100;
pub(crate) const RETRY_BACKOFF_MAX_MS: u64 = 1_000;
pub(crate) const RETRY_AFTER_MAX_SECS: u64 = 30;
pub(crate) const PARSED_ENDPOINT_CACHE_MAX_ENTRIES: usize = 512;
const RETRY_TRANSPORT_FAST_SECOND_MS: u64 = 10;

#[inline]
pub(crate) fn should_retry_upstream_status(status: http::StatusCode) -> bool {
    matches!(status.as_u16(), 429 | 503 | 529)
}

#[inline]
pub(crate) fn should_retry_transport_message(message: &str) -> bool {
    const NEEDLES: [&[u8]; 9] = [
        b"timed out",
        b"timeout",
        b"connection reset",
        b"connection aborted",
        b"broken pipe",
        b"http2 error",
        b"connection refused",
        b"unexpected eof",
        b"stream closed",
    ];
    let haystack = message.as_bytes();
    NEEDLES
        .iter()
        .any(|needle| contains_ascii_case_insensitive(haystack, needle))
}

#[inline]
pub(crate) fn retry_transport_delay(message: &str, attempt: u32) -> Duration {
    if has_fast_retry_transport_signature(message) {
        return if attempt == 0 {
            Duration::ZERO
        } else {
            Duration::from_millis(RETRY_TRANSPORT_FAST_SECOND_MS)
        };
    }
    retry_backoff_delay(attempt)
}

#[inline]
fn has_fast_retry_transport_signature(message: &str) -> bool {
    const FAST_RETRY_NEEDLES: [&[u8]; 6] = [
        b"connection reset",
        b"connection aborted",
        b"broken pipe",
        b"http2 error",
        b"unexpected eof",
        b"stream closed",
    ];
    let haystack = message.as_bytes();
    FAST_RETRY_NEEDLES
        .iter()
        .any(|needle| contains_ascii_case_insensitive(haystack, needle))
}

#[inline]
fn contains_ascii_case_insensitive(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if haystack.len() < needle.len() {
        return false;
    }

    haystack.windows(needle.len()).any(|window| {
        window
            .iter()
            .zip(needle.iter())
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
    })
}

#[inline]
pub(crate) fn retry_backoff_delay(attempt: u32) -> Duration {
    let shift = attempt.min(10);
    let multiplier = 1_u64 << shift;
    Duration::from_millis(
        RETRY_BACKOFF_BASE_MS
            .saturating_mul(multiplier)
            .min(RETRY_BACKOFF_MAX_MS),
    )
}

#[inline]
pub(crate) fn retry_delay(headers: &http::HeaderMap, attempt: u32) -> Duration {
    parse_retry_after_delay(headers).unwrap_or_else(|| retry_backoff_delay(attempt))
}

#[inline]
pub(crate) fn parse_retry_after_delay(headers: &http::HeaderMap) -> Option<Duration> {
    let raw = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    if raw.is_empty() {
        return None;
    }

    if let Ok(seconds) = raw.parse::<u64>() {
        return Some(Duration::from_secs(seconds.min(RETRY_AFTER_MAX_SECS)));
    }

    let target = httpdate::parse_http_date(raw).ok()?;
    let delay = target.duration_since(SystemTime::now()).unwrap_or_default();
    Some(delay.min(Duration::from_secs(RETRY_AFTER_MAX_SECS)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_retry_upstream_status() {
        assert!(should_retry_upstream_status(
            http::StatusCode::TOO_MANY_REQUESTS
        ));
        assert!(should_retry_upstream_status(
            http::StatusCode::SERVICE_UNAVAILABLE
        ));
        assert!(should_retry_upstream_status(
            http::StatusCode::from_u16(529).unwrap()
        ));
        assert!(!should_retry_upstream_status(http::StatusCode::BAD_REQUEST));
    }

    #[test]
    fn test_parse_retry_after_seconds() {
        let mut headers = http::HeaderMap::new();
        headers.insert(RETRY_AFTER, http::HeaderValue::from_static("5"));
        let delay = parse_retry_after_delay(&headers).unwrap();
        assert_eq!(delay, Duration::from_secs(5));
    }

    #[test]
    fn test_parse_retry_after_http_date() {
        let target = SystemTime::now() + Duration::from_secs(2);
        let mut headers = http::HeaderMap::new();
        headers.insert(
            RETRY_AFTER,
            http::HeaderValue::from_str(&httpdate::fmt_http_date(target)).unwrap(),
        );
        let delay = parse_retry_after_delay(&headers).unwrap();
        assert!(delay <= Duration::from_secs(RETRY_AFTER_MAX_SECS));
    }

    #[test]
    fn test_parse_retry_after_invalid() {
        let mut headers = http::HeaderMap::new();
        headers.insert(RETRY_AFTER, http::HeaderValue::from_static("not-a-delay"));
        assert!(parse_retry_after_delay(&headers).is_none());
    }

    #[test]
    fn test_retry_transport_delay_fast_path() {
        assert_eq!(
            retry_transport_delay("connection reset by peer", 0),
            Duration::ZERO
        );
        assert_eq!(
            retry_transport_delay("unexpected EOF while reading", 1),
            Duration::from_millis(RETRY_TRANSPORT_FAST_SECOND_MS)
        );
    }

    #[test]
    fn test_retry_transport_delay_regular_backoff() {
        assert_eq!(
            retry_transport_delay("timed out waiting for response", 0),
            retry_backoff_delay(0)
        );
    }

    #[test]
    fn test_should_retry_transport_message_h2_error() {
        assert!(should_retry_transport_message(
            "upstream failed with HTTP2 error: stream closed"
        ));
    }

    #[test]
    fn test_retry_transport_delay_h2_error_fast_path() {
        assert_eq!(
            retry_transport_delay("HTTP2 Error while reading frame", 0),
            Duration::ZERO
        );
        assert_eq!(
            retry_transport_delay("HTTP2 Error while reading frame", 1),
            Duration::from_millis(RETRY_TRANSPORT_FAST_SECOND_MS)
        );
    }
}
