use crate::config::AppConfig;
use crate::error::CanonicalError;
use crate::protocol::canonical::IngressApi;
use http::header::{HeaderName, AUTHORIZATION};
use rustc_hash::FxHashSet;

const X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");
const X_GOOG_API_KEY: HeaderName = HeaderName::from_static("x-goog-api-key");

/// Compact key index used in hot-path authentication.
pub enum AllowedClientKeys {
    Empty,
    Single { raw: Box<str>, bearer: Box<str> },
    Multiple(FxHashSet<String>),
}

/// Extract the API key from request headers, based on the ingress API convention.
///
/// - `OpenAiChat` / `OpenAiResponses`: `Authorization: Bearer <key>`
/// - Anthropic: `x-api-key: <key>`
/// - Gemini: `x-goog-api-key: <key>` first, then fall back to `Authorization: Bearer <key>`
///
/// # Errors
///
/// Returns `CanonicalError::Auth` when no expected API key header is present.
pub fn extract_api_key(
    ingress: IngressApi,
    headers: &http::HeaderMap,
) -> Result<&str, CanonicalError> {
    let key = match ingress {
        IngressApi::Anthropic => headers.get(X_API_KEY).and_then(|v| v.to_str().ok()),
        IngressApi::Gemini => headers
            .get(X_GOOG_API_KEY)
            .and_then(|v| v.to_str().ok())
            .or_else(|| {
                headers
                    .get(AUTHORIZATION)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.strip_prefix("Bearer "))
            }),
        IngressApi::OpenAiChat | IngressApi::OpenAiResponses => headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer ")),
    };

    key.ok_or_else(|| CanonicalError::Auth("Missing API key".to_string()))
}

#[inline]
fn strip_bearer_prefix_bytes(value: &http::HeaderValue) -> Option<&[u8]> {
    value.as_bytes().strip_prefix(b"Bearer ")
}

/// Extract API key bytes for routing hash only.
///
/// This avoids UTF-8 validation work when the caller only needs raw bytes
/// as hash input.
#[must_use]
pub(crate) fn extract_api_key_bytes_for_hash(
    ingress: IngressApi,
    headers: &http::HeaderMap,
) -> Option<&[u8]> {
    match ingress {
        IngressApi::Anthropic => {
            if let Some(value) = headers.get(X_API_KEY) {
                Some(value.as_bytes())
            } else {
                None
            }
        }
        IngressApi::Gemini => {
            if let Some(value) = headers.get(X_GOOG_API_KEY) {
                return Some(value.as_bytes());
            }
            if let Some(value) = headers.get(AUTHORIZATION) {
                return strip_bearer_prefix_bytes(value);
            }
            None
        }
        IngressApi::OpenAiChat | IngressApi::OpenAiResponses => {
            if let Some(value) = headers.get(AUTHORIZATION) {
                strip_bearer_prefix_bytes(value)
            } else {
                None
            }
        }
    }
}

/// Authenticate an incoming request by checking the extracted key against
/// pre-indexed `allowed_keys`.
///
/// # Errors
///
/// Returns `CanonicalError::Auth` when the API key is missing or invalid.
pub fn authenticate(
    ingress: IngressApi,
    headers: &http::HeaderMap,
    allowed_keys: &AllowedClientKeys,
) -> Result<(), CanonicalError> {
    match allowed_keys {
        AllowedClientKeys::Single { raw, bearer } => {
            authenticate_single_key(ingress, headers, raw.as_ref(), bearer.as_ref())
        }
        AllowedClientKeys::Multiple(allowed_set) => {
            let client_key = extract_api_key(ingress, headers)?;
            if allowed_set.contains(client_key) {
                Ok(())
            } else {
                Err(CanonicalError::Auth("Invalid API key".to_string()))
            }
        }
        AllowedClientKeys::Empty => Err(CanonicalError::Auth("Invalid API key".to_string())),
    }
}

fn authenticate_single_key(
    ingress: IngressApi,
    headers: &http::HeaderMap,
    raw_key: &str,
    bearer_key: &str,
) -> Result<(), CanonicalError> {
    let raw_key_bytes = raw_key.as_bytes();
    let bearer_key_bytes = bearer_key.as_bytes();
    match ingress {
        IngressApi::Anthropic => match headers.get(X_API_KEY) {
            Some(value) if value.as_bytes() == raw_key_bytes => Ok(()),
            Some(_) => Err(CanonicalError::Auth("Invalid API key".to_string())),
            None => Err(CanonicalError::Auth("Missing API key".to_string())),
        },
        IngressApi::OpenAiChat | IngressApi::OpenAiResponses => match headers.get(AUTHORIZATION) {
            Some(value) if value.as_bytes() == bearer_key_bytes => Ok(()),
            Some(_) => Err(CanonicalError::Auth("Invalid API key".to_string())),
            None => Err(CanonicalError::Auth("Missing API key".to_string())),
        },
        IngressApi::Gemini => {
            if let Some(value) = headers.get(X_GOOG_API_KEY) {
                return if value.as_bytes() == raw_key_bytes {
                    Ok(())
                } else {
                    Err(CanonicalError::Auth("Invalid API key".to_string()))
                };
            }
            match headers.get(AUTHORIZATION) {
                Some(value) if value.as_bytes() == bearer_key_bytes => Ok(()),
                Some(_) => Err(CanonicalError::Auth("Invalid API key".to_string())),
                None => Err(CanonicalError::Auth("Missing API key".to_string())),
            }
        }
    }
}

/// Build a hash-set index for allowed client keys.
#[must_use]
pub fn build_allowed_key_set(config: &AppConfig) -> AllowedClientKeys {
    let mut allowed_set: FxHashSet<String> = config
        .client_authentication
        .allowed_keys
        .iter()
        .cloned()
        .collect();

    match allowed_set.len() {
        0 => AllowedClientKeys::Empty,
        1 => match allowed_set.drain().next() {
            Some(single_key) => AllowedClientKeys::Single {
                bearer: format!("Bearer {single_key}").into_boxed_str(),
                raw: single_key.into_boxed_str(),
            },
            None => AllowedClientKeys::Empty,
        },
        _ => AllowedClientKeys::Multiple(allowed_set),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AppConfig, ClientAuthConfig, FeaturesConfig, ServerConfig};

    fn make_config(allowed_keys: Vec<String>) -> AppConfig {
        AppConfig {
            server: ServerConfig::default(),
            upstream_services: vec![],
            client_authentication: ClientAuthConfig { allowed_keys },
            features: FeaturesConfig::default(),
        }
    }

    #[test]
    fn test_extract_openai_bearer() {
        let mut headers = http::HeaderMap::new();
        headers.insert("authorization", "Bearer sk-test123".parse().unwrap());
        let key = extract_api_key(IngressApi::OpenAiChat, &headers).unwrap();
        assert_eq!(key, "sk-test123");
    }

    #[test]
    fn test_extract_openai_responses_bearer() {
        let mut headers = http::HeaderMap::new();
        headers.insert("authorization", "Bearer sk-resp456".parse().unwrap());
        let key = extract_api_key(IngressApi::OpenAiResponses, &headers).unwrap();
        assert_eq!(key, "sk-resp456");
    }

    #[test]
    fn test_extract_anthropic_api_key() {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-api-key", "ant-key789".parse().unwrap());
        let key = extract_api_key(IngressApi::Anthropic, &headers).unwrap();
        assert_eq!(key, "ant-key789");
    }

    #[test]
    fn test_extract_gemini_x_goog_api_key() {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-goog-api-key", "gem-key000".parse().unwrap());
        let key = extract_api_key(IngressApi::Gemini, &headers).unwrap();
        assert_eq!(key, "gem-key000");
    }

    #[test]
    fn test_extract_gemini_fallback_bearer() {
        let mut headers = http::HeaderMap::new();
        headers.insert("authorization", "Bearer gem-bearer".parse().unwrap());
        let key = extract_api_key(IngressApi::Gemini, &headers).unwrap();
        assert_eq!(key, "gem-bearer");
    }

    #[test]
    fn test_extract_gemini_prefers_x_goog() {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-goog-api-key", "preferred".parse().unwrap());
        headers.insert("authorization", "Bearer fallback".parse().unwrap());
        let key = extract_api_key(IngressApi::Gemini, &headers).unwrap();
        assert_eq!(key, "preferred");
    }

    #[test]
    fn test_extract_missing_key() {
        let headers = http::HeaderMap::new();
        let err = extract_api_key(IngressApi::OpenAiChat, &headers).unwrap_err();
        assert!(matches!(err, CanonicalError::Auth(_)));
    }

    #[test]
    fn test_extract_api_key_bytes_for_hash_openai_bearer() {
        let mut headers = http::HeaderMap::new();
        headers.insert("authorization", "Bearer sk-test123".parse().unwrap());
        let key = extract_api_key_bytes_for_hash(IngressApi::OpenAiChat, &headers).unwrap();
        assert_eq!(key, b"sk-test123");
    }

    #[test]
    fn test_extract_api_key_bytes_for_hash_gemini_prefers_x_goog() {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-goog-api-key", "preferred".parse().unwrap());
        headers.insert("authorization", "Bearer fallback".parse().unwrap());
        let key = extract_api_key_bytes_for_hash(IngressApi::Gemini, &headers).unwrap();
        assert_eq!(key, b"preferred");
    }

    #[test]
    fn test_authenticate_valid_key() {
        let config = make_config(vec!["valid-key".to_string()]);
        let key_index = build_allowed_key_set(&config);
        let mut headers = http::HeaderMap::new();
        headers.insert("authorization", "Bearer valid-key".parse().unwrap());
        assert!(authenticate(IngressApi::OpenAiChat, &headers, &key_index).is_ok());
    }

    #[test]
    fn test_authenticate_invalid_key() {
        let config = make_config(vec!["valid-key".to_string()]);
        let key_index = build_allowed_key_set(&config);
        let mut headers = http::HeaderMap::new();
        headers.insert("authorization", "Bearer wrong-key".parse().unwrap());
        let err = authenticate(IngressApi::OpenAiChat, &headers, &key_index).unwrap_err();
        assert!(matches!(err, CanonicalError::Auth(_)));
    }

    #[test]
    fn test_build_allowed_key_set_multiple() {
        let config = make_config(vec!["a".to_string(), "b".to_string(), "a".to_string()]);
        let index = build_allowed_key_set(&config);
        match index {
            AllowedClientKeys::Multiple(set) => {
                assert!(set.contains("a"));
                assert!(set.contains("b"));
                assert_eq!(set.len(), 2);
            }
            _ => panic!("expected multiple-key index"),
        }
    }

    #[test]
    fn test_build_allowed_key_set_single() {
        let config = make_config(vec!["single".to_string(), "single".to_string()]);
        let index = build_allowed_key_set(&config);
        match index {
            AllowedClientKeys::Single { raw, bearer } => {
                assert_eq!(raw.as_ref(), "single");
                assert_eq!(bearer.as_ref(), "Bearer single");
            }
            _ => panic!("expected single-key index"),
        }
    }

    #[test]
    fn test_build_allowed_key_set_empty() {
        let config = make_config(vec![]);
        let index = build_allowed_key_set(&config);
        assert!(matches!(index, AllowedClientKeys::Empty));
    }
}
