use http::HeaderMap;
use toolify_rs::auth::{authenticate, build_allowed_key_set};
use toolify_rs::config::{AppConfig, ClientAuthConfig, FeaturesConfig, ServerConfig};
use toolify_rs::error::CanonicalError;
use toolify_rs::protocol::canonical::IngressApi;

fn config_with_keys(keys: Vec<&str>) -> AppConfig {
    AppConfig {
        server: ServerConfig::default(),
        upstream_services: Vec::new(),
        client_authentication: ClientAuthConfig {
            allowed_keys: keys.into_iter().map(ToString::to_string).collect(),
        },
        features: FeaturesConfig::default(),
    }
}

#[test]
fn test_auth_openai_success() {
    let allowed = build_allowed_key_set(&config_with_keys(vec!["client-key"]));
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        "Bearer client-key".parse().expect("header"),
    );
    assert!(authenticate(IngressApi::OpenAiChat, &headers, &allowed).is_ok());
}

#[test]
fn test_auth_gemini_fallback_bearer_success() {
    let allowed = build_allowed_key_set(&config_with_keys(vec!["g-key"]));
    let mut headers = HeaderMap::new();
    headers.insert("authorization", "Bearer g-key".parse().expect("header"));
    assert!(authenticate(IngressApi::Gemini, &headers, &allowed).is_ok());
}

#[test]
fn test_auth_missing_key_is_error() {
    let allowed = build_allowed_key_set(&config_with_keys(vec!["client-key"]));
    let headers = HeaderMap::new();
    let err =
        authenticate(IngressApi::OpenAiChat, &headers, &allowed).expect_err("auth should fail");
    assert!(matches!(err, CanonicalError::Auth(_)));
}
