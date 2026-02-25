use std::sync::Arc;

use toolify_rs::auth::build_allowed_key_set;
use toolify_rs::config::{
    AppConfig, ClientAuthConfig, FcMode, FeaturesConfig, ServerConfig, UpstreamServiceConfig,
};
use toolify_rs::error::CanonicalError;
use toolify_rs::routing::ModelRouter;
use toolify_rs::state::{AppState, SessionClass};
use toolify_rs::transport::{HttpTransport, PreparedUpstream};

fn build_state() -> Arc<AppState> {
    let config = AppConfig {
        server: ServerConfig::default(),
        upstream_services: vec![
            UpstreamServiceConfig {
                name: "openai-a".to_string(),
                provider: "openai".to_string(),
                base_url: "http://127.0.0.1:8001/v1".to_string(),
                api_key: "k1".to_string(),
                models: vec!["m".to_string()],
                description: String::new(),
                is_default: true,
                fc_mode: FcMode::Native,
                api_version: None,
                proxy: None,
                proxy_stream: None,
                proxy_non_stream: None,
            },
            UpstreamServiceConfig {
                name: "anthropic-b".to_string(),
                provider: "anthropic".to_string(),
                base_url: "http://127.0.0.1:8002/v1".to_string(),
                api_key: "k2".to_string(),
                models: vec!["m".to_string()],
                description: String::new(),
                is_default: false,
                fc_mode: FcMode::Native,
                api_version: None,
                proxy: None,
                proxy_stream: None,
                proxy_non_stream: None,
            },
        ],
        client_authentication: ClientAuthConfig {
            allowed_keys: vec!["client-key".to_string()],
        },
        features: FeaturesConfig::default(),
    };

    let model_router = ModelRouter::new(&config);
    let prepared_upstreams = config
        .upstream_services
        .iter()
        .map(PreparedUpstream::new)
        .collect();
    let allowed_client_keys = build_allowed_key_set(&config);

    Arc::new(AppState::new(
        config,
        HttpTransport::new(&ServerConfig::default()),
        model_router,
        prepared_upstreams,
        allowed_client_keys,
    ))
}

#[test]
fn test_session_policy_portable_vs_anchored_order_when_primary_breaker_open() {
    let state = build_state();
    let model = "m";
    let request_hash = 0_u64; // deterministic primary = first upstream.

    // Open breaker for primary upstream/model group.
    let failure = CanonicalError::Upstream {
        status: 503,
        message: "temporarily unavailable".to_string(),
    };
    for _ in 0..5 {
        state.record_upstream_failure(0, model, &failure);
    }

    let portable = state
        .resolve_routes_with_policy(model, request_hash, SessionClass::Portable)
        .expect("portable routes");
    let anchored = state
        .resolve_routes_with_policy(model, request_hash, SessionClass::Anchored)
        .expect("anchored routes");

    assert_eq!(portable.len(), 2);
    assert_eq!(anchored.len(), 2);

    // Portable prefers healthy cross-provider candidate before blocked primary.
    assert_eq!(portable[0].upstream_index, 1);
    assert_eq!(portable[1].upstream_index, 0);

    // Anchored keeps same-provider candidate first even when breaker-open,
    // and only then degrades to cross-provider.
    assert_eq!(anchored[0].upstream_index, 0);
    assert_eq!(anchored[1].upstream_index, 1);
}
