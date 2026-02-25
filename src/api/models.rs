use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::{body::Body, http::StatusCode};

use crate::error::into_axum_response;
use crate::protocol::canonical::IngressApi;

use crate::state::AppState;

/// List all models from all upstream services in `OpenAI` format.
#[must_use]
pub async fn handler(State(state): State<Arc<AppState>>, headers: &HeaderMap) -> Response {
    const INGRESS: IngressApi = IngressApi::OpenAiChat;
    if let Err(err) = state.authenticate(INGRESS, headers) {
        return into_axum_response(&err, INGRESS);
    }
    state.maybe_refresh_models_cache().await;

    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        )],
        Body::from(state.models_response_body()),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::build_allowed_key_set;
    use crate::config::{
        AppConfig, ClientAuthConfig, FeaturesConfig, ServerConfig, UpstreamServiceConfig,
    };
    use crate::routing::ModelRouter;
    use crate::transport::{HttpTransport, PreparedUpstream};

    #[tokio::test]
    async fn test_handler_dedup_models_and_metadata() {
        let config = AppConfig {
            server: ServerConfig::default(),
            upstream_services: vec![
                UpstreamServiceConfig {
                    name: "svc_one".into(),
                    provider: "openai".into(),
                    base_url: "https://api.example.com".into(),
                    api_key: "k1".into(),
                    models: vec!["gpt-4:o1".into(), "gpt-4o".into()],
                    description: String::new(),
                    is_default: false,
                    fc_mode: crate::config::FcMode::default(),
                    api_version: None,
                    proxy: None,
                    proxy_stream: None,
                    proxy_non_stream: None,
                },
                UpstreamServiceConfig {
                    name: "svc_two".into(),
                    provider: "openai".into(),
                    base_url: "https://api.example.com".into(),
                    api_key: "k2".into(),
                    models: vec!["gpt-4:o2".into(), "gpt-4o-mini".into()],
                    description: String::new(),
                    is_default: false,
                    fc_mode: crate::config::FcMode::default(),
                    api_version: None,
                    proxy: None,
                    proxy_stream: None,
                    proxy_non_stream: None,
                },
            ],
            client_authentication: ClientAuthConfig {
                allowed_keys: vec!["test-key".into()],
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

        let state = Arc::new(AppState::new(
            config,
            HttpTransport::new(&ServerConfig::default()),
            model_router,
            prepared_upstreams,
            allowed_client_keys,
        ));

        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer test-key".parse().unwrap());
        let response = handler(State(state), &headers).await;
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let data = body
            .get("data")
            .and_then(serde_json::Value::as_array)
            .unwrap();

        let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["gpt-4", "gpt-4o", "gpt-4o-mini"]);

        assert_eq!(
            data.iter().find(|m| m["id"] == "gpt-4").unwrap()["owned_by"],
            "svc_one"
        );
        for m in data {
            assert_eq!(m["object"], "model");
            assert_eq!(m["created"], 1_677_610_602);
            assert_eq!(m["permission"].as_array().unwrap().len(), 0);
            assert_eq!(m["root"], m["id"]);
            assert!(matches!(m.get("parent"), Some(v) if v.is_null()));
        }
    }
}
