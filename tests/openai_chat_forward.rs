use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use axum::routing::post;
use axum::{Json, Router};
use serde_json::json;
use toolify_rs::auth::build_allowed_key_set;
use toolify_rs::config::{
    AppConfig, ClientAuthConfig, FcMode, FeaturesConfig, ServerConfig, UpstreamServiceConfig,
};
use toolify_rs::routing::dispatch::dispatch_request;
use toolify_rs::routing::ModelRouter;
use toolify_rs::state::AppState;
use toolify_rs::transport::{HttpTransport, PreparedUpstream};

fn build_state(base_url: String) -> Arc<AppState> {
    let config = AppConfig {
        server: ServerConfig::default(),
        upstream_services: vec![UpstreamServiceConfig {
            name: "mock-openai".to_string(),
            provider: "openai".to_string(),
            base_url,
            api_key: "upstream-secret".to_string(),
            models: vec!["gpt-4o-mini".to_string()],
            description: String::new(),
            is_default: true,
            fc_mode: FcMode::Native,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        }],
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

fn build_state_multi(base_urls: Vec<String>, allowed_keys: Vec<String>) -> Arc<AppState> {
    let upstream_services: Vec<UpstreamServiceConfig> = base_urls
        .into_iter()
        .enumerate()
        .map(|(index, base_url)| UpstreamServiceConfig {
            name: format!("mock-openai-{index}"),
            provider: "openai".to_string(),
            base_url,
            api_key: "upstream-secret".to_string(),
            models: vec!["gpt-4o-mini".to_string()],
            description: String::new(),
            is_default: index == 0,
            fc_mode: FcMode::Native,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        })
        .collect();

    build_state_multi_from_services(upstream_services, allowed_keys)
}

fn build_state_multi_from_services(
    upstream_services: Vec<UpstreamServiceConfig>,
    allowed_keys: Vec<String>,
) -> Arc<AppState> {
    let config = AppConfig {
        server: ServerConfig::default(),
        upstream_services,
        client_authentication: ClientAuthConfig { allowed_keys },
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

#[tokio::test]
async fn test_openai_chat_passthrough_forward() {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            Json(json!({
                "id": "chatcmpl_mock",
                "object": "chat.completion",
                "created": 1_727_000_000_u64,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "pong"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7
                }
            }))
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock upstream");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let state = build_state(format!("http://{addr}/v1"));
    let body = serde_json::to_vec(&json!({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "ping"
            }
        ],
        "stream": false
    }))
    .expect("serialize request");

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", "Bearer client-key")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("build request");

    let response = dispatch_request(state, Arc::<str>::from(""), request)
        .await
        .expect("dispatch");
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read response body");
    let payload: serde_json::Value = serde_json::from_slice(&body).expect("json payload");
    assert_eq!(payload["choices"][0]["message"]["content"], "pong");

    server.abort();
}

#[tokio::test]
async fn test_openai_chat_single_alias_model_raw_passthrough() {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            Json(json!({
                "id": "chatcmpl_alias",
                "object": "chat.completion",
                "created": 1_727_000_001_u64,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "alias-ok"
                        },
                        "finish_reason": "stop"
                    }
                ]
            }))
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind alias upstream");
    let addr = listener.local_addr().expect("alias upstream addr");
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let upstream_services = vec![UpstreamServiceConfig {
        name: "mock-openai-alias".to_string(),
        provider: "openai".to_string(),
        base_url: format!("http://{addr}/v1"),
        api_key: "upstream-secret".to_string(),
        models: vec!["smart:gpt-4o-mini".to_string()],
        description: String::new(),
        is_default: true,
        fc_mode: FcMode::Native,
        api_version: None,
        proxy: None,
        proxy_stream: None,
        proxy_non_stream: None,
    }];
    let state = build_state_multi_from_services(upstream_services, vec!["client-key".to_string()]);

    let body = serde_json::to_vec(&json!({
        "model": "smart",
        "messages": [
            {
                "role": "user",
                "content": "ping"
            }
        ],
        "stream": false
    }))
    .expect("serialize alias request");

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", "Bearer client-key")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("build alias request");

    let response = dispatch_request(state, Arc::<str>::from(""), request)
        .await
        .expect("dispatch alias request");
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read alias response body");
    let payload: serde_json::Value = serde_json::from_slice(&body).expect("alias json payload");
    assert_eq!(payload["choices"][0]["message"]["content"], "alias-ok");
    assert_eq!(payload["model"], "gpt-4o-mini");

    server.abort();
}

#[tokio::test]
async fn test_openai_chat_same_request_failover_to_alternate_upstream() {
    let fail_hits = Arc::new(AtomicUsize::new(0));
    let success_hits = Arc::new(AtomicUsize::new(0));

    let fail_hits_clone = Arc::clone(&fail_hits);
    let fail_app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let fail_hits = Arc::clone(&fail_hits_clone);
            async move {
                fail_hits.fetch_add(1, Ordering::Relaxed);
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "error": { "message": "overloaded" }
                    })),
                )
            }
        }),
    );
    let fail_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failing upstream");
    let fail_addr = fail_listener.local_addr().expect("failing upstream addr");
    let fail_server = tokio::spawn(async move {
        let _ = axum::serve(fail_listener, fail_app).await;
    });

    let success_hits_clone = Arc::clone(&success_hits);
    let success_app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let success_hits = Arc::clone(&success_hits_clone);
            async move {
                success_hits.fetch_add(1, Ordering::Relaxed);
                Json(json!({
                    "id": "chatcmpl_mock",
                    "object": "chat.completion",
                    "created": 1_727_000_000_u64,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "fallback-ok"
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7
                    }
                }))
            }
        }),
    );
    let success_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind success upstream");
    let success_addr = success_listener
        .local_addr()
        .expect("success upstream addr");
    let success_server = tokio::spawn(async move {
        let _ = axum::serve(success_listener, success_app).await;
    });

    let allowed_keys: Vec<String> = (0..64).map(|idx| format!("client-key-{idx}")).collect();
    let state = build_state_multi(
        vec![
            format!("http://{fail_addr}/v1"),
            format!("http://{success_addr}/v1"),
        ],
        allowed_keys.clone(),
    );

    let request_body = serde_json::to_vec(&json!({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "ping"
            }
        ],
        "stream": false
    }))
    .expect("serialize request");

    let mut observed_cross_upstream = false;
    for key in &allowed_keys {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(Body::from(request_body.clone()))
            .expect("build request");

        let response = dispatch_request(Arc::clone(&state), Arc::<str>::from(""), request)
            .await
            .expect("dispatch");
        assert_eq!(response.status(), StatusCode::OK);

        if fail_hits.load(Ordering::Relaxed) > 0 && success_hits.load(Ordering::Relaxed) > 0 {
            observed_cross_upstream = true;
            break;
        }
    }

    assert!(
        observed_cross_upstream,
        "expected at least one request to hit failing primary and succeed via alternate upstream"
    );

    fail_server.abort();
    success_server.abort();
}

#[tokio::test]
async fn test_openai_chat_stream_failover_to_alternate_upstream_before_first_byte() {
    let fail_hits = Arc::new(AtomicUsize::new(0));
    let success_hits = Arc::new(AtomicUsize::new(0));

    let fail_hits_clone = Arc::clone(&fail_hits);
    let fail_app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let fail_hits = Arc::clone(&fail_hits_clone);
            async move {
                fail_hits.fetch_add(1, Ordering::Relaxed);
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "error": { "message": "overloaded" }
                    })),
                )
            }
        }),
    );
    let fail_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failing upstream");
    let fail_addr = fail_listener.local_addr().expect("failing upstream addr");
    let fail_server = tokio::spawn(async move {
        let _ = axum::serve(fail_listener, fail_app).await;
    });

    let success_hits_clone = Arc::clone(&success_hits);
    let success_app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let success_hits = Arc::clone(&success_hits_clone);
            async move {
                success_hits.fetch_add(1, Ordering::Relaxed);
                let sse = concat!(
                    "data: {\"id\":\"chatcmpl_mock\",\"object\":\"chat.completion.chunk\",\"created\":1727000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"stream-fallback-ok\"},\"finish_reason\":null}]}\n\n",
                    "data: {\"id\":\"chatcmpl_mock\",\"object\":\"chat.completion.chunk\",\"created\":1727000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                    "data: [DONE]\n\n"
                );
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .expect("stream response")
            }
        }),
    );
    let success_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind success upstream");
    let success_addr = success_listener
        .local_addr()
        .expect("success upstream addr");
    let success_server = tokio::spawn(async move {
        let _ = axum::serve(success_listener, success_app).await;
    });

    let allowed_keys: Vec<String> = (0..64)
        .map(|idx| format!("client-key-stream-{idx}"))
        .collect();
    let state = build_state_multi(
        vec![
            format!("http://{fail_addr}/v1"),
            format!("http://{success_addr}/v1"),
        ],
        allowed_keys.clone(),
    );

    let request_body = serde_json::to_vec(&json!({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "ping"
            }
        ],
        "stream": true
    }))
    .expect("serialize request");

    let mut observed_cross_upstream = false;
    for key in &allowed_keys {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(Body::from(request_body.clone()))
            .expect("build request");

        let response = dispatch_request(Arc::clone(&state), Arc::<str>::from(""), request)
            .await
            .expect("dispatch");
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body");
        let body_text = String::from_utf8(body.to_vec()).expect("utf8 body");
        assert!(body_text.contains("stream-fallback-ok"));
        assert!(body_text.contains("[DONE]"));

        if fail_hits.load(Ordering::Relaxed) > 0 && success_hits.load(Ordering::Relaxed) > 0 {
            observed_cross_upstream = true;
            break;
        }
    }

    assert!(
        observed_cross_upstream,
        "expected stream request to fail on primary and succeed via alternate upstream before first byte"
    );

    fail_server.abort();
    success_server.abort();
}

#[tokio::test]
async fn test_openai_chat_transcode_failover_to_alternate_upstream() {
    let fail_hits = Arc::new(AtomicUsize::new(0));
    let success_hits = Arc::new(AtomicUsize::new(0));

    let fail_hits_clone = Arc::clone(&fail_hits);
    let fail_app = Router::new().route(
        "/v1/messages",
        post(move || {
            let fail_hits = Arc::clone(&fail_hits_clone);
            async move {
                fail_hits.fetch_add(1, Ordering::Relaxed);
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": "overloaded" }
                    })),
                )
            }
        }),
    );
    let fail_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failing anthropic upstream");
    let fail_addr = fail_listener
        .local_addr()
        .expect("failing anthropic upstream addr");
    let fail_server = tokio::spawn(async move {
        let _ = axum::serve(fail_listener, fail_app).await;
    });

    let success_hits_clone = Arc::clone(&success_hits);
    let success_app = Router::new().route(
        "/v1/messages",
        post(move || {
            let success_hits = Arc::clone(&success_hits_clone);
            async move {
                success_hits.fetch_add(1, Ordering::Relaxed);
                Json(json!({
                    "id": "msg_mock",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-haiku-latest",
                    "content": [
                        { "type": "text", "text": "anthropic-fallback-ok" }
                    ],
                    "stop_reason": "end_turn",
                    "stop_sequence": null,
                    "usage": { "input_tokens": 5, "output_tokens": 2 }
                }))
            }
        }),
    );
    let success_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind success anthropic upstream");
    let success_addr = success_listener
        .local_addr()
        .expect("success anthropic upstream addr");
    let success_server = tokio::spawn(async move {
        let _ = axum::serve(success_listener, success_app).await;
    });

    let allowed_keys: Vec<String> = (0..64)
        .map(|idx| format!("client-key-anthropic-{idx}"))
        .collect();
    let upstream_services = vec![
        UpstreamServiceConfig {
            name: "mock-anthropic-0".to_string(),
            provider: "anthropic".to_string(),
            base_url: format!("http://{fail_addr}/v1"),
            api_key: "upstream-secret".to_string(),
            models: vec!["gpt-4o-mini:claude-3-5-haiku-latest".to_string()],
            description: String::new(),
            is_default: true,
            fc_mode: FcMode::Native,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        },
        UpstreamServiceConfig {
            name: "mock-anthropic-1".to_string(),
            provider: "anthropic".to_string(),
            base_url: format!("http://{success_addr}/v1"),
            api_key: "upstream-secret".to_string(),
            models: vec!["gpt-4o-mini:claude-3-5-haiku-latest".to_string()],
            description: String::new(),
            is_default: false,
            fc_mode: FcMode::Native,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        },
    ];
    let state = build_state_multi_from_services(upstream_services, allowed_keys.clone());

    let request_body = serde_json::to_vec(&json!({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "ping"
            }
        ],
        "stream": false
    }))
    .expect("serialize request");

    let mut observed_cross_upstream = false;
    for key in &allowed_keys {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(Body::from(request_body.clone()))
            .expect("build request");

        let response = dispatch_request(Arc::clone(&state), Arc::<str>::from(""), request)
            .await
            .expect("dispatch");
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body");
        let payload: serde_json::Value = serde_json::from_slice(&body).expect("json payload");
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            "anthropic-fallback-ok"
        );

        if fail_hits.load(Ordering::Relaxed) > 0 && success_hits.load(Ordering::Relaxed) > 0 {
            observed_cross_upstream = true;
            break;
        }
    }

    assert!(
        observed_cross_upstream,
        "expected at least one request to hit failing primary and succeed via alternate transcode upstream"
    );

    fail_server.abort();
    success_server.abort();
}

#[tokio::test]
async fn test_openai_chat_transcode_stream_failover_to_alternate_upstream_before_first_byte() {
    let fail_hits = Arc::new(AtomicUsize::new(0));
    let success_hits = Arc::new(AtomicUsize::new(0));

    let fail_hits_clone = Arc::clone(&fail_hits);
    let fail_app = Router::new().route(
        "/v1/messages",
        post(move || {
            let fail_hits = Arc::clone(&fail_hits_clone);
            async move {
                fail_hits.fetch_add(1, Ordering::Relaxed);
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": "overloaded" }
                    })),
                )
            }
        }),
    );
    let fail_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failing anthropic upstream");
    let fail_addr = fail_listener
        .local_addr()
        .expect("failing anthropic upstream addr");
    let fail_server = tokio::spawn(async move {
        let _ = axum::serve(fail_listener, fail_app).await;
    });

    let success_hits_clone = Arc::clone(&success_hits);
    let success_app = Router::new().route(
        "/v1/messages",
        post(move || {
            let success_hits = Arc::clone(&success_hits_clone);
            async move {
                success_hits.fetch_add(1, Ordering::Relaxed);
                let sse = concat!(
                    "event: message_start\n",
                    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_mock\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-5-haiku-latest\",\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n",
                    "event: content_block_delta\n",
                    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"stream-transcode-fallback-ok\"}}\n\n",
                    "event: message_delta\n",
                    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":0,\"output_tokens\":1}}\n\n",
                    "event: message_stop\n",
                    "data: {\"type\":\"message_stop\"}\n\n"
                );
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .expect("stream response")
            }
        }),
    );
    let success_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind success anthropic upstream");
    let success_addr = success_listener
        .local_addr()
        .expect("success anthropic upstream addr");
    let success_server = tokio::spawn(async move {
        let _ = axum::serve(success_listener, success_app).await;
    });

    let allowed_keys: Vec<String> = (0..64)
        .map(|idx| format!("client-key-anthropic-stream-{idx}"))
        .collect();
    let upstream_services = vec![
        UpstreamServiceConfig {
            name: "mock-anthropic-stream-0".to_string(),
            provider: "anthropic".to_string(),
            base_url: format!("http://{fail_addr}/v1"),
            api_key: "upstream-secret".to_string(),
            models: vec!["gpt-4o-mini:claude-3-5-haiku-latest".to_string()],
            description: String::new(),
            is_default: true,
            fc_mode: FcMode::Native,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        },
        UpstreamServiceConfig {
            name: "mock-anthropic-stream-1".to_string(),
            provider: "anthropic".to_string(),
            base_url: format!("http://{success_addr}/v1"),
            api_key: "upstream-secret".to_string(),
            models: vec!["gpt-4o-mini:claude-3-5-haiku-latest".to_string()],
            description: String::new(),
            is_default: false,
            fc_mode: FcMode::Native,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        },
    ];
    let state = build_state_multi_from_services(upstream_services, allowed_keys.clone());

    let request_body = serde_json::to_vec(&json!({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "ping"
            }
        ],
        "stream": true
    }))
    .expect("serialize request");

    let mut observed_cross_upstream = false;
    for key in &allowed_keys {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(Body::from(request_body.clone()))
            .expect("build request");

        let response = dispatch_request(Arc::clone(&state), Arc::<str>::from(""), request)
            .await
            .expect("dispatch");
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body");
        let body_text = String::from_utf8(body.to_vec()).expect("utf8 body");
        assert!(body_text.contains("stream-transcode-fallback-ok"));
        assert!(body_text.contains("[DONE]"));

        if fail_hits.load(Ordering::Relaxed) > 0 && success_hits.load(Ordering::Relaxed) > 0 {
            observed_cross_upstream = true;
            break;
        }
    }

    assert!(
        observed_cross_upstream,
        "expected transcode stream request to fail on primary and succeed via alternate upstream before first byte"
    );

    fail_server.abort();
    success_server.abort();
}
