use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
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

fn allowed_keys(prefix: &str) -> Vec<String> {
    (0..96).map(|idx| format!("{prefix}-{idx}")).collect()
}

#[tokio::test]
async fn test_openai_responses_same_request_failover_to_alternate_upstream() {
    let fail_hits = Arc::new(AtomicUsize::new(0));
    let success_hits = Arc::new(AtomicUsize::new(0));

    let fail_hits_clone = Arc::clone(&fail_hits);
    let fail_app = Router::new().route(
        "/v1/responses",
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
        .expect("bind failing responses upstream");
    let fail_addr = fail_listener.local_addr().expect("failing responses addr");
    let fail_server = tokio::spawn(async move {
        let _ = axum::serve(fail_listener, fail_app).await;
    });

    let success_hits_clone = Arc::clone(&success_hits);
    let success_app = Router::new().route(
        "/v1/responses",
        post(move || {
            let success_hits = Arc::clone(&success_hits_clone);
            async move {
                success_hits.fetch_add(1, Ordering::Relaxed);
                Json(json!({
                    "id": "resp_mock",
                    "object": "response",
                    "model": "gpt-4o-mini",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_mock",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                { "type": "output_text", "text": "responses-fallback-ok" }
                            ]
                        }
                    ]
                }))
            }
        }),
    );
    let success_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind success responses upstream");
    let success_addr = success_listener
        .local_addr()
        .expect("success responses addr");
    let success_server = tokio::spawn(async move {
        let _ = axum::serve(success_listener, success_app).await;
    });

    let keys = allowed_keys("client-key-responses");
    let upstream_services = vec![
        UpstreamServiceConfig {
            name: "responses-0".to_string(),
            provider: "openai-responses".to_string(),
            base_url: format!("http://{fail_addr}/v1"),
            api_key: "upstream-secret".to_string(),
            models: vec!["gpt-4o-mini".to_string()],
            description: String::new(),
            is_default: true,
            fc_mode: FcMode::Native,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        },
        UpstreamServiceConfig {
            name: "responses-1".to_string(),
            provider: "openai-responses".to_string(),
            base_url: format!("http://{success_addr}/v1"),
            api_key: "upstream-secret".to_string(),
            models: vec!["gpt-4o-mini".to_string()],
            description: String::new(),
            is_default: false,
            fc_mode: FcMode::Native,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        },
    ];
    let state = build_state_multi_from_services(upstream_services, keys.clone());
    let request_body = serde_json::to_vec(&json!({
        "model": "gpt-4o-mini",
        "input": [
            {
                "role": "user",
                "content": [{ "type": "input_text", "text": "ping" }]
            }
        ],
        "stream": false
    }))
    .expect("serialize request");

    let mut observed_failover = false;
    for key in &keys {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
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
            payload["output"][0]["content"][0]["text"],
            "responses-fallback-ok"
        );

        if fail_hits.load(Ordering::Relaxed) > 0 && success_hits.load(Ordering::Relaxed) > 0 {
            observed_failover = true;
            break;
        }
    }

    assert!(
        observed_failover,
        "expected at least one request to fail on primary and succeed on alternate responses upstream"
    );

    fail_server.abort();
    success_server.abort();
}

#[tokio::test]
async fn test_anthropic_same_request_failover_to_alternate_upstream() {
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
    let fail_addr = fail_listener.local_addr().expect("failing anthropic addr");
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
                    "content": [{ "type": "text", "text": "anthropic-fallback-ok" }],
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
        .expect("success anthropic addr");
    let success_server = tokio::spawn(async move {
        let _ = axum::serve(success_listener, success_app).await;
    });

    let keys = allowed_keys("client-key-anthropic");
    let upstream_services = vec![
        UpstreamServiceConfig {
            name: "anthropic-0".to_string(),
            provider: "anthropic".to_string(),
            base_url: format!("http://{fail_addr}/v1"),
            api_key: "upstream-secret".to_string(),
            models: vec!["claude-3-5-haiku-latest".to_string()],
            description: String::new(),
            is_default: true,
            fc_mode: FcMode::Native,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        },
        UpstreamServiceConfig {
            name: "anthropic-1".to_string(),
            provider: "anthropic".to_string(),
            base_url: format!("http://{success_addr}/v1"),
            api_key: "upstream-secret".to_string(),
            models: vec!["claude-3-5-haiku-latest".to_string()],
            description: String::new(),
            is_default: false,
            fc_mode: FcMode::Native,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        },
    ];
    let state = build_state_multi_from_services(upstream_services, keys.clone());
    let request_body = serde_json::to_vec(&json!({
        "model": "claude-3-5-haiku-latest",
        "messages": [{ "role": "user", "content": "ping" }],
        "stream": false
    }))
    .expect("serialize request");

    let mut observed_failover = false;
    for key in &keys {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("x-api-key", key)
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
        assert_eq!(payload["content"][0]["text"], "anthropic-fallback-ok");

        if fail_hits.load(Ordering::Relaxed) > 0 && success_hits.load(Ordering::Relaxed) > 0 {
            observed_failover = true;
            break;
        }
    }

    assert!(
        observed_failover,
        "expected at least one request to fail on primary and succeed on alternate anthropic upstream"
    );

    fail_server.abort();
    success_server.abort();
}

#[tokio::test]
async fn test_gemini_same_request_failover_to_alternate_upstream() {
    let fail_hits = Arc::new(AtomicUsize::new(0));
    let success_hits = Arc::new(AtomicUsize::new(0));

    let fail_hits_clone = Arc::clone(&fail_hits);
    let fail_app = Router::new().route(
        "/v1beta/models/gemini-2.5-pro:generateContent",
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
        .expect("bind failing gemini upstream");
    let fail_addr = fail_listener.local_addr().expect("failing gemini addr");
    let fail_server = tokio::spawn(async move {
        let _ = axum::serve(fail_listener, fail_app).await;
    });

    let success_hits_clone = Arc::clone(&success_hits);
    let success_app = Router::new().route(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        post(move || {
            let success_hits = Arc::clone(&success_hits_clone);
            async move {
                success_hits.fetch_add(1, Ordering::Relaxed);
                Json(json!({
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{ "text": "gemini-fallback-ok" }]
                            },
                            "finishReason": "STOP",
                            "index": 0
                        }
                    ]
                }))
            }
        }),
    );
    let success_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind success gemini upstream");
    let success_addr = success_listener.local_addr().expect("success gemini addr");
    let success_server = tokio::spawn(async move {
        let _ = axum::serve(success_listener, success_app).await;
    });

    let keys = allowed_keys("client-key-gemini");
    let upstream_services = vec![
        UpstreamServiceConfig {
            name: "gemini-0".to_string(),
            provider: "gemini".to_string(),
            base_url: format!("http://{fail_addr}/v1beta"),
            api_key: "upstream-secret".to_string(),
            models: vec!["gemini-2.5-pro".to_string()],
            description: String::new(),
            is_default: true,
            fc_mode: FcMode::Native,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        },
        UpstreamServiceConfig {
            name: "gemini-1".to_string(),
            provider: "gemini".to_string(),
            base_url: format!("http://{success_addr}/v1beta"),
            api_key: "upstream-secret".to_string(),
            models: vec!["gemini-2.5-pro".to_string()],
            description: String::new(),
            is_default: false,
            fc_mode: FcMode::Native,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        },
    ];
    let state = build_state_multi_from_services(upstream_services, keys.clone());
    let request_body = serde_json::to_vec(&json!({
        "contents": [
            {
                "role": "user",
                "parts": [{ "text": "ping" }]
            }
        ]
    }))
    .expect("serialize request");

    let mut observed_failover = false;
    for key in &keys {
        let request = Request::builder()
            .method("POST")
            .uri("/v1beta/models/gemini-2.5-pro:generateContent")
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
            payload["candidates"][0]["content"]["parts"][0]["text"],
            "gemini-fallback-ok"
        );

        if fail_hits.load(Ordering::Relaxed) > 0 && success_hits.load(Ordering::Relaxed) > 0 {
            observed_failover = true;
            break;
        }
    }

    assert!(
        observed_failover,
        "expected at least one request to fail on primary and succeed on alternate gemini upstream"
    );

    fail_server.abort();
    success_server.abort();
}

#[tokio::test]
async fn test_gemini_auto_mode_native_to_inject_retry() {
    let bodies = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
    let hits = Arc::new(AtomicUsize::new(0));

    let bodies_clone = Arc::clone(&bodies);
    let hits_clone = Arc::clone(&hits);
    let app = Router::new().route(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        post(move |body: bytes::Bytes| {
            let bodies = Arc::clone(&bodies_clone);
            let hits = Arc::clone(&hits_clone);
            async move {
                hits.fetch_add(1, Ordering::Relaxed);
                let payload: serde_json::Value =
                    serde_json::from_slice(&body).expect("request json");
                bodies.lock().expect("lock bodies").push(payload.clone());

                if payload.get("tools").is_some() {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": { "message": "tools unsupported" }
                        })),
                    );
                }
                (
                    StatusCode::OK,
                    Json(json!({
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{ "text": "inject-retry-ok" }]
                                },
                                "finishReason": "STOP",
                                "index": 0
                            }
                        ]
                    })),
                )
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind gemini auto upstream");
    let addr = listener.local_addr().expect("gemini auto addr");
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let upstream_services = vec![UpstreamServiceConfig {
        name: "gemini-auto".to_string(),
        provider: "gemini".to_string(),
        base_url: format!("http://{addr}/v1beta"),
        api_key: "upstream-secret".to_string(),
        models: vec!["gemini-2.5-pro".to_string()],
        description: String::new(),
        is_default: true,
        fc_mode: FcMode::Auto,
        api_version: None,
        proxy: None,
        proxy_stream: None,
        proxy_non_stream: None,
    }];
    let state = build_state_multi_from_services(upstream_services, vec!["client-key".to_string()]);

    let request_body = serde_json::to_vec(&json!({
        "contents": [{ "role": "user", "parts": [{ "text": "call weather" }] }],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": { "city": { "type": "string" } },
                            "required": ["city"]
                        }
                    }
                ]
            }
        ]
    }))
    .expect("serialize request");

    let request = Request::builder()
        .method("POST")
        .uri("/v1beta/models/gemini-2.5-pro:generateContent")
        .header("authorization", "Bearer client-key")
        .header("content-type", "application/json")
        .body(Body::from(request_body))
        .expect("build request");

    let response = dispatch_request(state, Arc::<str>::from(""), request)
        .await
        .expect("dispatch");
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read response body");
    let payload: serde_json::Value = serde_json::from_slice(&body).expect("json payload");
    assert_eq!(
        payload["candidates"][0]["content"]["parts"][0]["text"],
        "inject-retry-ok"
    );

    let seen = bodies.lock().expect("lock bodies");
    assert_eq!(
        hits.load(Ordering::Relaxed),
        2,
        "expected native + inject retry"
    );
    assert!(seen
        .first()
        .is_some_and(|first| first.get("tools").is_some()));
    assert!(
        seen.get(1)
            .is_some_and(|second| second.get("tools").is_none()),
        "second request should be inject mode payload without native tools field"
    );

    server.abort();
}

#[tokio::test]
async fn test_openai_chat_auto_mode_native_to_inject_retry() {
    let bodies = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
    let hits = Arc::new(AtomicUsize::new(0));

    let bodies_clone = Arc::clone(&bodies);
    let hits_clone = Arc::clone(&hits);
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |body: bytes::Bytes| {
            let bodies = Arc::clone(&bodies_clone);
            let hits = Arc::clone(&hits_clone);
            async move {
                hits.fetch_add(1, Ordering::Relaxed);
                let payload: serde_json::Value =
                    serde_json::from_slice(&body).expect("request json");
                bodies.lock().expect("lock bodies").push(payload.clone());

                if payload.get("tools").is_some() {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": { "message": "tools unsupported" }
                        })),
                    );
                }

                (
                    StatusCode::OK,
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
                                    "content": "inject-retry-ok"
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 2,
                            "total_tokens": 7
                        }
                    })),
                )
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind openai auto upstream");
    let addr = listener.local_addr().expect("openai auto addr");
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let upstream_services = vec![UpstreamServiceConfig {
        name: "openai-auto".to_string(),
        provider: "openai".to_string(),
        base_url: format!("http://{addr}/v1"),
        api_key: "upstream-secret".to_string(),
        models: vec!["gpt-4o-mini".to_string()],
        description: String::new(),
        is_default: true,
        fc_mode: FcMode::Auto,
        api_version: None,
        proxy: None,
        proxy_stream: None,
        proxy_non_stream: None,
    }];
    let state = build_state_multi_from_services(upstream_services, vec!["client-key".to_string()]);

    let request_body = serde_json::to_vec(&json!({
        "model": "gpt-4o-mini",
        "messages": [{ "role": "user", "content": "call weather" }],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"]
                    }
                }
            }
        ],
        "tool_choice": "auto"
    }))
    .expect("serialize request");

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", "Bearer client-key")
        .header("content-type", "application/json")
        .body(Body::from(request_body))
        .expect("build request");

    let response = dispatch_request(state, Arc::<str>::from(""), request)
        .await
        .expect("dispatch");
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read response body");
    let payload: serde_json::Value = serde_json::from_slice(&body).expect("json payload");
    assert_eq!(
        payload["choices"][0]["message"]["content"],
        "inject-retry-ok"
    );

    let seen = bodies.lock().expect("lock bodies");
    assert_eq!(
        hits.load(Ordering::Relaxed),
        2,
        "expected native + inject retry"
    );
    assert!(seen
        .first()
        .is_some_and(|first| first.get("tools").is_some()));
    assert!(
        seen.get(1)
            .is_some_and(|second| second.get("tools").is_none()),
        "second request should be inject mode payload without native tools field"
    );

    server.abort();
}

#[tokio::test]
async fn test_openai_responses_auto_mode_native_to_inject_retry() {
    let bodies = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
    let hits = Arc::new(AtomicUsize::new(0));

    let bodies_clone = Arc::clone(&bodies);
    let hits_clone = Arc::clone(&hits);
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: bytes::Bytes| {
            let bodies = Arc::clone(&bodies_clone);
            let hits = Arc::clone(&hits_clone);
            async move {
                hits.fetch_add(1, Ordering::Relaxed);
                let payload: serde_json::Value =
                    serde_json::from_slice(&body).expect("request json");
                bodies.lock().expect("lock bodies").push(payload.clone());

                if payload.get("tools").is_some() {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": { "message": "tools unsupported" }
                        })),
                    );
                }

                (
                    StatusCode::OK,
                    Json(json!({
                        "id": "resp_mock",
                        "object": "response",
                        "model": "gpt-4o-mini",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_mock",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    { "type": "output_text", "text": "inject-retry-ok" }
                                ]
                            }
                        ]
                    })),
                )
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind responses auto upstream");
    let addr = listener.local_addr().expect("responses auto addr");
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let upstream_services = vec![UpstreamServiceConfig {
        name: "responses-auto".to_string(),
        provider: "openai-responses".to_string(),
        base_url: format!("http://{addr}/v1"),
        api_key: "upstream-secret".to_string(),
        models: vec!["gpt-4o-mini".to_string()],
        description: String::new(),
        is_default: true,
        fc_mode: FcMode::Auto,
        api_version: None,
        proxy: None,
        proxy_stream: None,
        proxy_non_stream: None,
    }];
    let state = build_state_multi_from_services(upstream_services, vec!["client-key".to_string()]);

    let request_body = serde_json::to_vec(&json!({
        "model": "gpt-4o-mini",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "call weather" }]
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }
            }
        ],
        "tool_choice": "auto"
    }))
    .expect("serialize request");

    let request = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("authorization", "Bearer client-key")
        .header("content-type", "application/json")
        .body(Body::from(request_body))
        .expect("build request");

    let response = dispatch_request(state, Arc::<str>::from(""), request)
        .await
        .expect("dispatch");
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read response body");
    let payload: serde_json::Value = serde_json::from_slice(&body).expect("json payload");
    assert_eq!(
        payload["output"][0]["content"][0]["text"],
        "inject-retry-ok"
    );

    let seen = bodies.lock().expect("lock bodies");
    assert_eq!(
        hits.load(Ordering::Relaxed),
        2,
        "expected native + inject retry"
    );
    assert!(seen
        .first()
        .is_some_and(|first| first.get("tools").is_some()));
    assert!(
        seen.get(1)
            .is_some_and(|second| second.get("tools").is_none()),
        "second request should be inject mode payload without native tools field"
    );

    server.abort();
}

#[tokio::test]
async fn test_anthropic_auto_mode_native_to_inject_retry() {
    let bodies = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
    let hits = Arc::new(AtomicUsize::new(0));

    let bodies_clone = Arc::clone(&bodies);
    let hits_clone = Arc::clone(&hits);
    let app = Router::new().route(
        "/v1/messages",
        post(move |body: bytes::Bytes| {
            let bodies = Arc::clone(&bodies_clone);
            let hits = Arc::clone(&hits_clone);
            async move {
                hits.fetch_add(1, Ordering::Relaxed);
                let payload: serde_json::Value =
                    serde_json::from_slice(&body).expect("request json");
                bodies.lock().expect("lock bodies").push(payload.clone());

                if payload.get("tools").is_some() {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "type": "error",
                            "error": { "type": "invalid_request_error", "message": "tools unsupported" }
                        })),
                    );
                }

                (
                    StatusCode::OK,
                    Json(json!({
                        "id": "msg_mock",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-3-5-haiku-latest",
                        "content": [{ "type": "text", "text": "inject-retry-ok" }],
                        "stop_reason": "end_turn",
                        "stop_sequence": null,
                        "usage": { "input_tokens": 5, "output_tokens": 2 }
                    })),
                )
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind anthropic auto upstream");
    let addr = listener.local_addr().expect("anthropic auto addr");
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let upstream_services = vec![UpstreamServiceConfig {
        name: "anthropic-auto".to_string(),
        provider: "anthropic".to_string(),
        base_url: format!("http://{addr}/v1"),
        api_key: "upstream-secret".to_string(),
        models: vec!["claude-3-5-haiku-latest".to_string()],
        description: String::new(),
        is_default: true,
        fc_mode: FcMode::Auto,
        api_version: None,
        proxy: None,
        proxy_stream: None,
        proxy_non_stream: None,
    }];
    let state = build_state_multi_from_services(upstream_services, vec!["client-key".to_string()]);

    let request_body = serde_json::to_vec(&json!({
        "model": "claude-3-5-haiku-latest",
        "max_tokens": 256,
        "messages": [{ "role": "user", "content": "call weather" }],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }
            }
        ],
        "tool_choice": { "type": "auto" }
    }))
    .expect("serialize request");

    let request = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("x-api-key", "client-key")
        .header("content-type", "application/json")
        .body(Body::from(request_body))
        .expect("build request");

    let response = dispatch_request(state, Arc::<str>::from(""), request)
        .await
        .expect("dispatch");
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read response body");
    let payload: serde_json::Value = serde_json::from_slice(&body).expect("json payload");
    assert_eq!(payload["content"][0]["text"], "inject-retry-ok");

    let seen = bodies.lock().expect("lock bodies");
    assert_eq!(
        hits.load(Ordering::Relaxed),
        2,
        "expected native + inject retry"
    );
    assert!(seen
        .first()
        .is_some_and(|first| first.get("tools").is_some()));
    assert!(
        seen.get(1)
            .is_some_and(|second| second.get("tools").is_none()),
        "second request should be inject mode payload without native tools field"
    );

    server.abort();
}

#[tokio::test]
async fn test_gemini_fc_non_stream_failover_to_alternate_upstream() {
    let fail_hits = Arc::new(AtomicUsize::new(0));
    let success_hits = Arc::new(AtomicUsize::new(0));

    let fail_hits_clone = Arc::clone(&fail_hits);
    let fail_app = Router::new().route(
        "/v1beta/models/gemini-2.5-pro:generateContent",
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
        .expect("bind failing gemini upstream");
    let fail_addr = fail_listener.local_addr().expect("failing gemini addr");
    let fail_server = tokio::spawn(async move {
        let _ = axum::serve(fail_listener, fail_app).await;
    });

    let success_hits_clone = Arc::clone(&success_hits);
    let success_app = Router::new().route(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        post(move || {
            let success_hits = Arc::clone(&success_hits_clone);
            async move {
                success_hits.fetch_add(1, Ordering::Relaxed);
                Json(json!({
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{ "text": "gemini-fc-fallback-ok" }]
                            },
                            "finishReason": "STOP",
                            "index": 0
                        }
                    ]
                }))
            }
        }),
    );
    let success_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind success gemini upstream");
    let success_addr = success_listener.local_addr().expect("success gemini addr");
    let success_server = tokio::spawn(async move {
        let _ = axum::serve(success_listener, success_app).await;
    });

    let keys = allowed_keys("client-key-gemini-fc");
    let upstream_services = vec![
        UpstreamServiceConfig {
            name: "gemini-fc-0".to_string(),
            provider: "gemini".to_string(),
            base_url: format!("http://{fail_addr}/v1beta"),
            api_key: "upstream-secret".to_string(),
            models: vec!["gemini-2.5-pro".to_string()],
            description: String::new(),
            is_default: true,
            fc_mode: FcMode::Inject,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        },
        UpstreamServiceConfig {
            name: "gemini-fc-1".to_string(),
            provider: "gemini".to_string(),
            base_url: format!("http://{success_addr}/v1beta"),
            api_key: "upstream-secret".to_string(),
            models: vec!["gemini-2.5-pro".to_string()],
            description: String::new(),
            is_default: false,
            fc_mode: FcMode::Inject,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        },
    ];
    let state = build_state_multi_from_services(upstream_services, keys.clone());

    let request_body = serde_json::to_vec(&json!({
        "contents": [
            {
                "role": "user",
                "parts": [{ "text": "call weather" }]
            }
        ],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": { "city": { "type": "string" } },
                            "required": ["city"]
                        }
                    }
                ]
            }
        ]
    }))
    .expect("serialize request");

    let mut observed_failover = false;
    for key in &keys {
        let request = Request::builder()
            .method("POST")
            .uri("/v1beta/models/gemini-2.5-pro:generateContent")
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
            payload["candidates"][0]["content"]["parts"][0]["text"],
            "gemini-fc-fallback-ok"
        );

        if fail_hits.load(Ordering::Relaxed) > 0 && success_hits.load(Ordering::Relaxed) > 0 {
            observed_failover = true;
            break;
        }
    }

    assert!(
        observed_failover,
        "expected at least one request to fail on primary and succeed on alternate gemini upstream in FC non-stream path"
    );

    fail_server.abort();
    success_server.abort();
}
