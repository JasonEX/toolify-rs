use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smallvec::smallvec;
use std::sync::Arc;

use toolify_rs::auth::{authenticate, build_allowed_key_set};
use toolify_rs::config::{
    AppConfig, ClientAuthConfig, FcMode, FeaturesConfig, ServerConfig, UpstreamServiceConfig,
};
use toolify_rs::error::CanonicalError;
use toolify_rs::fc::detector::StreamingFcDetector;
use toolify_rs::fc::{parser, prompt, response_text_contains_trigger};
use toolify_rs::protocol::canonical::{
    CanonicalMessage, CanonicalPart, CanonicalRequest, CanonicalRole, CanonicalToolChoice,
    GenerationParams, IngressApi,
};
use toolify_rs::routing::{ModelRouter, RouteTarget};
use toolify_rs::state::{AppState, SessionClass};
use toolify_rs::transport::{HttpTransport, PreparedUpstream};

fn make_upstream(index: usize, provider: &str, is_default: bool) -> UpstreamServiceConfig {
    UpstreamServiceConfig {
        name: format!("{provider}-{index}"),
        provider: provider.to_string(),
        base_url: format!("http://127.0.0.1:{}/v1", 9000 + index),
        api_key: format!("k-{index}"),
        models: vec!["m".to_string()],
        description: String::new(),
        is_default,
        fc_mode: FcMode::Native,
        api_version: None,
        proxy: None,
        proxy_stream: None,
        proxy_non_stream: None,
    }
}

fn build_state(openai_count: usize, anthropic_count: usize) -> AppState {
    let mut upstream_services = Vec::with_capacity(openai_count + anthropic_count);
    for idx in 0..openai_count {
        upstream_services.push(make_upstream(idx, "openai", idx == 0));
    }
    for idx in 0..anthropic_count {
        upstream_services.push(make_upstream(openai_count + idx, "anthropic", false));
    }

    let config = AppConfig {
        server: ServerConfig::default(),
        upstream_services,
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

    AppState::new(
        config,
        HttpTransport::new(&ServerConfig::default()),
        model_router,
        prepared_upstreams,
        allowed_client_keys,
    )
}

fn payload_without_trigger(total_bytes: usize) -> Vec<u8> {
    let prefix = br#"{"choices":[{"delta":{"content":""#;
    let suffix = br#""}}]}"#;
    let fill_len = total_bytes.saturating_sub(prefix.len() + suffix.len());

    let mut bytes = Vec::with_capacity(prefix.len() + fill_len + suffix.len());
    bytes.extend_from_slice(prefix);
    bytes.extend(vec![b'a'; fill_len]);
    bytes.extend_from_slice(suffix);
    bytes
}

fn payload_with_trigger(total_bytes: usize) -> Vec<u8> {
    let trigger = prompt::get_trigger_signal().as_bytes();
    let prefix = br#"{"choices":[{"delta":{"content":""#;
    let suffix = br#""}}]}"#;

    let fill_len = total_bytes.saturating_sub(prefix.len() + trigger.len() + suffix.len());
    let head_len = fill_len / 2;
    let tail_len = fill_len - head_len;

    let mut bytes =
        Vec::with_capacity(prefix.len() + head_len + trigger.len() + tail_len + suffix.len());
    bytes.extend_from_slice(prefix);
    bytes.extend(vec![b'a'; head_len]);
    bytes.extend_from_slice(trigger);
    bytes.extend(vec![b'b'; tail_len]);
    bytes.extend_from_slice(suffix);
    bytes
}

fn fc_parse_payload_function_call() -> String {
    let trigger = prompt::get_trigger_signal();
    format!(
        "ok\n{trigger}\n<function_calls><function_call><tool>get_weather</tool><args_json>{{\"city\":\"London\"}}</args_json></function_call></function_calls>"
    )
}

fn fc_parse_payload_mixed() -> String {
    let trigger = prompt::get_trigger_signal();
    format!(
        "ok\n{trigger}\n<function_calls><invoke name=\"a\"><parameter name=\"x\">1</parameter></invoke><function_call><tool>b</tool><args_json>{{}}</args_json></function_call></function_calls>"
    )
}

fn bench_route_resolution(c: &mut Criterion) {
    let model = "m";
    let request_hash = 0_u64;

    let healthy_state = build_state(4, 4);
    c.bench_function("route_resolve_portable_healthy_8", |b| {
        b.iter(|| {
            black_box(
                healthy_state
                    .resolve_routes_with_policy(
                        black_box(model),
                        black_box(request_hash),
                        black_box(SessionClass::Portable),
                    )
                    .expect("route resolve"),
            )
        });
    });

    let degraded_state = build_state(4, 4);
    let open_err = CanonicalError::Upstream {
        status: 503,
        message: "temporarily unavailable".to_string(),
    };
    for _ in 0..5 {
        degraded_state.record_upstream_failure(0, model, &open_err);
    }

    c.bench_function("route_resolve_portable_primary_open_8", |b| {
        b.iter(|| {
            black_box(
                degraded_state
                    .resolve_routes_with_policy(
                        black_box(model),
                        black_box(request_hash),
                        black_box(SessionClass::Portable),
                    )
                    .expect("route resolve"),
            )
        });
    });

    c.bench_function("route_resolve_anchored_primary_open_8", |b| {
        b.iter(|| {
            black_box(
                degraded_state
                    .resolve_routes_with_policy(
                        black_box(model),
                        black_box(request_hash),
                        black_box(SessionClass::Anchored),
                    )
                    .expect("route resolve"),
            )
        });
    });
}

fn bench_route_sticky_hash(c: &mut Criterion) {
    let state = build_state(4, 4);
    let model = "m";
    let prompt = payload_without_trigger(1024);

    let mut openai_headers = http::HeaderMap::new();
    openai_headers.insert("authorization", "Bearer client-key".parse().unwrap());
    c.bench_function("route_sticky_hash_openai_bearer_1k", |b| {
        b.iter(|| {
            black_box(state.route_sticky_hash(
                black_box(IngressApi::OpenAiChat),
                black_box(&openai_headers),
                black_box(model),
                black_box(&prompt),
            ))
        });
    });

    let mut gemini_headers = http::HeaderMap::new();
    gemini_headers.insert("x-goog-api-key", "client-key".parse().unwrap());
    c.bench_function("route_sticky_hash_gemini_x_goog_1k", |b| {
        b.iter(|| {
            black_box(state.route_sticky_hash(
                black_box(IngressApi::Gemini),
                black_box(&gemini_headers),
                black_box(model),
                black_box(&prompt),
            ))
        });
    });
}

fn bench_authentication(c: &mut Criterion) {
    let single_cfg = AppConfig {
        server: ServerConfig::default(),
        upstream_services: vec![],
        client_authentication: ClientAuthConfig {
            allowed_keys: vec!["client-key".to_string()],
        },
        features: FeaturesConfig::default(),
    };
    let single_keys = build_allowed_key_set(&single_cfg);
    let mut single_headers = http::HeaderMap::new();
    single_headers.insert("authorization", "Bearer client-key".parse().unwrap());
    c.bench_function("auth_single_openai_bearer", |b| {
        b.iter(|| {
            black_box(
                authenticate(
                    black_box(IngressApi::OpenAiChat),
                    black_box(&single_headers),
                    black_box(&single_keys),
                )
                .is_ok(),
            )
        });
    });

    let mut multi_allowed = Vec::with_capacity(64);
    for idx in 0..63 {
        multi_allowed.push(format!("k-{idx:03}"));
    }
    multi_allowed.push("client-key".to_string());
    let multi_cfg = AppConfig {
        server: ServerConfig::default(),
        upstream_services: vec![],
        client_authentication: ClientAuthConfig {
            allowed_keys: multi_allowed,
        },
        features: FeaturesConfig::default(),
    };
    let multi_keys = build_allowed_key_set(&multi_cfg);
    let mut multi_headers = http::HeaderMap::new();
    multi_headers.insert("authorization", "Bearer client-key".parse().unwrap());
    c.bench_function("auth_multi64_openai_bearer", |b| {
        b.iter(|| {
            black_box(
                authenticate(
                    black_box(IngressApi::OpenAiChat),
                    black_box(&multi_headers),
                    black_box(&multi_keys),
                )
                .is_ok(),
            )
        });
    });
}

fn bench_model_router_single_candidate(c: &mut Criterion) {
    let state = build_state(1, 0);
    let model = "m";

    c.bench_function("model_router_single_candidate_legacy_double_lookup", |b| {
        b.iter(|| {
            if !black_box(
                state
                    .model_router
                    .requires_request_hash_for_ordering(black_box(model)),
            ) {
                black_box(
                    state
                        .model_router
                        .resolve(black_box(model), 0)
                        .expect("route resolve"),
                );
            }
        });
    });

    c.bench_function("model_router_single_candidate_direct_lookup", |b| {
        b.iter(|| {
            black_box(
                state
                    .model_router
                    .resolve_if_single_candidate(black_box(model))
                    .expect("route resolve"),
            );
        });
    });
}

#[inline]
fn start_candidate_index_legacy(
    route_candidates: &[RouteTarget<'_>],
    route: RouteTarget<'_>,
) -> usize {
    route_candidates
        .iter()
        .position(|candidate| {
            candidate.upstream_index == route.upstream_index
                && candidate.actual_model == route.actual_model
        })
        .unwrap_or(0)
}

#[inline]
fn start_candidate_index_fast(
    route_candidates: &[RouteTarget<'_>],
    route: RouteTarget<'_>,
) -> usize {
    if let Some(first) = route_candidates.first() {
        if first.upstream_index == route.upstream_index && first.actual_model == route.actual_model
        {
            return 0;
        }
    }
    route_candidates
        .iter()
        .position(|candidate| {
            candidate.upstream_index == route.upstream_index
                && candidate.actual_model == route.actual_model
        })
        .unwrap_or(0)
}

fn bench_start_candidate_index(c: &mut Criterion) {
    let route_candidates = [
        RouteTarget {
            upstream_index: 0,
            actual_model: "m0",
            known_model_id: None,
        },
        RouteTarget {
            upstream_index: 1,
            actual_model: "m1",
            known_model_id: None,
        },
        RouteTarget {
            upstream_index: 2,
            actual_model: "m2",
            known_model_id: None,
        },
        RouteTarget {
            upstream_index: 3,
            actual_model: "m3",
            known_model_id: None,
        },
        RouteTarget {
            upstream_index: 4,
            actual_model: "m4",
            known_model_id: None,
        },
        RouteTarget {
            upstream_index: 5,
            actual_model: "m5",
            known_model_id: None,
        },
        RouteTarget {
            upstream_index: 6,
            actual_model: "m6",
            known_model_id: None,
        },
        RouteTarget {
            upstream_index: 7,
            actual_model: "m7",
            known_model_id: None,
        },
    ];

    let route_first = route_candidates[0];
    let route_late = route_candidates[6];

    c.bench_function("start_candidate_index_legacy_hit_first_8", |b| {
        b.iter(|| {
            black_box(start_candidate_index_legacy(
                black_box(&route_candidates),
                black_box(route_first),
            ))
        });
    });
    c.bench_function("start_candidate_index_fast_hit_first_8", |b| {
        b.iter(|| {
            black_box(start_candidate_index_fast(
                black_box(&route_candidates),
                black_box(route_first),
            ))
        });
    });
    c.bench_function("start_candidate_index_legacy_hit_late_8", |b| {
        b.iter(|| {
            black_box(start_candidate_index_legacy(
                black_box(&route_candidates),
                black_box(route_late),
            ))
        });
    });
    c.bench_function("start_candidate_index_fast_hit_late_8", |b| {
        b.iter(|| {
            black_box(start_candidate_index_fast(
                black_box(&route_candidates),
                black_box(route_late),
            ))
        });
    });
}

fn bench_fc_trigger_scan(c: &mut Criterion) {
    let payload_small_no = payload_without_trigger(256);
    let payload_small_yes = payload_with_trigger(256);
    let payload_mid_no = payload_without_trigger(4096);
    let payload_mid_yes = payload_with_trigger(4096);
    let payload_large_no = payload_without_trigger(65536);
    let payload_large_yes = payload_with_trigger(65536);

    c.bench_function("fc_trigger_scan_256_no", |b| {
        b.iter(|| black_box(response_text_contains_trigger(black_box(&payload_small_no))));
    });
    c.bench_function("fc_trigger_scan_256_yes", |b| {
        b.iter(|| {
            black_box(response_text_contains_trigger(black_box(
                &payload_small_yes,
            )))
        });
    });
    c.bench_function("fc_trigger_scan_4k_no", |b| {
        b.iter(|| black_box(response_text_contains_trigger(black_box(&payload_mid_no))));
    });
    c.bench_function("fc_trigger_scan_4k_yes", |b| {
        b.iter(|| black_box(response_text_contains_trigger(black_box(&payload_mid_yes))));
    });
    c.bench_function("fc_trigger_scan_64k_no", |b| {
        b.iter(|| black_box(response_text_contains_trigger(black_box(&payload_large_no))));
    });
    c.bench_function("fc_trigger_scan_64k_yes", |b| {
        b.iter(|| {
            black_box(response_text_contains_trigger(black_box(
                &payload_large_yes,
            )))
        });
    });
}

fn bench_fc_parser(c: &mut Criterion) {
    let trigger = prompt::get_trigger_signal();
    let payload_function_call = fc_parse_payload_function_call();
    let payload_mixed = fc_parse_payload_mixed();

    c.bench_function("fc_parse_function_call_fast_path", |b| {
        b.iter(|| {
            black_box(
                parser::parse_function_calls(black_box(&payload_function_call), black_box(trigger))
                    .expect("fc parse"),
            )
        });
    });

    c.bench_function("fc_parse_mixed_fallback_path", |b| {
        b.iter(|| {
            black_box(
                parser::parse_function_calls(black_box(&payload_mixed), black_box(trigger))
                    .expect("fc parse"),
            )
        });
    });
}

fn bench_fc_detector(c: &mut Criterion) {
    let trigger = prompt::get_trigger_signal();
    let tool_parsing_no_open_first = format!("{trigger}{}", "x".repeat(64));
    let tool_parsing_with_open_first = format!("{trigger}<function_calls><function_call>");
    let chunk = "x".repeat(1024);

    c.bench_function("fc_detector_tool_parsing_no_open_1k", |b| {
        b.iter(|| {
            let mut detector = StreamingFcDetector::new(black_box(trigger));
            black_box(detector.feed(black_box(&tool_parsing_no_open_first)));
            black_box(detector.feed(black_box(&chunk)));
        });
    });

    c.bench_function("fc_detector_tool_parsing_with_open_1k", |b| {
        b.iter(|| {
            let mut detector = StreamingFcDetector::new(black_box(trigger));
            black_box(detector.feed(black_box(&tool_parsing_with_open_first)));
            black_box(detector.feed(black_box(&chunk)));
        });
    });
}

fn make_model_switch_bench_request(prompt_len: usize) -> CanonicalRequest {
    CanonicalRequest {
        request_id: uuid::Uuid::from_u128(1),
        ingress_api: IngressApi::OpenAiChat,
        model: "smart".to_string(),
        stream: false,
        system_prompt: None,
        messages: vec![CanonicalMessage {
            role: CanonicalRole::User,
            parts: smallvec![CanonicalPart::Text("x".repeat(prompt_len))],
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        }],
        tools: Arc::<[toolify_rs::protocol::canonical::CanonicalToolSpec]>::from([]),
        tool_choice: CanonicalToolChoice::None,
        generation: GenerationParams::default(),
        provider_extensions: None,
    }
}

fn bench_no_tools_model_switch(c: &mut Criterion) {
    let base = make_model_switch_bench_request(4096);
    let candidate_models = ["m_a", "m_b", "m_c", "m_d"];

    c.bench_function("no_tools_model_switch_clone_per_candidate_4x4k", |b| {
        b.iter(|| {
            for model in candidate_models {
                let mut req = base.clone();
                req.model.clear();
                req.model.push_str(black_box(model));
                black_box(req.messages.len());
            }
        });
    });

    let mut reusable = base.clone();
    c.bench_function("no_tools_model_switch_reuse_mut_4x4k", |b| {
        b.iter(|| {
            for model in candidate_models {
                reusable.model.clear();
                reusable.model.push_str(black_box(model));
                black_box(reusable.messages.len());
            }
        });
    });
}

criterion_group!(
    benches,
    bench_route_resolution,
    bench_route_sticky_hash,
    bench_authentication,
    bench_model_router_single_candidate,
    bench_start_candidate_index,
    bench_fc_trigger_scan,
    bench_fc_parser,
    bench_fc_detector,
    bench_no_tools_model_switch
);
criterion_main!(benches);
