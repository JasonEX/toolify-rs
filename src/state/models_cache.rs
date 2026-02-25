use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bytes::Bytes;
use http::Method;
use parking_lot::RwLock;
use serde_json::Value;

use super::AppState;
use crate::config::{AppConfig, UpstreamServiceConfig};
use crate::routing::ModelRouter;
use crate::transport::{build_provider_headers_prepared, PreparedUpstream};

pub(crate) struct ModelsCache {
    body: RwLock<Bytes>,
    ttl_secs: u64,
    next_refresh_unix: AtomicU64,
    refreshing: AtomicBool,
}

impl ModelsCache {
    #[must_use]
    pub(crate) fn new(initial_body: Bytes, ttl_secs: u64) -> Self {
        Self {
            body: RwLock::new(initial_body),
            ttl_secs,
            next_refresh_unix: AtomicU64::new(0),
            refreshing: AtomicBool::new(false),
        }
    }

    #[must_use]
    pub(crate) fn body(&self) -> Bytes {
        self.body.read().clone()
    }

    pub(crate) fn set_body(&self, body: Bytes) {
        *self.body.write() = body;
    }

    #[must_use]
    pub(crate) fn try_begin_refresh(&self, now: u64) -> bool {
        if self.ttl_secs == 0 {
            return false;
        }

        let next = self.next_refresh_unix.load(Ordering::Relaxed);
        if now < next {
            return false;
        }
        if self.refreshing.swap(true, Ordering::AcqRel) {
            return false;
        }

        self.next_refresh_unix
            .store(now.saturating_add(self.ttl_secs), Ordering::Relaxed);
        true
    }

    pub(crate) fn finish_refresh(&self) {
        self.refreshing.store(false, Ordering::Release);
    }
}

pub(crate) fn build_initial_models_response_body(config: &AppConfig) -> Bytes {
    build_models_response_body_from_visible(&build_visible_models_from_config(config))
}

pub(crate) async fn build_dynamic_models_response_body(state: &AppState) -> Option<Bytes> {
    let mut visible_models = BTreeMap::new();
    let mut any_dynamic_success = false;

    for (index, service) in state.config.upstream_services.iter().enumerate() {
        insert_config_visible_models(&mut visible_models, service);
        let Some(prepared) = state.prepared_upstreams.get(index) else {
            continue;
        };

        if let Some(models) = fetch_upstream_models(state, prepared, service).await {
            any_dynamic_success = true;
            for model_id in models {
                if !model_routes_to_upstream(&state.model_router, &model_id, index) {
                    continue;
                }
                visible_models
                    .entry(model_id)
                    .or_insert_with(|| service.name.clone());
            }
        }
    }

    if !any_dynamic_success {
        return None;
    }
    Some(build_models_response_body_from_visible(&visible_models))
}

fn build_visible_models_from_config(config: &AppConfig) -> BTreeMap<String, String> {
    let mut visible_models = BTreeMap::new();
    for service in &config.upstream_services {
        insert_config_visible_models(&mut visible_models, service);
    }
    visible_models
}

fn insert_config_visible_models(
    visible_models: &mut BTreeMap<String, String>,
    service: &UpstreamServiceConfig,
) {
    for model_name in &service.models {
        let visible_model = model_name.split(':').next().unwrap_or(model_name);
        visible_models
            .entry(visible_model.to_string())
            .or_insert_with(|| service.name.clone());
    }
}

fn build_models_response_body_from_visible(visible_models: &BTreeMap<String, String>) -> Bytes {
    let models: Vec<Value> = visible_models
        .iter()
        .map(|(id, owned_by)| {
            serde_json::json!({
                "id": id.as_str(),
                "object": "model",
                "created": 1_677_610_602,
                "owned_by": owned_by.as_str(),
                "permission": [],
                "root": id.as_str(),
                "parent": null,
            })
        })
        .collect();

    let payload = serde_json::json!({
        "object": "list",
        "data": models,
    });
    serde_json::to_vec(&payload).map_or_else(
        |_| Bytes::from_static(br#"{"object":"list","data":[]}"#),
        Bytes::from,
    )
}

fn model_routes_to_upstream(
    model_router: &ModelRouter,
    model: &str,
    upstream_index: usize,
) -> bool {
    model_router.has_candidate_for_upstream(model, upstream_index)
}

async fn fetch_upstream_models(
    state: &AppState,
    prepared: &PreparedUpstream,
    service: &UpstreamServiceConfig,
) -> Option<Vec<String>> {
    let url = build_models_url(&service.base_url);
    let response = state
        .transport
        .send_request(
            &url,
            Method::GET,
            build_provider_headers_prepared(prepared),
            Bytes::new(),
            prepared.proxy_for(false),
        )
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }

    let body = response.bytes().await.ok()?;
    let payload: Value = serde_json::from_slice(&body).ok()?;
    let mut model_ids = extract_model_ids_from_payload(&payload);
    if model_ids.is_empty() {
        return None;
    }
    model_ids.sort_unstable();
    model_ids.dedup();
    Some(model_ids)
}

fn build_models_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if let Some(root) = trimmed.strip_suffix("/chat/completions") {
        return format!("{root}/models");
    }
    if let Some(root) = trimmed.strip_suffix("/responses") {
        return format!("{root}/models");
    }
    if let Some(root) = trimmed.strip_suffix("/messages") {
        return format!("{root}/models");
    }
    format!("{trimmed}/models")
}

fn extract_model_ids_from_payload(payload: &Value) -> Vec<String> {
    let mut out = Vec::new();

    if let Some(items) = payload.get("data").and_then(Value::as_array) {
        for item in items {
            let candidate = item
                .get("id")
                .and_then(Value::as_str)
                .or_else(|| item.get("name").and_then(Value::as_str));
            if let Some(model) = candidate.and_then(normalize_model_id) {
                out.push(model);
            }
        }
    }

    if let Some(items) = payload.get("models").and_then(Value::as_array) {
        for item in items {
            let candidate = item
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| item.get("id").and_then(Value::as_str));
            if let Some(model) = candidate.and_then(normalize_model_id) {
                out.push(model);
            }
        }
    }

    out
}

fn normalize_model_id(raw: &str) -> Option<String> {
    let mut model = raw.trim();
    if model.is_empty() {
        return None;
    }
    if let Some(stripped) = model.strip_prefix("models/") {
        model = stripped;
    }
    if let Some((name, action)) = model.split_once(':') {
        if matches!(action, "generateContent" | "streamGenerateContent") {
            model = name;
        }
    }
    if model.is_empty() {
        None
    } else {
        Some(model.to_string())
    }
}
