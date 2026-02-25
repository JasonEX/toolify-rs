use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::config::{AppConfig, FcMode};
use crate::routing::RouteTarget;
use crate::util::unix_now_secs;

#[derive(Debug, Clone, Copy)]
enum FcPolicy {
    Inject,
    Native,
    Auto,
}

#[derive(Debug, Clone, Copy)]
pub struct FcDecision {
    pub fc_active: bool,
    pub auto_fallback_allowed: bool,
}

struct AutoInjectCacheShard {
    known_models: Vec<AtomicU64>,
    dynamic_models: RwLock<FxHashMap<String, u64>>,
}

pub(crate) struct FcPolicyCache {
    policies: Vec<FcPolicy>,
    auto_inject_cache: Vec<AutoInjectCacheShard>,
}

impl FcPolicyCache {
    #[must_use]
    pub(crate) fn new(config: &AppConfig, upstream_count: usize, known_model_count: usize) -> Self {
        let enable_fc = config.features.enable_function_calling;
        let policies: Vec<FcPolicy> = config
            .upstream_services
            .iter()
            .map(|upstream| {
                if enable_fc {
                    match upstream.fc_mode {
                        FcMode::Inject => FcPolicy::Inject,
                        FcMode::Native => FcPolicy::Native,
                        FcMode::Auto => FcPolicy::Auto,
                    }
                } else {
                    FcPolicy::Native
                }
            })
            .collect();

        let mut auto_inject_cache = Vec::with_capacity(upstream_count);
        for _ in 0..upstream_count {
            let mut known_models = Vec::with_capacity(known_model_count);
            known_models.resize_with(known_model_count, || AtomicU64::new(0));
            auto_inject_cache.push(AutoInjectCacheShard {
                known_models,
                dynamic_models: RwLock::new(FxHashMap::default()),
            });
        }

        Self {
            policies,
            auto_inject_cache,
        }
    }

    #[must_use]
    pub(crate) fn decision(&self, route: &RouteTarget<'_>, has_tools: bool) -> FcDecision {
        if !has_tools {
            return FcDecision {
                fc_active: false,
                auto_fallback_allowed: false,
            };
        }

        match self
            .policies
            .get(route.upstream_index)
            .copied()
            .unwrap_or(FcPolicy::Native)
        {
            FcPolicy::Inject => FcDecision {
                fc_active: true,
                auto_fallback_allowed: false,
            },
            FcPolicy::Native | FcPolicy::Auto => FcDecision {
                fc_active: self.auto_inject_cached(route),
                auto_fallback_allowed: true,
            },
        }
    }

    #[must_use]
    pub(crate) fn auto_inject_cached(&self, route: &RouteTarget<'_>) -> bool {
        let Some(shard) = self.auto_inject_cache.get(route.upstream_index) else {
            return false;
        };
        let now = unix_now_secs();

        if let Some(model_id) = route.known_model_id {
            return shard
                .known_models
                .get(model_id)
                .is_some_and(|expiry| expiry.load(Ordering::Relaxed) > now);
        }

        shard
            .dynamic_models
            .read()
            .get(route.actual_model)
            .is_some_and(|expiry| *expiry > now)
    }

    pub(crate) fn mark_auto_inject(&self, route: &RouteTarget<'_>) {
        const AUTO_INJECT_CACHE_TTL_SECS: u64 = 15 * 60;

        if let Some(shard) = self.auto_inject_cache.get(route.upstream_index) {
            let now = unix_now_secs();
            let expiry = now.saturating_add(AUTO_INJECT_CACHE_TTL_SECS);
            if let Some(model_id) = route.known_model_id {
                if let Some(slot) = shard.known_models.get(model_id) {
                    slot.store(expiry, Ordering::Relaxed);
                    return;
                }
            }
            let mut dynamic = shard.dynamic_models.write();
            dynamic.insert(route.actual_model.to_string(), expiry);
            if dynamic.len() > 128 {
                dynamic.retain(|_, model_expiry| *model_expiry > now);
            }
        }
    }
}
