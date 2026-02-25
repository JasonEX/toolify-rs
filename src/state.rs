mod fc_policy;
mod models_cache;
mod request_id;
mod route_breaker;

use std::sync::Arc;

use bytes::Bytes;
use smallvec::SmallVec;

use crate::auth::{authenticate, AllowedClientKeys};
use crate::config::AppConfig;
use crate::error::CanonicalError;
use crate::protocol::canonical::IngressApi;
use crate::routing::policy::{
    resolve_routes_with_policy as resolve_routes_with_policy_impl,
    resolve_routes_with_policy_all_allowed as resolve_routes_with_policy_all_allowed_impl,
    route_sticky_hash as route_sticky_hash_impl,
};
pub use crate::routing::session::SessionClass;
use crate::routing::{ModelRouter, RouteTarget};
use crate::transport::{HttpTransport, PreparedUpstream};
use crate::util::unix_now_secs;

pub use fc_policy::FcDecision;
use fc_policy::FcPolicyCache;
use models_cache::{
    build_dynamic_models_response_body, build_initial_models_response_body, ModelsCache,
};
use request_id::RequestIdGenerator;
use route_breaker::{should_try_alternate_upstream, RouteBreakerRegistry};

/// Shared application state accessible to all handlers.
pub struct AppState {
    pub config: AppConfig,
    pub transport: HttpTransport,
    pub model_router: ModelRouter,
    pub prepared_upstreams: Vec<PreparedUpstream>,
    routing: RoutingState,
    resilience: ResilienceState,
    caches: CacheState,
    infra: InfraState,
}

struct RoutingState {
    upstream_names: Vec<Arc<str>>,
}

struct ResilienceState {
    fc_policy_cache: FcPolicyCache,
    route_breakers: RouteBreakerRegistry,
}

struct CacheState {
    models_cache: ModelsCache,
}

struct InfraState {
    allowed_client_keys: AllowedClientKeys,
    request_ids: RequestIdGenerator,
}

impl AppState {
    #[must_use]
    pub fn new(
        config: AppConfig,
        transport: HttpTransport,
        model_router: ModelRouter,
        prepared_upstreams: Vec<PreparedUpstream>,
        allowed_client_keys: AllowedClientKeys,
    ) -> Self {
        let models_cache_ttl_secs = config.server.models_cache_ttl_secs;
        let models_response_body = build_initial_models_response_body(&config);
        let upstream_names: Vec<Arc<str>> = config
            .upstream_services
            .iter()
            .map(|upstream| Arc::from(upstream.name.as_str()))
            .collect();
        let known_model_count = model_router.known_model_count();
        let upstream_count = prepared_upstreams.len();
        let fc_policy_cache = FcPolicyCache::new(&config, upstream_count, known_model_count);

        Self {
            config,
            transport,
            model_router,
            prepared_upstreams,
            routing: RoutingState { upstream_names },
            resilience: ResilienceState {
                fc_policy_cache,
                route_breakers: RouteBreakerRegistry::new(upstream_count),
            },
            caches: CacheState {
                models_cache: ModelsCache::new(models_response_body, models_cache_ttl_secs),
            },
            infra: InfraState {
                allowed_client_keys,
                request_ids: RequestIdGenerator::new(),
            },
        }
    }

    pub fn next_request_seq(&self) -> u64 {
        self.infra.request_ids.next_seq()
    }

    /// Authenticate an ingress request using the prebuilt key index.
    ///
    /// # Errors
    ///
    /// Returns `CanonicalError::Auth` when the API key is missing or invalid.
    pub fn authenticate(
        &self,
        ingress: IngressApi,
        headers: &http::HeaderMap,
    ) -> Result<(), CanonicalError> {
        authenticate(ingress, headers, &self.infra.allowed_client_keys)
    }

    #[must_use]
    pub fn request_uuid(&self, request_seq: u64) -> uuid::Uuid {
        self.infra.request_ids.request_uuid(request_seq)
    }

    #[must_use]
    pub fn route_sticky_hash(
        &self,
        ingress: IngressApi,
        headers: &http::HeaderMap,
        model: &str,
        prompt_prefix: &[u8],
    ) -> u64 {
        route_sticky_hash_impl(ingress, headers, model, prompt_prefix)
    }

    /// Resolve route with session-aware failover policy.
    ///
    /// - Portable: same-provider candidates first, then cross-provider.
    /// - Anchored: same-provider only unless all same-provider candidates are unavailable.
    ///
    /// # Errors
    ///
    /// Returns `CanonicalError::InvalidRequest` when no route can be resolved.
    pub fn resolve_route_with_policy<'a>(
        &'a self,
        model: &'a str,
        request_hash: u64,
        session_class: SessionClass,
    ) -> Result<RouteTarget<'a>, CanonicalError> {
        self.resolve_routes_with_policy(model, request_hash, session_class)?
            .first()
            .copied()
            .ok_or_else(|| CanonicalError::InvalidRequest(format!("No upstream for '{model}'")))
    }

    /// Resolve ordered route candidates with session-aware failover policy.
    ///
    /// Ordering policy:
    /// - Same-provider candidates are always attempted first.
    /// - `Portable`: then cross-provider candidates.
    /// - `Anchored`: cross-provider candidates are appended only after same-provider
    ///   candidates so callers can degrade only after exhausting anchored routes.
    ///
    /// Breaker-open routes are kept at the tail of each tier as best-effort probes.
    ///
    /// # Errors
    ///
    /// Returns `CanonicalError::InvalidRequest` when no route can be resolved.
    pub fn resolve_routes_with_policy<'a>(
        &'a self,
        model: &'a str,
        request_hash: u64,
        session_class: SessionClass,
    ) -> Result<SmallVec<[RouteTarget<'a>; 4]>, CanonicalError> {
        if self.resilience.route_breakers.has_any_entries() {
            resolve_routes_with_policy_impl(
                &self.model_router,
                &self.prepared_upstreams,
                model,
                request_hash,
                session_class,
                |upstream_index, model_group| {
                    self.resilience
                        .route_breakers
                        .allows_route(upstream_index, model_group)
                },
            )
        } else {
            resolve_routes_with_policy_all_allowed_impl(
                &self.model_router,
                &self.prepared_upstreams,
                model,
                request_hash,
            )
        }
    }

    pub fn record_upstream_success(&self, upstream_index: usize, model_group: &str) {
        self.resilience
            .route_breakers
            .record_success(upstream_index, model_group);
    }

    pub fn record_upstream_failure(
        &self,
        upstream_index: usize,
        model_group: &str,
        err: &CanonicalError,
    ) {
        self.resilience
            .route_breakers
            .record_failure(upstream_index, model_group, err);
    }

    pub fn record_upstream_outcome<T>(
        &self,
        upstream_index: usize,
        model_group: &str,
        result: &Result<T, CanonicalError>,
    ) {
        self.resilience
            .route_breakers
            .record_outcome(upstream_index, model_group, result);
    }

    #[must_use]
    pub fn fc_decision(&self, route: &RouteTarget<'_>, has_tools: bool) -> FcDecision {
        self.resilience.fc_policy_cache.decision(route, has_tools)
    }

    #[must_use]
    pub fn upstream_name(&self, upstream_index: usize) -> &str {
        self.routing
            .upstream_names
            .get(upstream_index)
            .map_or("<unknown-upstream>", AsRef::as_ref)
    }

    #[must_use]
    pub fn auto_inject_cached(&self, route: &RouteTarget<'_>) -> bool {
        self.resilience.fc_policy_cache.auto_inject_cached(route)
    }

    pub fn mark_auto_inject(&self, route: &RouteTarget<'_>) {
        self.resilience.fc_policy_cache.mark_auto_inject(route);
    }

    #[must_use]
    pub fn should_try_alternate_upstream(&self, err: &CanonicalError) -> bool {
        should_try_alternate_upstream(err)
    }

    #[must_use]
    pub fn models_response_body(&self) -> Bytes {
        self.caches.models_cache.body()
    }

    pub async fn maybe_refresh_models_cache(&self) {
        let now = unix_now_secs();
        if !self.caches.models_cache.try_begin_refresh(now) {
            return;
        }

        if let Some(body) = build_dynamic_models_response_body(self).await {
            self.caches.models_cache.set_body(body);
        }

        self.caches.models_cache.finish_refresh();
    }
}
