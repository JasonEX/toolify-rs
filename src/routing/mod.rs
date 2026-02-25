pub mod dispatch;
pub(crate) mod policy;
pub mod session;

use std::sync::Arc;

use rustc_hash::FxHashMap;
use smallvec::{smallvec, SmallVec};

use crate::config::AppConfig;
use crate::error::CanonicalError;
use crate::util::mix_u64;

/// The resolved target for a request.
#[derive(Debug, Clone, Copy)]
pub struct RouteTarget<'a> {
    pub upstream_index: usize,
    pub actual_model: &'a str,
    pub known_model_id: Option<usize>,
}

/// A candidate entry in the model index: the upstream service index and the
/// actual model name to send to that upstream.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    upstream_index: usize,
    model_id: usize,
}

/// Pre-built model router that indexes all upstream models and aliases for
/// efficient lookup.
#[derive(Debug, Clone)]
pub struct ModelRouter {
    /// Maps a requested model name (or alias) to one or more candidates.
    model_index: FxHashMap<String, Vec<Candidate>>,
    /// Interned real model names used by candidates.
    interned_models: Vec<Arc<str>>,
    /// Fast path when the model index has exactly one key and one candidate.
    single_exact_route: Option<SingleExactRoute>,
}

#[derive(Debug, Clone)]
struct SingleExactRoute {
    model: Arc<str>,
    candidate: Candidate,
}

impl ModelRouter {
    /// Build a `ModelRouter` from the application configuration.
    #[must_use]
    pub fn new(config: &AppConfig) -> Self {
        let mut model_index: FxHashMap<String, Vec<Candidate>> = FxHashMap::default();
        let mut interned_models: Vec<Arc<str>> = Vec::new();
        let mut interned_index: FxHashMap<String, usize> = FxHashMap::default();

        for (idx, svc) in config.upstream_services.iter().enumerate() {
            for entry in &svc.models {
                if let Some(colon_pos) = entry.find(':') {
                    // Alias entry — "alias:real_model"
                    let alias = &entry[..colon_pos];
                    let real_model = &entry[colon_pos + 1..];
                    let model_id =
                        intern_model_id(real_model, &mut interned_index, &mut interned_models);
                    model_index
                        .entry(alias.to_string())
                        .or_default()
                        .push(Candidate {
                            upstream_index: idx,
                            model_id,
                        });
                } else {
                    // Exact model entry
                    let model_id =
                        intern_model_id(entry, &mut interned_index, &mut interned_models);
                    model_index
                        .entry(entry.clone())
                        .or_default()
                        .push(Candidate {
                            upstream_index: idx,
                            model_id,
                        });
                }
            }
        }

        let single_exact_route = if model_index.len() == 1 {
            model_index.iter().next().and_then(|(model, candidates)| {
                if candidates.len() == 1 {
                    Some(SingleExactRoute {
                        model: Arc::from(model.as_str()),
                        candidate: candidates[0],
                    })
                } else {
                    None
                }
            })
        } else {
            None
        };
        Self {
            model_index,
            interned_models,
            single_exact_route,
        }
    }

    #[must_use]
    pub fn known_model_count(&self) -> usize {
        self.interned_models.len()
    }

    /// Return whether route resolution for this model requires a request hash.
    ///
    /// Hashing is only needed when the model resolves to multiple candidates.
    #[must_use]
    pub fn requires_request_hash_for_ordering(&self, model: &str) -> bool {
        if self.single_exact_route.is_some() {
            return false;
        }
        self.model_index
            .get(model)
            .is_some_and(|candidates| candidates.len() > 1)
    }

    #[must_use]
    pub fn has_candidate_for_upstream(&self, model: &str, upstream_index: usize) -> bool {
        if let Some(single_route) = &self.single_exact_route {
            return model == single_route.model.as_ref()
                && single_route.candidate.upstream_index == upstream_index;
        }

        self.model_index.get(model).is_some_and(|candidates| {
            candidates
                .iter()
                .any(|candidate| candidate.upstream_index == upstream_index)
        })
    }

    /// Resolve which upstream service and actual model name to use for a given
    /// requested model.
    ///
    /// Resolution order:
    /// 1. Exact model match in the index.
    /// 2. Alias match — if a single candidate, use it directly; if multiple
    ///    candidates (alias group), pick one deterministically based on
    ///    `request_hash`.
    /// 3. No match — return an error.
    ///
    /// # Errors
    ///
    /// Returns `CanonicalError::InvalidRequest` when no route can be resolved.
    pub fn resolve<'a>(
        &'a self,
        model: &'a str,
        request_hash: u64,
    ) -> Result<RouteTarget<'a>, CanonicalError> {
        self.resolve_with_lazy_hash(model, || request_hash)
    }

    /// Resolve when and only when the model has a single candidate that does
    /// not require request-hash ordering.
    ///
    /// Returns `Ok(None)` for alias groups with multiple candidates.
    ///
    /// # Errors
    ///
    /// Returns `CanonicalError::InvalidRequest` when no route can be resolved.
    pub fn resolve_if_single_candidate<'a>(
        &'a self,
        model: &'a str,
    ) -> Result<Option<RouteTarget<'a>>, CanonicalError> {
        if let Some(single_route) = &self.single_exact_route {
            if model == single_route.model.as_ref() {
                return self.route_from_candidate(single_route.candidate).map(Some);
            }
            return Err(CanonicalError::InvalidRequest(format!(
                "No upstream found for model '{model}'"
            )));
        }

        if let Some(candidates) = self.model_index.get(model) {
            if candidates.len() == 1 {
                return self.route_from_candidate(candidates[0]).map(Some);
            }
            return Ok(None);
        }

        Err(CanonicalError::InvalidRequest(format!(
            "No upstream found for model '{model}'"
        )))
    }

    /// Resolve an ordered route list for failover.
    ///
    /// The first entry is the sticky-primary choice derived from `request_hash`.
    /// Remaining entries follow deterministic ring order for retry/failover.
    ///
    /// # Errors
    ///
    /// Returns `CanonicalError::InvalidRequest` when no route can be resolved.
    pub fn resolve_ordered<'a>(
        &'a self,
        model: &'a str,
        request_hash: u64,
    ) -> Result<SmallVec<[RouteTarget<'a>; 4]>, CanonicalError> {
        if let Some(single_route) = &self.single_exact_route {
            if model == single_route.model.as_ref() {
                return Ok(smallvec![self.route_from_candidate(single_route.candidate)?]);
            }
            return Err(CanonicalError::InvalidRequest(format!(
                "No upstream found for model '{model}'"
            )));
        }

        if let Some(candidates) = self.model_index.get(model) {
            if candidates.len() == 1 {
                return Ok(smallvec![self.route_from_candidate(candidates[0])?]);
            }

            let start = select_alias_group_index(candidates.len(), request_hash);
            let mut ordered = SmallVec::<[RouteTarget<'a>; 4]>::with_capacity(candidates.len());
            let mut seen_lo = 0_u64;
            let mut seen_hi = 0_u64;
            for candidate in candidates[start..].iter().chain(candidates[..start].iter()) {
                let upstream_index = candidate.upstream_index;
                if is_small_upstream_index(upstream_index) {
                    if !mark_unseen_upstream(upstream_index, &mut seen_lo, &mut seen_hi) {
                        continue;
                    }
                } else if ordered
                    .iter()
                    .any(|existing: &RouteTarget<'_>| existing.upstream_index == upstream_index)
                {
                    continue;
                }
                ordered.push(self.route_from_candidate(*candidate)?);
            }
            if !ordered.is_empty() {
                return Ok(ordered);
            }
        }

        Err(CanonicalError::InvalidRequest(format!(
            "No upstream found for model '{model}'"
        )))
    }

    /// Resolve routing target and lazily request a hash only when an alias
    /// group has multiple candidates.
    ///
    /// # Errors
    ///
    /// Returns `CanonicalError::InvalidRequest` when no route can be resolved.
    pub fn resolve_with_lazy_hash<'a, F>(
        &'a self,
        model: &'a str,
        mut request_hash: F,
    ) -> Result<RouteTarget<'a>, CanonicalError>
    where
        F: FnMut() -> u64,
    {
        if let Some(single_route) = &self.single_exact_route {
            if model == single_route.model.as_ref() {
                return self.route_from_candidate(single_route.candidate);
            }
            return Err(CanonicalError::InvalidRequest(format!(
                "No upstream found for model '{model}'"
            )));
        }

        // 1 & 2. Index lookup (covers both exact and alias matches)
        if let Some(candidates) = self.model_index.get(model) {
            let candidate = if candidates.len() == 1 {
                candidates[0]
            } else {
                *select_from_alias_group(candidates, request_hash())
            };
            return self.route_from_candidate(candidate);
        }

        // 3. No match
        Err(CanonicalError::InvalidRequest(format!(
            "No upstream found for model '{model}'"
        )))
    }

    fn route_from_candidate(
        &self,
        candidate: Candidate,
    ) -> Result<RouteTarget<'_>, CanonicalError> {
        let actual_model = self
            .interned_models
            .get(candidate.model_id)
            .map(AsRef::as_ref)
            .ok_or_else(|| {
                CanonicalError::InvalidRequest(format!(
                    "Invalid internal model id {}",
                    candidate.model_id
                ))
            })?;
        Ok(RouteTarget {
            upstream_index: candidate.upstream_index,
            actual_model,
            known_model_id: Some(candidate.model_id),
        })
    }
}

/// Deterministically select a candidate from an alias group based on the
/// request hash. The same `request_hash` always yields the same candidate,
/// which guarantees that retries for the same logical request hit the same
/// upstream (S3-I10).
fn select_from_alias_group(candidates: &[Candidate], request_hash: u64) -> &Candidate {
    &candidates[select_alias_group_index(candidates.len(), request_hash)]
}

#[inline]
fn select_alias_group_index(candidate_count: usize, request_hash: u64) -> usize {
    let hash = mix_u64(request_hash);
    let len_u64 = u64::try_from(candidate_count).unwrap_or(u64::MAX);
    usize::try_from(hash % len_u64).unwrap_or_default()
}

#[inline]
fn mark_unseen_upstream(upstream_index: usize, seen_lo: &mut u64, seen_hi: &mut u64) -> bool {
    if upstream_index < u64::BITS as usize {
        let mask = 1_u64 << upstream_index;
        let unseen = (*seen_lo & mask) == 0;
        *seen_lo |= mask;
        return unseen;
    }
    if upstream_index < (u64::BITS as usize * 2) {
        let shift = upstream_index - (u64::BITS as usize);
        let mask = 1_u64 << shift;
        let unseen = (*seen_hi & mask) == 0;
        *seen_hi |= mask;
        return unseen;
    }
    // Rare high-index fallback: caller should use linear duplicate check.
    false
}

#[inline]
const fn is_small_upstream_index(upstream_index: usize) -> bool {
    upstream_index < (u64::BITS as usize * 2)
}

fn intern_model_id(
    model: &str,
    interned_index: &mut FxHashMap<String, usize>,
    interned_models: &mut Vec<Arc<str>>,
) -> usize {
    if let Some(id) = interned_index.get(model) {
        return *id;
    }

    let id = interned_models.len();
    interned_models.push(Arc::from(model));
    interned_index.insert(model.to_string(), id);
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AppConfig, ClientAuthConfig, FcMode, FeaturesConfig, ServerConfig, UpstreamServiceConfig,
    };

    fn make_upstream(name: &str, models: Vec<&str>, is_default: bool) -> UpstreamServiceConfig {
        UpstreamServiceConfig {
            name: name.to_string(),
            provider: "openai".to_string(),
            base_url: format!("https://{name}.example.com"),
            api_key: "test-key".to_string(),
            models: models.into_iter().map(String::from).collect(),
            description: String::new(),
            is_default,
            fc_mode: FcMode::default(),
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        }
    }

    fn make_config(services: Vec<UpstreamServiceConfig>) -> AppConfig {
        AppConfig {
            server: ServerConfig::default(),
            upstream_services: services,
            client_authentication: ClientAuthConfig {
                allowed_keys: vec!["key".to_string()],
            },
            features: FeaturesConfig::default(),
        }
    }

    #[test]
    fn test_exact_model_match() {
        let config = make_config(vec![make_upstream("svc1", vec!["gpt-4o"], false)]);
        let router = ModelRouter::new(&config);
        let result = router.resolve("gpt-4o", 1).unwrap();
        assert_eq!(config.upstream_services[result.upstream_index].name, "svc1");
        assert_eq!(result.actual_model, "gpt-4o");
        assert!(result.known_model_id.is_some());
    }

    #[test]
    fn test_alias_single_candidate() {
        let config = make_config(vec![make_upstream("svc1", vec!["smart:gpt-4o"], false)]);
        let router = ModelRouter::new(&config);
        let result = router.resolve("smart", 1).unwrap();
        assert_eq!(config.upstream_services[result.upstream_index].name, "svc1");
        assert_eq!(result.actual_model, "gpt-4o");
        assert!(result.known_model_id.is_some());
    }

    #[test]
    fn test_alias_group_deterministic() {
        let config = make_config(vec![
            make_upstream("openai", vec!["smart:gpt-4o"], false),
            make_upstream("anthropic", vec!["smart:claude-3.5-sonnet"], false),
        ]);
        let router = ModelRouter::new(&config);
        let result1 = router.resolve("smart", 42).unwrap();
        let result2 = router.resolve("smart", 42).unwrap();
        // Same request hash must always resolve to the same candidate
        assert_eq!(result1.upstream_index, result2.upstream_index);
        assert_eq!(result1.actual_model, result2.actual_model);
        assert_eq!(result1.known_model_id, result2.known_model_id);
    }

    #[test]
    fn test_alias_group_different_hashes_can_differ() {
        // With enough different hashes, we should eventually hit both candidates.
        let config = make_config(vec![
            make_upstream("openai", vec!["smart:gpt-4o"], false),
            make_upstream("anthropic", vec!["smart:claude-3.5-sonnet"], false),
        ]);
        let router = ModelRouter::new(&config);
        let mut names = std::collections::HashSet::new();
        for request_hash in 0..100 {
            let result = router.resolve("smart", request_hash).unwrap();
            names.insert(config.upstream_services[result.upstream_index].name.clone());
        }
        assert!(
            names.len() > 1,
            "Expected alias group to resolve to multiple upstreams across different hashes"
        );
    }

    #[test]
    fn test_requires_request_hash_for_ordering() {
        let single = make_config(vec![make_upstream("svc1", vec!["m1"], false)]);
        let single_router = ModelRouter::new(&single);
        assert!(!single_router.requires_request_hash_for_ordering("m1"));
        assert!(!single_router.requires_request_hash_for_ordering("missing"));

        let alias_group = make_config(vec![
            make_upstream("openai", vec!["smart:gpt-4o"], false),
            make_upstream("anthropic", vec!["smart:claude-3.5-sonnet"], false),
        ]);
        let alias_router = ModelRouter::new(&alias_group);
        assert!(alias_router.requires_request_hash_for_ordering("smart"));
        assert!(!alias_router.requires_request_hash_for_ordering("missing"));
    }

    #[test]
    fn test_resolve_if_single_candidate() {
        let single = make_config(vec![make_upstream("svc1", vec!["m1"], false)]);
        let single_router = ModelRouter::new(&single);
        let single_route = single_router.resolve_if_single_candidate("m1").unwrap();
        assert!(single_route.is_some());
        assert!(single_router
            .resolve_if_single_candidate("missing")
            .is_err());

        let alias_group = make_config(vec![
            make_upstream("openai", vec!["smart:gpt-4o"], false),
            make_upstream("anthropic", vec!["smart:claude-3.5-sonnet"], false),
        ]);
        let alias_router = ModelRouter::new(&alias_group);
        let alias_route = alias_router.resolve_if_single_candidate("smart").unwrap();
        assert!(alias_route.is_none());
    }

    #[test]
    fn test_same_model_can_have_multiple_upstreams() {
        let config = make_config(vec![
            make_upstream("a", vec!["gpt-4o"], false),
            make_upstream("b", vec!["gpt-4o"], false),
        ]);
        let router = ModelRouter::new(&config);
        let ordered = router.resolve_ordered("gpt-4o", 7).unwrap();
        assert_eq!(ordered.len(), 2);
        assert!(router.has_candidate_for_upstream("gpt-4o", 0));
        assert!(router.has_candidate_for_upstream("gpt-4o", 1));
    }

    #[test]
    fn test_resolve_ordered_returns_ring_without_duplicates() {
        let config = make_config(vec![
            make_upstream("a", vec!["smart:m1"], false),
            make_upstream("b", vec!["smart:m2"], false),
            make_upstream("b-dup", vec!["smart:m2"], false),
        ]);
        let router = ModelRouter::new(&config);
        let ordered = router.resolve_ordered("smart", 123).unwrap();
        assert!(ordered.len() >= 2);
        let mut seen = std::collections::HashSet::new();
        for route in ordered {
            assert!(seen.insert(route.upstream_index));
        }
    }

    #[test]
    fn test_no_match_returns_error() {
        let config = make_config(vec![
            make_upstream("svc1", vec!["gpt-4o"], false),
            make_upstream("default-svc", vec!["other-model"], true),
        ]);
        let router = ModelRouter::new(&config);
        let result = router.resolve("unknown-model", 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_router_struct() {
        let config = make_config(vec![
            make_upstream("openai", vec!["gpt-4o", "smart:gpt-4o"], false),
            make_upstream(
                "anthropic",
                vec!["claude-3.5-sonnet", "smart:claude-3.5-sonnet"],
                true,
            ),
        ]);
        let router = ModelRouter::new(&config);

        // Index should have 3 keys: gpt-4o, claude-3.5-sonnet, smart
        assert!(router.model_index.contains_key("gpt-4o"));
        assert!(router.model_index.contains_key("claude-3.5-sonnet"));
        assert!(router.model_index.contains_key("smart"));
        // "smart" alias group should have 2 candidates
        assert_eq!(router.model_index["smart"].len(), 2);
        // known model pool should dedupe actual model strings
        assert_eq!(router.known_model_count(), 2);
    }
}
