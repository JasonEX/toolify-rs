use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use parking_lot::{RwLock, RwLockUpgradableReadGuard};
use rustc_hash::FxHashMap;

use crate::error::CanonicalError;
use crate::util::unix_now_secs;

#[derive(Debug, Clone, Copy, Default)]
struct RouteBreakerState {
    consecutive_failures: u32,
    open_until_unix: u64,
    half_open_probe_in_flight: bool,
}

pub(crate) struct RouteBreakerRegistry {
    shards: Vec<RwLock<FxHashMap<String, RouteBreakerState>>>,
    has_entries: Vec<AtomicBool>,
    active_shards: AtomicUsize,
}

const ROUTE_BREAKER_FAILURE_THRESHOLD: u32 = 5;

impl RouteBreakerRegistry {
    #[must_use]
    pub(crate) fn new(upstream_count: usize) -> Self {
        let mut shards = Vec::with_capacity(upstream_count);
        let mut has_entries = Vec::with_capacity(upstream_count);
        for _ in 0..upstream_count {
            shards.push(RwLock::new(FxHashMap::default()));
            has_entries.push(AtomicBool::new(false));
        }
        Self {
            shards,
            has_entries,
            active_shards: AtomicUsize::new(0),
        }
    }

    pub(crate) fn record_success(&self, upstream_index: usize, model_group: &str) {
        let (Some(shard), Some(has_entries)) = (
            self.shards.get(upstream_index),
            self.has_entries.get(upstream_index),
        ) else {
            return;
        };
        if !has_entries.load(Ordering::Acquire) {
            return;
        }
        let mut breaker_map = shard.write();
        let was_empty = breaker_map.is_empty();
        if was_empty {
            if has_entries.swap(false, Ordering::AcqRel) {
                self.decrement_active_shards();
            }
            return;
        }
        breaker_map.remove(model_group);
        if breaker_map.is_empty() && has_entries.swap(false, Ordering::AcqRel) {
            self.decrement_active_shards();
        }
    }

    pub(crate) fn record_failure(
        &self,
        upstream_index: usize,
        model_group: &str,
        err: &CanonicalError,
    ) {
        if !should_record_breaker_failure(err) {
            return;
        }
        let Some(shard) = self.shards.get(upstream_index) else {
            return;
        };

        let now = unix_now_secs();
        let mut breaker_map = shard.write();
        let was_empty = breaker_map.is_empty();
        let state = breaker_map.entry(model_group.to_string()).or_default();

        state.consecutive_failures = state.consecutive_failures.saturating_add(1);
        state.half_open_probe_in_flight = false;
        if state.consecutive_failures >= ROUTE_BREAKER_FAILURE_THRESHOLD {
            let open_secs = route_breaker_open_secs(state.consecutive_failures);
            state.open_until_unix = now.saturating_add(open_secs);
        }

        if breaker_map.len() > 256 {
            breaker_map.retain(|_, breaker_state| {
                breaker_state.open_until_unix > now || breaker_state.consecutive_failures > 0
            });
        }
        if was_empty {
            if let Some(has_entries) = self.has_entries.get(upstream_index) {
                if !has_entries.swap(true, Ordering::AcqRel) {
                    self.active_shards.fetch_add(1, Ordering::AcqRel);
                }
            }
        }
    }

    pub(crate) fn record_outcome<T>(
        &self,
        upstream_index: usize,
        model_group: &str,
        result: &Result<T, CanonicalError>,
    ) {
        match result {
            Ok(_) => self.record_success(upstream_index, model_group),
            Err(err) => self.record_failure(upstream_index, model_group, err),
        }
    }

    #[must_use]
    pub(crate) fn allows_route(&self, upstream_index: usize, model_group: &str) -> bool {
        self.allows_request(upstream_index, model_group)
    }

    #[must_use]
    pub(crate) fn has_any_entries(&self) -> bool {
        self.active_shards.load(Ordering::Acquire) != 0
    }

    fn allows_request(&self, upstream_index: usize, model_group: &str) -> bool {
        let (Some(shard), Some(has_entries)) = (
            self.shards.get(upstream_index),
            self.has_entries.get(upstream_index),
        ) else {
            return true;
        };
        if !has_entries.load(Ordering::Acquire) {
            return true;
        }

        let breaker_map = shard.upgradable_read();
        let Some(state) = breaker_map.get(model_group) else {
            return true;
        };

        if state.open_until_unix == 0 {
            return true;
        }

        let now = unix_now_secs();
        if now < state.open_until_unix {
            return false;
        }
        if state.half_open_probe_in_flight {
            return false;
        }

        let mut breaker_map = RwLockUpgradableReadGuard::upgrade(breaker_map);
        let Some(state) = breaker_map.get_mut(model_group) else {
            return true;
        };
        if state.open_until_unix == 0 {
            return true;
        }
        if now < state.open_until_unix {
            return false;
        }
        if state.half_open_probe_in_flight {
            return false;
        }

        state.half_open_probe_in_flight = true;
        true
    }

    fn decrement_active_shards(&self) {
        let mut current = self.active_shards.load(Ordering::Acquire);
        while current > 0 {
            match self.active_shards.compare_exchange_weak(
                current,
                current - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }
}

#[must_use]
pub(crate) fn should_try_alternate_upstream(err: &CanonicalError) -> bool {
    match err {
        CanonicalError::Transport(_) => true,
        CanonicalError::Upstream { status, .. } => {
            matches!(*status, 408 | 425 | 429 | 500 | 502 | 503 | 504 | 529)
        }
        _ => false,
    }
}

#[inline]
fn should_record_breaker_failure(err: &CanonicalError) -> bool {
    match err {
        CanonicalError::Transport(_) => true,
        CanonicalError::Upstream { status, .. } => {
            matches!(*status, 429 | 529) || (500..=599).contains(status)
        }
        _ => false,
    }
}

#[inline]
fn route_breaker_open_secs(consecutive_failures: u32) -> u64 {
    match consecutive_failures.saturating_sub(ROUTE_BREAKER_FAILURE_THRESHOLD) {
        0 => 5,
        1 => 15,
        2 => 45,
        _ => 120,
    }
}
