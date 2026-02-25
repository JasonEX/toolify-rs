use std::cell::Cell;
use std::hash::Hasher;

use smallvec::SmallVec;

use crate::auth::extract_api_key_bytes_for_hash;
use crate::error::CanonicalError;
use crate::protocol::canonical::IngressApi;
use crate::transport::PreparedUpstream;
use crate::util::{mix_u64, unix_now_secs};

use super::session::SessionClass;
use super::{ModelRouter, RouteTarget};

const ROUTE_STICKY_BUCKET_SECS: u64 = 15 * 60;
const ROUTE_STICKY_PREFIX_MAX_BYTES: usize = 256;
const ROUTE_STICKY_REFRESH_MASK: u64 = (1_u64 << 10) - 1;

thread_local! {
    static ROUTE_STICKY_BUCKET_AND_TICK: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

#[inline]
fn sticky_bucket_now() -> u64 {
    ROUTE_STICKY_BUCKET_AND_TICK.with(|cache| {
        let (cached, old_tick) = cache.get();
        let tick = old_tick.wrapping_add(1);
        if cached != 0 && (tick & ROUTE_STICKY_REFRESH_MASK) != 0 {
            cache.set((cached, tick));
            return cached;
        }

        let fresh = unix_now_secs() / ROUTE_STICKY_BUCKET_SECS;
        cache.set((fresh, tick));
        fresh
    })
}

#[must_use]
pub(crate) fn route_sticky_hash(
    ingress: IngressApi,
    headers: &http::HeaderMap,
    model: &str,
    prompt_prefix: &[u8],
) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    if let Some(client_key) = extract_api_key_bytes_for_hash(ingress, headers) {
        hasher.write(client_key);
    }
    hasher.write_u8(0xff);
    hasher.write(model.as_bytes());
    hasher.write_u8(0xfe);
    let prefix_len = prompt_prefix.len().min(ROUTE_STICKY_PREFIX_MAX_BYTES);
    hasher.write(&prompt_prefix[..prefix_len]);
    hasher.write_u64(sticky_bucket_now());
    mix_u64(hasher.finish())
}

pub(crate) fn resolve_routes_with_policy<'a, F>(
    model_router: &'a ModelRouter,
    prepared_upstreams: &[PreparedUpstream],
    model: &'a str,
    request_hash: u64,
    session_class: SessionClass,
    mut allows_route: F,
) -> Result<SmallVec<[RouteTarget<'a>; 4]>, CanonicalError>
where
    F: FnMut(usize, &str) -> bool,
{
    let ordered = model_router.resolve_ordered(model, request_hash)?;
    if ordered.is_empty() {
        return Err(CanonicalError::InvalidRequest(format!(
            "No upstream found for model '{model}'"
        )));
    }
    if ordered.len() == 1 {
        return Ok(ordered);
    }

    let primary_provider = prepared_upstreams[ordered[0].upstream_index].provider_kind();

    let mut final_order = SmallVec::<[RouteTarget<'a>; 4]>::with_capacity(ordered.len());
    let mut cross_allowed = SmallVec::<[RouteTarget<'a>; 4]>::new();
    let mut same_blocked: Option<SmallVec<[RouteTarget<'a>; 4]>> = None;
    let mut cross_blocked: Option<SmallVec<[RouteTarget<'a>; 4]>> = None;
    for route in ordered {
        let breaker_allowed = allows_route(route.upstream_index, model);
        let same_provider =
            prepared_upstreams[route.upstream_index].provider_kind() == primary_provider;
        if same_provider {
            if breaker_allowed {
                final_order.push(route);
            } else {
                same_blocked
                    .get_or_insert_with(SmallVec::<[RouteTarget<'a>; 4]>::new)
                    .push(route);
            }
        } else if breaker_allowed {
            cross_allowed.push(route);
        } else {
            cross_blocked
                .get_or_insert_with(SmallVec::<[RouteTarget<'a>; 4]>::new)
                .push(route);
        }
    }

    if matches!(session_class, SessionClass::Portable) {
        final_order.extend(cross_allowed);
        if let Some(blocked) = same_blocked {
            final_order.extend(blocked);
        }
        if let Some(blocked) = cross_blocked {
            final_order.extend(blocked);
        }
    } else {
        if let Some(blocked) = same_blocked {
            final_order.extend(blocked);
        }
        // Anchored degrade mode: only after all same-provider candidates are exhausted.
        final_order.extend(cross_allowed);
        if let Some(blocked) = cross_blocked {
            final_order.extend(blocked);
        }
    }

    Ok(final_order)
}

pub(crate) fn resolve_routes_with_policy_all_allowed<'a>(
    model_router: &'a ModelRouter,
    prepared_upstreams: &[PreparedUpstream],
    model: &'a str,
    request_hash: u64,
) -> Result<SmallVec<[RouteTarget<'a>; 4]>, CanonicalError> {
    let ordered = model_router.resolve_ordered(model, request_hash)?;
    if ordered.is_empty() {
        return Err(CanonicalError::InvalidRequest(format!(
            "No upstream found for model '{model}'"
        )));
    }
    if ordered.len() == 1 {
        return Ok(ordered);
    }

    let primary_provider = prepared_upstreams[ordered[0].upstream_index].provider_kind();
    let mut final_order = SmallVec::<[RouteTarget<'a>; 4]>::with_capacity(ordered.len());
    let mut saw_cross_provider = false;
    for route in &ordered {
        if prepared_upstreams[route.upstream_index].provider_kind() == primary_provider {
            final_order.push(*route);
        } else {
            saw_cross_provider = true;
        }
    }
    if !saw_cross_provider {
        return Ok(final_order);
    }
    for route in &ordered {
        if prepared_upstreams[route.upstream_index].provider_kind() != primary_provider {
            final_order.push(*route);
        }
    }
    Ok(final_order)
}
