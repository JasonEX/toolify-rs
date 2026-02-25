use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};

use crate::config::FeaturesConfig;
use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::canonical::CanonicalToolSpec;
use crate::util::{mix_u64, sampled_bytes_hash};

use super::{
    decode_openai_wire_tool_choice, decode_openai_wire_tools, parse_openai_tool_choice_token,
    parse_openai_tools_token,
};

const SIMPLE_INJECT_CACHE_CAPACITY: usize = 16;
const SIMPLE_INJECT_CACHE_SET_COUNT: usize = 4;
const SIMPLE_INJECT_CACHE_SET_WAYS: usize =
    SIMPLE_INJECT_CACHE_CAPACITY / SIMPLE_INJECT_CACHE_SET_COUNT;
const SIMPLE_INJECT_CACHE_MAX_TOOLS_BYTES: usize = 64 * 1024;
const SIMPLE_INJECT_CACHE_MAX_TOOL_CHOICE_BYTES: usize = 4 * 1024;
static SIMPLE_INJECT_CACHE: LazyLock<
    [parking_lot::Mutex<VecDeque<SimpleInjectCacheEntry>>; SIMPLE_INJECT_CACHE_SET_COUNT],
> = LazyLock::new(|| {
    std::array::from_fn(|_| {
        parking_lot::Mutex::new(VecDeque::with_capacity(SIMPLE_INJECT_CACHE_SET_WAYS))
    })
});

#[derive(Clone)]
struct SimpleInjectCacheEntry {
    key_hash: u64,
    tools_token: Arc<[u8]>,
    tool_choice_token: Option<Arc<[u8]>>,
    saved_tools: Arc<[CanonicalToolSpec]>,
    prompt_artifacts: Arc<fc::prompt::PromptArtifacts>,
}

type SimpleInjectArtifacts = (Arc<[CanonicalToolSpec]>, Arc<fc::prompt::PromptArtifacts>);

thread_local! {
    static SIMPLE_INJECT_LAST_HIT: RefCell<Option<SimpleInjectCacheEntry>> =
        const { RefCell::new(None) };
}

fn simple_inject_tool_choice_token_eq(left: Option<&[u8]>, right: Option<&[u8]>) -> bool {
    match (left, right) {
        (Some(l), Some(r)) => l == r,
        (None, None) => true,
        _ => false,
    }
}

fn simple_inject_set_index(key_hash: u64) -> usize {
    let idx_u64 = key_hash & u64::try_from(SIMPLE_INJECT_CACHE_SET_COUNT - 1).unwrap_or(0);
    usize::try_from(idx_u64).unwrap_or(0)
}

fn simple_inject_cache_get(
    key_hash: u64,
    tools_token: &[u8],
    tool_choice_token: Option<&[u8]>,
) -> Option<SimpleInjectArtifacts> {
    if let Some(hit) = simple_inject_thread_cache_get(key_hash, tools_token, tool_choice_token) {
        return Some(hit);
    }

    let set = &SIMPLE_INJECT_CACHE[simple_inject_set_index(key_hash)];
    let mut guard = set.lock();
    let pos = guard.iter().rposition(|entry| {
        entry.key_hash == key_hash
            && entry.tools_token.as_ref() == tools_token
            && simple_inject_tool_choice_token_eq(
                entry.tool_choice_token.as_deref(),
                tool_choice_token,
            )
    })?;
    let move_to_back = pos + 1 != guard.len();
    let entry = if move_to_back {
        guard.remove(pos)?
    } else {
        guard.get(pos)?.clone()
    };
    if move_to_back {
        guard.push_back(entry.clone());
    }
    drop(guard);
    simple_inject_thread_cache_set(&entry);
    Some((
        Arc::clone(&entry.saved_tools),
        Arc::clone(&entry.prompt_artifacts),
    ))
}

fn simple_inject_cache_insert(
    key_hash: u64,
    tools_token: &[u8],
    tool_choice_token: Option<&[u8]>,
    saved_tools: Arc<[CanonicalToolSpec]>,
    prompt_artifacts: Arc<fc::prompt::PromptArtifacts>,
) {
    if tools_token.len() > SIMPLE_INJECT_CACHE_MAX_TOOLS_BYTES {
        return;
    }
    if tool_choice_token
        .is_some_and(|token| token.len() > SIMPLE_INJECT_CACHE_MAX_TOOL_CHOICE_BYTES)
    {
        return;
    }

    let set = &SIMPLE_INJECT_CACHE[simple_inject_set_index(key_hash)];
    let mut guard = set.lock();
    if let Some(pos) = guard.iter().position(|entry| {
        entry.key_hash == key_hash
            && entry.tools_token.as_ref() == tools_token
            && simple_inject_tool_choice_token_eq(
                entry.tool_choice_token.as_deref(),
                tool_choice_token,
            )
    }) {
        let _ = guard.remove(pos);
    }

    if guard.len() >= SIMPLE_INJECT_CACHE_SET_WAYS {
        let _ = guard.pop_front();
    }

    let entry = SimpleInjectCacheEntry {
        key_hash,
        tools_token: Arc::from(tools_token),
        tool_choice_token: tool_choice_token.map(Arc::from),
        saved_tools,
        prompt_artifacts,
    };
    simple_inject_thread_cache_set(&entry);
    guard.push_back(entry);
}

#[inline]
fn simple_inject_thread_cache_get(
    key_hash: u64,
    tools_token: &[u8],
    tool_choice_token: Option<&[u8]>,
) -> Option<SimpleInjectArtifacts> {
    SIMPLE_INJECT_LAST_HIT.with(|slot| {
        let guard = slot.borrow();
        let entry = guard.as_ref()?;
        if entry.key_hash != key_hash
            || entry.tools_token.as_ref() != tools_token
            || !simple_inject_tool_choice_token_eq(
                entry.tool_choice_token.as_deref(),
                tool_choice_token,
            )
        {
            return None;
        }
        Some((
            Arc::clone(&entry.saved_tools),
            Arc::clone(&entry.prompt_artifacts),
        ))
    })
}

#[inline]
fn simple_inject_thread_cache_set(entry: &SimpleInjectCacheEntry) {
    SIMPLE_INJECT_LAST_HIT.with(|slot| {
        *slot.borrow_mut() = Some(entry.clone());
    });
}

fn simple_inject_key_hash(tools_token: &[u8], tool_choice_token: Option<&[u8]>) -> u64 {
    let tools_hash = sampled_bytes_hash(tools_token);
    let choice_hash = tool_choice_token
        .map(sampled_bytes_hash)
        .map_or(0x01_u64, |hash| hash.rotate_left(13));
    mix_u64(tools_hash ^ 0x9e37_79b9_u64 ^ choice_hash)
}

fn simple_inject_cacheable(tools_token: &[u8], tool_choice_token: Option<&[u8]>) -> bool {
    tools_token.len() <= SIMPLE_INJECT_CACHE_MAX_TOOLS_BYTES
        && tool_choice_token
            .is_none_or(|token| token.len() <= SIMPLE_INJECT_CACHE_MAX_TOOL_CHOICE_BYTES)
}

pub(super) fn resolve_simple_inject_artifacts(
    tools_token: &[u8],
    tool_choice_token: Option<&[u8]>,
    features: &FeaturesConfig,
) -> Result<Option<SimpleInjectArtifacts>, CanonicalError> {
    let cacheable = simple_inject_cacheable(tools_token, tool_choice_token);
    let mut key_hash: Option<u64> = None;
    if cacheable {
        let hash = simple_inject_key_hash(tools_token, tool_choice_token);
        key_hash = Some(hash);
        if let Some(hit) = simple_inject_cache_get(hash, tools_token, tool_choice_token) {
            return Ok(Some(hit));
        }
    }

    let tool_choice = match tool_choice_token {
        Some(token) => {
            let parsed = parse_openai_tool_choice_token(token)?;
            decode_openai_wire_tool_choice(parsed.as_ref())
        }
        None => crate::protocol::canonical::CanonicalToolChoice::Auto,
    };
    if matches!(
        tool_choice,
        crate::protocol::canonical::CanonicalToolChoice::None
    ) {
        return Ok(None);
    }

    let saved_tools_vec = decode_openai_wire_tools(parse_openai_tools_token(tools_token)?);
    if saved_tools_vec.is_empty() {
        return Ok(None);
    }
    let prompt_artifacts = Arc::new(fc::prompt::generate_fc_prompt_artifacts(
        &saved_tools_vec,
        &tool_choice,
        features.prompt_template.as_deref(),
    )?);
    let saved_tools = Arc::<[CanonicalToolSpec]>::from(saved_tools_vec);

    if cacheable {
        let hash =
            key_hash.unwrap_or_else(|| simple_inject_key_hash(tools_token, tool_choice_token));
        simple_inject_cache_insert(
            hash,
            tools_token,
            tool_choice_token,
            Arc::clone(&saved_tools),
            Arc::clone(&prompt_artifacts),
        );
    }

    Ok(Some((saved_tools, prompt_artifacts)))
}
