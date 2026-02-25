use std::sync::Arc;

use axum::response::Response;
use smallvec::SmallVec;

use crate::api::common::CommonProbeRanges;
use crate::api::engine::pipeline::{encode_for_provider, UpstreamIoRequest};
use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalRequest, CanonicalToolSpec, ProviderKind};
use crate::routing::RouteTarget;
use crate::state::AppState;
use crate::transport::{
    build_provider_headers_prepared, build_upstream_url_prepared, static_parsed_upstream_uri,
    static_parsed_upstream_url,
};

use super::bootstrap::{should_continue_stream_failover, start_candidate_index};
use super::types::{AutoFallbackInput, CompatFlowSpec};

pub(crate) struct StreamFailoverInput<'a, S: CompatFlowSpec> {
    pub(crate) state: &'a Arc<AppState>,
    pub(crate) body: &'a bytes::Bytes,
    pub(crate) requested_model: &'a str,
    pub(crate) probe_ranges: Option<&'a CommonProbeRanges>,
    pub(crate) route_candidates: &'a [RouteTarget<'a>],
    pub(crate) route: RouteTarget<'a>,
    pub(crate) upstream_canonical: &'a CanonicalRequest,
    pub(crate) saved_tools: &'a [CanonicalToolSpec],
    pub(crate) fc_active: bool,
    pub(crate) auto_fallback_allowed: bool,
    pub(crate) request_seq: u64,
    pub(crate) request_id: uuid::Uuid,
    pub(crate) client_model: &'a str,
    pub(crate) wire_request: Option<&'a S::WireRequest>,
}

pub(crate) async fn run_stream_failover<S: CompatFlowSpec>(
    input: StreamFailoverInput<'_, S>,
) -> Result<Response, CanonicalError> {
    // Stream failover is only attempted before returning a response body to the client.
    let start_idx = start_candidate_index(input.route_candidates, input.route);
    let mut last_err: Option<CanonicalError> = None;
    let mut candidate_canonical = input.upstream_canonical.clone();
    let mut encoded_body_cache: SmallVec<[(ProviderKind, &str, bytes::Bytes); 4]> = SmallVec::new();
    for idx in start_idx..input.route_candidates.len() {
        let candidate_route = input.route_candidates[idx];
        let candidate_prepared_upstream =
            &input.state.prepared_upstreams[candidate_route.upstream_index];
        let candidate_provider = candidate_prepared_upstream.provider_kind();

        candidate_canonical.model.clear();
        candidate_canonical
            .model
            .push_str(candidate_route.actual_model);

        let candidate_url = build_upstream_url_prepared(
            candidate_prepared_upstream,
            candidate_route.actual_model,
            candidate_canonical.stream,
        );
        let candidate_parsed_url = static_parsed_upstream_url(
            candidate_prepared_upstream,
            candidate_route.actual_model,
            candidate_canonical.stream,
        );
        let candidate_hyper_uri = static_parsed_upstream_uri(
            candidate_prepared_upstream,
            candidate_route.actual_model,
            candidate_canonical.stream,
        );
        let proxy_url = candidate_prepared_upstream.proxy_for(candidate_canonical.stream);
        let candidate_headers = build_provider_headers_prepared(candidate_prepared_upstream);
        let io_ctx = UpstreamIoRequest {
            state: input.state.as_ref(),
            url: candidate_url.as_ref(),
            parsed_url: candidate_parsed_url,
            parsed_hyper_uri: candidate_hyper_uri,
            proxy_url,
            preconfigured_proxy_client: input.state.transport.preconfigured_proxy_client(proxy_url),
            upstream_headers: candidate_headers,
            provider: candidate_provider,
            client_model: input.client_model,
        };
        let candidate_body = encoded_body_for_candidate(
            &mut encoded_body_cache,
            candidate_provider,
            candidate_route.actual_model,
            &candidate_canonical,
        )?;
        let attempt_result = S::handle_streaming(
            io_ctx,
            candidate_body,
            input.request_seq,
            input.fc_active,
            input.saved_tools,
        )
        .await;

        if input.auto_fallback_allowed && !input.fc_active {
            match attempt_result {
                Ok(response) => {
                    input.state.record_upstream_success(
                        candidate_route.upstream_index,
                        input.requested_model,
                    );
                    return Ok(response);
                }
                Err(err) => {
                    let request_ref = input.wire_request.ok_or_else(|| {
                        CanonicalError::Internal(
                            "wire request missing for fallback retry".to_string(),
                        )
                    })?;
                    let fallback_result = S::run_auto_fallback(AutoFallbackInput {
                        state: input.state.as_ref(),
                        body: input.body,
                        wire_request: request_ref,
                        route: candidate_route,
                        client_model: input.client_model,
                        requested_model: input.requested_model,
                        request_seq: input.request_seq,
                        request_id: input.request_id,
                        probe_ranges: input.probe_ranges,
                        err,
                    })
                    .await;
                    match fallback_result {
                        Ok(response) => return Ok(response),
                        Err(fallback_err) => {
                            if should_continue_stream_failover(
                                input.state.as_ref(),
                                &fallback_err,
                                idx,
                                input.route_candidates.len(),
                            ) {
                                last_err = Some(fallback_err);
                                continue;
                            }
                            return Err(fallback_err);
                        }
                    }
                }
            }
        }

        input.state.record_upstream_outcome(
            candidate_route.upstream_index,
            input.requested_model,
            &attempt_result,
        );
        match attempt_result {
            Ok(response) => return Ok(response),
            Err(err) => {
                if should_continue_stream_failover(
                    input.state.as_ref(),
                    &err,
                    idx,
                    input.route_candidates.len(),
                ) {
                    last_err = Some(err);
                    continue;
                }
                return Err(err);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| {
        CanonicalError::Internal("No upstream candidate available for stream failover".to_string())
    }))
}

#[inline]
fn encoded_body_for_candidate<'a>(
    cache: &mut SmallVec<[(ProviderKind, &'a str, bytes::Bytes); 4]>,
    provider: ProviderKind,
    model: &'a str,
    canonical: &CanonicalRequest,
) -> Result<bytes::Bytes, CanonicalError> {
    if let Some((_, _, cached_body)) = cache.iter().find(|(cached_provider, cached_model, _)| {
        *cached_provider == provider && *cached_model == model
    }) {
        return Ok(cached_body.clone());
    }

    let encoded = encode_for_provider(provider, canonical)?;
    cache.push((provider, model, encoded.clone()));
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::protocol::canonical::{CanonicalToolChoice, GenerationParams, IngressApi};

    fn sample_canonical(model: &str) -> CanonicalRequest {
        CanonicalRequest {
            request_id: uuid::Uuid::nil(),
            ingress_api: IngressApi::OpenAiChat,
            model: model.to_string(),
            stream: true,
            system_prompt: None,
            messages: Vec::new(),
            tools: Arc::from([]),
            tool_choice: CanonicalToolChoice::Auto,
            generation: GenerationParams::default(),
            provider_extensions: None,
        }
    }

    #[test]
    fn encoded_body_cache_reuses_non_consecutive_provider_model_pair() {
        let mut cache: SmallVec<[(ProviderKind, &str, bytes::Bytes); 4]> = SmallVec::new();
        let mut canonical = sample_canonical("gpt-4.1");
        let first =
            encoded_body_for_candidate(&mut cache, ProviderKind::OpenAi, "gpt-4.1", &canonical)
                .expect("encode first");

        canonical.model = "gpt-4.1-mini".to_string();
        let _second = encoded_body_for_candidate(
            &mut cache,
            ProviderKind::OpenAi,
            "gpt-4.1-mini",
            &canonical,
        )
        .expect("encode second");

        canonical.model = "gpt-4.1".to_string();
        let third =
            encoded_body_for_candidate(&mut cache, ProviderKind::OpenAi, "gpt-4.1", &canonical)
                .expect("reuse first");

        assert_eq!(cache.len(), 2);
        assert_eq!(first, third);
    }

    #[test]
    fn encoded_body_cache_isolated_by_provider() {
        let mut cache: SmallVec<[(ProviderKind, &str, bytes::Bytes); 4]> = SmallVec::new();
        let canonical = sample_canonical("shared-model");

        let _openai = encoded_body_for_candidate(
            &mut cache,
            ProviderKind::OpenAi,
            "shared-model",
            &canonical,
        )
        .expect("encode openai");
        let _anthropic = encoded_body_for_candidate(
            &mut cache,
            ProviderKind::Anthropic,
            "shared-model",
            &canonical,
        )
        .expect("encode anthropic");
        let _openai_again = encoded_body_for_candidate(
            &mut cache,
            ProviderKind::OpenAi,
            "shared-model",
            &canonical,
        )
        .expect("reuse openai");

        assert_eq!(cache.len(), 2);
    }
}
