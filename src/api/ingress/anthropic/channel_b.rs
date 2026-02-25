use std::sync::Arc;

use crate::protocol::canonical::IngressApi;
use crate::state::AppState;
use crate::transport::PreparedUpstream;

use crate::api::engine::channel_b::core::{run_channel_b_fast_path_uri_url, UriUrlEndpointConfig};
pub(crate) use crate::api::engine::channel_b::core::{ChannelBFastPathOutcome, ChannelBPlan};

const CONFIG: UriUrlEndpointConfig = UriUrlEndpointConfig {
    ingress: IngressApi::Anthropic,
    request_label: "Anthropic request",
    uri_getter: anthropic_uri_getter,
    url_getter: anthropic_url_getter,
    rewrite_model_field: true,
};

fn anthropic_uri_getter(prepared_upstream: &PreparedUpstream) -> Option<&http::Uri> {
    prepared_upstream.anthropic_messages_uri_parsed()
}

fn anthropic_url_getter(prepared_upstream: &PreparedUpstream) -> Option<&url::Url> {
    prepared_upstream.anthropic_messages_url_parsed()
}

pub(crate) async fn run_channel_b_fast_path<'a>(
    state: &Arc<AppState>,
    body: &bytes::Bytes,
    request_seq: &mut Option<u64>,
    plan: ChannelBPlan<'a>,
) -> ChannelBFastPathOutcome<'a> {
    run_channel_b_fast_path_uri_url(state, body, request_seq, plan, CONFIG).await
}
