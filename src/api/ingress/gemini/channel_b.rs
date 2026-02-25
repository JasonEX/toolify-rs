use std::sync::Arc;

use crate::protocol::canonical::IngressApi;
use crate::state::AppState;
use crate::transport::PreparedUpstream;

use crate::api::engine::channel_b::core::{run_channel_b_fast_path_uri_url, UriUrlEndpointConfig};
pub(crate) use crate::api::engine::channel_b::core::{ChannelBFastPathOutcome, ChannelBPlan};

const CONFIG: UriUrlEndpointConfig = UriUrlEndpointConfig {
    ingress: IngressApi::Gemini,
    request_label: "Gemini request",
    uri_getter: gemini_uri_getter,
    url_getter: gemini_url_getter,
    rewrite_model_field: false,
};

fn gemini_uri_getter(_prepared_upstream: &PreparedUpstream) -> Option<&http::Uri> {
    None
}

fn gemini_url_getter(_prepared_upstream: &PreparedUpstream) -> Option<&url::Url> {
    None
}

pub(crate) async fn run_channel_b_fast_path<'a>(
    state: &Arc<AppState>,
    body: &bytes::Bytes,
    request_seq: &mut Option<u64>,
    plan: ChannelBPlan<'a>,
) -> ChannelBFastPathOutcome<'a> {
    run_channel_b_fast_path_uri_url(state, body, request_seq, plan, CONFIG).await
}
