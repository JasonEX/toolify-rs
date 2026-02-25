use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalResponse, CanonicalToolSpec, IngressApi};
use crate::protocol::gemini::response_encoder::encode_gemini_response;
use crate::util::format_request_seq_hex;

use crate::api::engine::pipeline::{
    handle_non_streaming_common, handle_non_streaming_preencoded_common, handle_streaming_request,
    UpstreamIoRequest,
};

const INGRESS: IngressApi = IngressApi::Gemini;

pub(in crate::api) async fn handle_non_streaming(
    ctx: UpstreamIoRequest<'_>,
    upstream_canonical: &crate::protocol::canonical::CanonicalRequest,
    fc_active: bool,
    saved_tools: &[CanonicalToolSpec],
) -> Result<Response, CanonicalError> {
    handle_non_streaming_common(
        ctx,
        upstream_canonical,
        fc_active,
        saved_tools,
        false,
        INGRESS,
        encode_gemini_client_response,
    )
    .await
}

pub(in crate::api) async fn handle_non_streaming_preencoded(
    ctx: UpstreamIoRequest<'_>,
    upstream_body: bytes::Bytes,
    fc_active: bool,
    saved_tools: &[CanonicalToolSpec],
) -> Result<Response, CanonicalError> {
    handle_non_streaming_preencoded_common(
        ctx,
        upstream_body,
        fc_active,
        saved_tools,
        false,
        INGRESS,
        encode_gemini_client_response,
    )
    .await
}

pub(in crate::api) async fn handle_streaming(
    ctx: UpstreamIoRequest<'_>,
    upstream_body: bytes::Bytes,
    request_seq: u64,
    fc_active: bool,
    saved_tools: &[CanonicalToolSpec],
) -> Result<Response, CanonicalError> {
    handle_streaming_request(
        ctx,
        upstream_body,
        INGRESS,
        format_request_seq_hex("gemini-", request_seq),
        fc_active,
        saved_tools,
    )
    .await
}

fn encode_gemini_client_response(
    upstream_response: &CanonicalResponse,
    _client_model: &str,
) -> Result<Response, CanonicalError> {
    let client_response = encode_gemini_response(upstream_response)?;
    Ok(Json(client_response).into_response())
}
