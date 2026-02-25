use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalResponse, CanonicalToolSpec, IngressApi};
use crate::protocol::openai_chat::response_encoder::encode_openai_chat_response;
use crate::util::format_request_seq_hex;

use crate::api::engine::pipeline::{
    handle_non_streaming_common, handle_non_streaming_preencoded_common, handle_streaming_request,
    UpstreamIoRequest,
};

const INGRESS: IngressApi = IngressApi::OpenAiChat;

pub(in crate::api) async fn handle_non_streaming(
    ctx: UpstreamIoRequest<'_>,
    upstream_canonical: &crate::protocol::canonical::CanonicalRequest,
    fc_active: bool,
    saved_tools: &[CanonicalToolSpec],
    response_model_passthrough_ok: bool,
) -> Result<Response, CanonicalError> {
    handle_non_streaming_common(
        ctx,
        upstream_canonical,
        fc_active,
        saved_tools,
        response_model_passthrough_ok,
        INGRESS,
        encode_openai_chat_client_response,
    )
    .await
}

pub(in crate::api) async fn handle_non_streaming_preencoded(
    ctx: UpstreamIoRequest<'_>,
    upstream_body: bytes::Bytes,
    fc_active: bool,
    saved_tools: &[CanonicalToolSpec],
    response_model_passthrough_ok: bool,
) -> Result<Response, CanonicalError> {
    handle_non_streaming_preencoded_common(
        ctx,
        upstream_body,
        fc_active,
        saved_tools,
        response_model_passthrough_ok,
        INGRESS,
        encode_openai_chat_client_response,
    )
    .await
}

fn encode_openai_chat_client_response(
    upstream_response: &CanonicalResponse,
    client_model: &str,
) -> Result<Response, CanonicalError> {
    let client_response = encode_openai_chat_response(upstream_response, client_model)?;
    Ok(Json(client_response).into_response())
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
        format_request_seq_hex("chatcmpl-", request_seq),
        fc_active,
        saved_tools,
    )
    .await
}
