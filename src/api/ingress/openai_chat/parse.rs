use crate::api::engine::pipeline::{parse_common_request_probe, CommonRequestProbe};
use crate::error::CanonicalError;
use crate::protocol::openai_chat::OpenAiChatRequest;

pub(crate) fn parse_openai_chat_probe(
    body: &bytes::Bytes,
) -> Result<CommonRequestProbe<'_>, CanonicalError> {
    parse_common_request_probe(body, "OpenAI Chat request")
}

pub(crate) fn parse_openai_chat_request_wire(
    body: &bytes::Bytes,
) -> Result<OpenAiChatRequest, CanonicalError> {
    serde_json::from_slice(body).map_err(|e| {
        CanonicalError::InvalidRequest(format!("Invalid OpenAI Chat request body: {e}"))
    })
}
