use std::sync::LazyLock;

use crate::error::CanonicalError;
use crate::protocol::canonical::ProviderKind;

static TOOL_CALLS_FINDER: LazyLock<memchr::memmem::Finder<'static>> =
    LazyLock::new(|| memchr::memmem::Finder::new(br#""tool_calls""#));
static CONTENT_FINDER: LazyLock<memchr::memmem::Finder<'static>> =
    LazyLock::new(|| memchr::memmem::Finder::new(br#""content":""#));
static REFUSAL_FINDER: LazyLock<memchr::memmem::Finder<'static>> =
    LazyLock::new(|| memchr::memmem::Finder::new(br#""refusal":""#));

#[inline]
fn looks_like_openai_text_only_response(body: &[u8]) -> bool {
    TOOL_CALLS_FINDER.find(body).is_none()
        && (CONTENT_FINDER.find(body).is_some() || REFUSAL_FINDER.find(body).is_some())
}

pub(crate) fn encode_for_provider(
    provider: ProviderKind,
    canonical: &crate::protocol::canonical::CanonicalRequest,
) -> Result<bytes::Bytes, CanonicalError> {
    match provider {
        ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => {
            let wire =
                crate::protocol::openai_chat::encoder::encode_openai_chat_request(canonical)?;
            serde_json::to_vec(&wire)
                .map(bytes::Bytes::from)
                .map_err(|e| CanonicalError::Translation(format!("Serialization error: {e}")))
        }
        ProviderKind::Anthropic => {
            let wire = crate::protocol::anthropic::encoder::encode_anthropic_request(canonical)?;
            serde_json::to_vec(&wire)
                .map(bytes::Bytes::from)
                .map_err(|e| CanonicalError::Translation(format!("Serialization error: {e}")))
        }
        ProviderKind::Gemini => {
            let wire = crate::protocol::gemini::encoder::encode_gemini_request(canonical)?;
            serde_json::to_vec(&wire)
                .map(bytes::Bytes::from)
                .map_err(|e| CanonicalError::Translation(format!("Serialization error: {e}")))
        }
        ProviderKind::OpenAiResponses => {
            let wire =
                crate::protocol::openai_responses::encoder::encode_responses_request(canonical)?;
            serde_json::to_vec(&wire)
                .map(bytes::Bytes::from)
                .map_err(|e| CanonicalError::Translation(format!("Serialization error: {e}")))
        }
    }
}

/// Decode an upstream response body into a canonical response.
pub(crate) fn decode_response_from_provider(
    provider: ProviderKind,
    body: &[u8],
) -> Result<crate::protocol::canonical::CanonicalResponse, CanonicalError> {
    match provider {
        ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => {
            if looks_like_openai_text_only_response(body) {
                if let Some(decoded) =
                    crate::protocol::openai_chat::response_decoder::try_decode_openai_chat_text_response_bytes(body)
                {
                    return Ok(decoded);
                }
            }

            let wire: crate::protocol::openai_chat::OpenAiChatResponse =
                serde_json::from_slice(body).map_err(|e| {
                    CanonicalError::Translation(format!("Failed to parse OpenAI response: {e}"))
                })?;
            crate::protocol::openai_chat::response_decoder::decode_openai_chat_response_owned(wire)
        }
        ProviderKind::Anthropic => {
            let wire: crate::protocol::anthropic::AnthropicResponse = serde_json::from_slice(body)
                .map_err(|e| {
                    CanonicalError::Translation(format!("Failed to parse Anthropic response: {e}"))
                })?;
            crate::protocol::anthropic::response_decoder::decode_anthropic_response_owned(wire)
        }
        ProviderKind::Gemini => {
            let wire: crate::protocol::gemini::GeminiResponse = serde_json::from_slice(body)
                .map_err(|e| {
                    CanonicalError::Translation(format!("Failed to parse Gemini response: {e}"))
                })?;
            crate::protocol::gemini::response_decoder::decode_gemini_response_owned(wire)
        }
        ProviderKind::OpenAiResponses => {
            let wire: crate::protocol::openai_responses::ResponsesOutput =
                serde_json::from_slice(body).map_err(|e| {
                    CanonicalError::Translation(format!(
                        "Failed to parse Responses API output: {e}"
                    ))
                })?;
            crate::protocol::openai_responses::response_decoder::decode_responses_output_owned(wire)
        }
    }
}
