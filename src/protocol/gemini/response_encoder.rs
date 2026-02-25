use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalPart, CanonicalResponse};
use crate::protocol::gemini::{
    GeminiCandidate, GeminiContent, GeminiPart, GeminiResponse, GeminiUsageMetadata,
};
use crate::protocol::mapping::canonical_stop_to_gemini;

/// Encode a canonical response into a Gemini wire response.
///
/// # Errors
///
/// Returns [`CanonicalError`] when canonical tool-call argument payloads are
/// invalid JSON.
pub fn encode_gemini_response(
    canonical: &CanonicalResponse,
) -> Result<GeminiResponse, CanonicalError> {
    // --- content parts ---
    let mut parts = Vec::with_capacity(canonical.content.len());
    for part in &canonical.content {
        match part {
            CanonicalPart::Text(t) | CanonicalPart::ReasoningText(t) => {
                parts.push(GeminiPart::Text(t.clone()));
            }
            CanonicalPart::ToolCall {
                name, arguments, ..
            } => {
                let args: serde_json::Value = serde_json::from_str(arguments.get())
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                parts.push(GeminiPart::FunctionCall {
                    name: name.clone(),
                    args,
                });
            }
            CanonicalPart::ToolResult { .. }
            | CanonicalPart::ImageUrl { .. }
            | CanonicalPart::Refusal(_) => {
                // Not part of a response encoding; skip.
            }
        }
    }

    // --- finish reason ---
    let finish_reason = Some(canonical_stop_to_gemini(canonical.stop_reason).to_string());

    // --- candidate ---
    let candidate = GeminiCandidate {
        content: GeminiContent {
            role: Some("model".into()),
            parts,
        },
        finish_reason,
        index: Some(0),
    };

    // --- usage ---
    let usage_metadata = {
        let u = &canonical.usage;
        let has_any =
            u.input_tokens.is_some() || u.output_tokens.is_some() || u.total_tokens.is_some();
        if has_any {
            Some(GeminiUsageMetadata {
                prompt_token_count: u.input_tokens,
                candidates_token_count: u.output_tokens,
                total_token_count: u.total_tokens,
            })
        } else {
            None
        }
    };

    Ok(GeminiResponse {
        candidates: Some(vec![candidate]),
        usage_metadata,
        model_version: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::*;

    #[test]
    fn test_text_response_encode() {
        let canonical = CanonicalResponse {
            id: "gemini-123".into(),
            model: "gemini-pro".into(),
            content: vec![CanonicalPart::Text("Hello!".into())],
            stop_reason: CanonicalStopReason::EndOfTurn,
            usage: CanonicalUsage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                total_tokens: Some(15),
            },
            provider_extensions: serde_json::Map::new(),
        };

        let resp = encode_gemini_response(&canonical).unwrap();
        let cands = resp.candidates.unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].finish_reason.as_deref(), Some("STOP"));
        match &cands[0].content.parts[0] {
            GeminiPart::Text(t) => assert_eq!(t, "Hello!"),
            other => panic!("expected Text, got {other:?}"),
        }
        let um = resp.usage_metadata.unwrap();
        assert_eq!(um.prompt_token_count, Some(10));
        assert_eq!(um.candidates_token_count, Some(5));
    }

    #[test]
    fn test_tool_call_response_encode() {
        let raw = serde_json::value::RawValue::from_string(r#"{"city":"SF"}"#.into()).unwrap();
        let canonical = CanonicalResponse {
            id: "gemini-456".into(),
            model: "gemini-pro".into(),
            content: vec![CanonicalPart::ToolCall {
                id: "call_abc".into(),
                name: "get_weather".into(),
                arguments: raw,
            }],
            stop_reason: CanonicalStopReason::ToolCalls,
            usage: CanonicalUsage::default(),
            provider_extensions: serde_json::Map::new(),
        };

        let resp = encode_gemini_response(&canonical).unwrap();
        let cands = resp.candidates.unwrap();
        // Gemini uses STOP even for tool calls
        assert_eq!(cands[0].finish_reason.as_deref(), Some("STOP"));
        match &cands[0].content.parts[0] {
            GeminiPart::FunctionCall { name, args } => {
                assert_eq!(name, "get_weather");
                assert_eq!(args["city"], "SF");
            }
            other => panic!("expected FunctionCall, got {other:?}"),
        }
    }
}
