use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicU64;

use crate::error::CanonicalError;
use crate::protocol::canonical::{
    CanonicalPart, CanonicalResponse, CanonicalStopReason, CanonicalUsage,
};
use crate::protocol::gemini::{GeminiCandidate, GeminiPart, GeminiResponse, GeminiUsageMetadata};
use crate::protocol::mapping::gemini_stop_to_canonical;
use crate::util::next_generated_id;

static GENERATED_GEMINI_ID_SEQ: AtomicU64 = AtomicU64::new(1);

fn next_generated_gemini_id() -> String {
    next_generated_id("gemini", &GENERATED_GEMINI_ID_SEQ)
}

/// Decode a Gemini generateContent response into the canonical IR.
///
/// # Errors
///
/// Returns [`CanonicalError`] when no candidate is present or when tool-call
/// arguments cannot be converted into canonical raw JSON.
pub fn decode_gemini_response(
    response: &GeminiResponse,
    model: &str,
) -> Result<CanonicalResponse, CanonicalError> {
    let candidate = response
        .candidates
        .as_ref()
        .and_then(|c| c.first())
        .ok_or_else(|| CanonicalError::Translation("Gemini response has no candidates".into()))?;

    let (parts, has_function_call) = decode_candidate_parts_ref(candidate)?;

    // --- stop reason ---
    let stop_reason = decode_stop_reason(candidate.finish_reason.as_deref(), has_function_call);

    // --- usage ---
    let usage = decode_usage_ref(response.usage_metadata.as_ref());

    // --- response id ---
    let id = next_generated_gemini_id();

    Ok(CanonicalResponse {
        id,
        model: model.to_string(),
        content: parts,
        stop_reason,
        usage,
        provider_extensions: serde_json::Map::new(),
    })
}

/// Decode a Gemini generateContent response into canonical IR by consuming ownership.
///
/// # Errors
///
/// Returns [`CanonicalError`] when no candidate is present or when tool-call
/// arguments cannot be converted into canonical raw JSON.
pub fn decode_gemini_response_owned(
    response: GeminiResponse,
) -> Result<CanonicalResponse, CanonicalError> {
    let GeminiResponse {
        candidates,
        usage_metadata,
        model_version,
    } = response;
    let mut candidates = candidates
        .ok_or_else(|| CanonicalError::Translation("Gemini response has no candidates".into()))?;
    let candidate = candidates
        .drain(..)
        .next()
        .ok_or_else(|| CanonicalError::Translation("Gemini response has no candidates".into()))?;
    let stop_reason = decode_stop_reason(candidate.finish_reason.as_deref(), false);
    let (content, has_function_call) = decode_candidate_parts_owned(candidate)?;
    let stop_reason = if stop_reason == CanonicalStopReason::EndOfTurn && has_function_call {
        CanonicalStopReason::ToolCalls
    } else {
        stop_reason
    };
    let usage = decode_usage_owned(usage_metadata);

    Ok(CanonicalResponse {
        id: next_generated_gemini_id(),
        model: model_version.unwrap_or_default(),
        content,
        stop_reason,
        usage,
        provider_extensions: serde_json::Map::new(),
    })
}

fn decode_usage_ref(usage_metadata: Option<&GeminiUsageMetadata>) -> CanonicalUsage {
    usage_metadata
        .map(|usage| CanonicalUsage {
            input_tokens: usage.prompt_token_count,
            output_tokens: usage.candidates_token_count,
            total_tokens: usage.total_token_count,
        })
        .unwrap_or_default()
}

fn decode_usage_owned(usage_metadata: Option<GeminiUsageMetadata>) -> CanonicalUsage {
    usage_metadata
        .map(|usage| CanonicalUsage {
            input_tokens: usage.prompt_token_count,
            output_tokens: usage.candidates_token_count,
            total_tokens: usage.total_token_count,
        })
        .unwrap_or_default()
}

fn decode_stop_reason(finish_reason: Option<&str>, has_function_call: bool) -> CanonicalStopReason {
    finish_reason.map_or(
        if has_function_call {
            CanonicalStopReason::ToolCalls
        } else {
            CanonicalStopReason::EndOfTurn
        },
        |finish_reason| {
            let stop_reason = gemini_stop_to_canonical(finish_reason);
            if stop_reason == CanonicalStopReason::EndOfTurn && has_function_call {
                CanonicalStopReason::ToolCalls
            } else {
                stop_reason
            }
        },
    )
}

fn decode_candidate_parts_ref(
    candidate: &GeminiCandidate,
) -> Result<(Vec<CanonicalPart>, bool), CanonicalError> {
    let mut content = Vec::new();
    let mut has_function_call = false;
    let mut call_counter: usize = 0;
    let mut pending_calls_by_name: HashMap<String, VecDeque<String>> = HashMap::new();

    for part in &candidate.content.parts {
        match part {
            GeminiPart::Text(text) => {
                content.push(CanonicalPart::Text(text.clone()));
            }
            GeminiPart::FunctionCall { name, args } => {
                has_function_call = true;
                let call_id = format!("call_{call_counter}");
                call_counter += 1;
                pending_calls_by_name
                    .entry(name.clone())
                    .or_default()
                    .push_back(call_id.clone());
                let arguments = serde_json::value::to_raw_value(args).map_err(|e| {
                    CanonicalError::Translation(format!(
                        "Failed to convert Gemini function_call arguments to RawValue: {e}"
                    ))
                })?;
                content.push(CanonicalPart::ToolCall {
                    id: call_id,
                    name: name.clone(),
                    arguments,
                });
            }
            GeminiPart::FunctionResponse { name, response } => {
                let call_id = pending_calls_by_name
                    .get_mut(name)
                    .and_then(VecDeque::pop_front)
                    .unwrap_or_else(|| {
                        let fallback = format!("call_{call_counter}");
                        call_counter += 1;
                        fallback
                    });
                let output = serde_json::to_string(response).unwrap_or_else(|_| "{}".into());
                content.push(CanonicalPart::ToolResult {
                    tool_call_id: call_id,
                    content: output,
                });
            }
            GeminiPart::InlineData { .. } => {}
        }
    }

    Ok((content, has_function_call))
}

fn decode_candidate_parts_owned(
    candidate: GeminiCandidate,
) -> Result<(Vec<CanonicalPart>, bool), CanonicalError> {
    let mut content = Vec::new();
    let mut has_function_call = false;
    let mut call_counter: usize = 0;
    let mut pending_calls_by_name: HashMap<String, VecDeque<String>> = HashMap::new();

    for part in candidate.content.parts {
        match part {
            GeminiPart::Text(text) => {
                content.push(CanonicalPart::Text(text));
            }
            GeminiPart::FunctionCall { name, args } => {
                has_function_call = true;
                let call_id = format!("call_{call_counter}");
                call_counter += 1;
                pending_calls_by_name
                    .entry(name.clone())
                    .or_default()
                    .push_back(call_id.clone());
                let arguments = serde_json::value::to_raw_value(&args).map_err(|e| {
                    CanonicalError::Translation(format!(
                        "Failed to convert Gemini function_call arguments to RawValue: {e}"
                    ))
                })?;
                content.push(CanonicalPart::ToolCall {
                    id: call_id,
                    name,
                    arguments,
                });
            }
            GeminiPart::FunctionResponse { name, response } => {
                let call_id = pending_calls_by_name
                    .get_mut(&name)
                    .and_then(VecDeque::pop_front)
                    .unwrap_or_else(|| {
                        let fallback = format!("call_{call_counter}");
                        call_counter += 1;
                        fallback
                    });
                let output = serde_json::to_string(&response).unwrap_or_else(|_| "{}".into());
                content.push(CanonicalPart::ToolResult {
                    tool_call_id: call_id,
                    content: output,
                });
            }
            GeminiPart::InlineData { .. } => {}
        }
    }

    Ok((content, has_function_call))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::gemini::*;

    #[test]
    fn test_text_response_decode() {
        let resp = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: GeminiContent {
                    role: Some("model".into()),
                    parts: vec![GeminiPart::Text("Hello!".into())],
                },
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }]),
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
                total_token_count: Some(15),
            }),
            model_version: None,
        };

        let canonical = decode_gemini_response(&resp, "gemini-pro").unwrap();
        assert!(canonical.id.starts_with("gemini-"));
        assert_eq!(canonical.model, "gemini-pro");
        assert_eq!(canonical.stop_reason, CanonicalStopReason::EndOfTurn);
        assert_eq!(canonical.usage.input_tokens, Some(10));
        assert_eq!(canonical.usage.output_tokens, Some(5));
        match &canonical.content[0] {
            CanonicalPart::Text(t) => assert_eq!(t, "Hello!"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn test_function_call_response_decode() {
        let resp = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: GeminiContent {
                    role: Some("model".into()),
                    parts: vec![GeminiPart::FunctionCall {
                        name: "get_weather".into(),
                        args: serde_json::json!({"city": "SF"}),
                    }],
                },
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }]),
            usage_metadata: None,
            model_version: None,
        };

        let canonical = decode_gemini_response(&resp, "gemini-pro").unwrap();
        // STOP + FunctionCall => ToolCalls
        assert_eq!(canonical.stop_reason, CanonicalStopReason::ToolCalls);
        match &canonical.content[0] {
            CanonicalPart::ToolCall { id, name, .. } => {
                assert!(id.starts_with("call_"));
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn test_no_candidates_error() {
        let resp = GeminiResponse {
            candidates: None,
            usage_metadata: None,
            model_version: None,
        };

        let result = decode_gemini_response(&resp, "gemini-pro");
        assert!(result.is_err());
    }

    #[test]
    fn test_function_call_and_response_id_binding() {
        let resp = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: GeminiContent {
                    role: Some("model".into()),
                    parts: vec![
                        GeminiPart::FunctionCall {
                            name: "add_numbers".into(),
                            args: serde_json::json!({"a": 1, "b": 2}),
                        },
                        GeminiPart::FunctionResponse {
                            name: "add_numbers".into(),
                            response: serde_json::json!({"sum": 3}),
                        },
                        GeminiPart::FunctionResponse {
                            name: "add_numbers".into(),
                            response: serde_json::json!({"sum": 10}),
                        },
                    ],
                },
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }]),
            usage_metadata: None,
            model_version: None,
        };

        let canonical = decode_gemini_response(&resp, "gemini-pro").unwrap();
        assert_eq!(canonical.content.len(), 3);

        match &canonical.content[0] {
            CanonicalPart::ToolCall { id, name, .. } => {
                assert_eq!(id, "call_0");
                assert_eq!(name, "add_numbers");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }

        match &canonical.content[1] {
            CanonicalPart::ToolResult {
                tool_call_id,
                content,
            } => {
                assert_eq!(tool_call_id, "call_0");
                assert!(content.contains('3'));
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }

        match &canonical.content[2] {
            CanonicalPart::ToolResult { tool_call_id, .. } => {
                assert_eq!(tool_call_id, "call_1");
            }
            other => panic!("expected fallback ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_owned_matches_borrowed() {
        let response = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: GeminiContent {
                    role: Some("model".into()),
                    parts: vec![GeminiPart::Text("Hello!".into())],
                },
                finish_reason: Some("STOP".into()),
                index: Some(0),
            }]),
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
                total_token_count: Some(15),
            }),
            model_version: Some("gemini-pro".into()),
        };

        let borrowed = decode_gemini_response(&response, "gemini-pro").unwrap();
        let owned = decode_gemini_response_owned(response).unwrap();
        assert_eq!(borrowed.model, owned.model);
        assert_eq!(borrowed.content.len(), owned.content.len());
        assert!(matches!(
            (borrowed.content.first(), owned.content.first()),
            (Some(CanonicalPart::Text(left)), Some(CanonicalPart::Text(right))) if left == right
        ));
        assert_eq!(borrowed.stop_reason, owned.stop_reason);
        assert_eq!(borrowed.usage.total_tokens, owned.usage.total_tokens);
    }
}
