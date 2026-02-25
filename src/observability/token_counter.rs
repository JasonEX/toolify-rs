use crate::protocol::canonical::{CanonicalPart, CanonicalRequest, CanonicalUsage};
use std::time::Duration;
use tracing::info;

/// Estimate the number of tokens in `text` for the given model.
///
/// Uses a lightweight heuristic (`bytes / 4`) to avoid loading model BPE tables.
#[must_use]
pub fn estimate_tokens(text: &str, _model: &str) -> u64 {
    (text.len() as u64).div_ceil(4)
}

/// Estimate the total input tokens for a canonical request.
///
/// Sums tokens from the system prompt, all text parts of messages,
/// and serialized tool definitions.
#[must_use]
pub fn estimate_request_tokens(request: &CanonicalRequest) -> u64 {
    let model = &request.model;
    let mut total: u64 = 0;

    // System prompt
    if let Some(ref system) = request.system_prompt {
        total += estimate_tokens(system, model);
    }

    // Message text parts
    for msg in &request.messages {
        for part in &msg.parts {
            match part {
                CanonicalPart::Text(text)
                | CanonicalPart::ReasoningText(text)
                | CanonicalPart::Refusal(text) => {
                    total += estimate_tokens(text, model);
                }
                CanonicalPart::ToolResult { content, .. } => {
                    total += estimate_tokens(content, model);
                }
                CanonicalPart::ToolCall { arguments, .. } => {
                    total += estimate_tokens(arguments.get(), model);
                }
                CanonicalPart::ImageUrl { .. } => {
                    // Images are not counted via text tokenization
                }
            }
        }
    }

    // Tool definitions (serialized)
    for tool in request.tools.iter() {
        if let Ok(serialized) = serde_json::to_string(&tool.function.parameters) {
            total += estimate_tokens(&serialized, model);
        }
        if let Some(ref desc) = tool.function.description {
            total += estimate_tokens(desc, model);
        }
        total += estimate_tokens(&tool.function.name, model);
    }

    total
}

/// Merge upstream-reported usage with local estimates.
///
/// D6 rules:
/// - Always prefer upstream non-zero values
/// - Only fill `None` or zero fields with estimates
/// - Never overwrite non-zero upstream values
/// - Compute total = input + output if total is missing
#[must_use]
pub fn merge_usage(
    upstream: &CanonicalUsage,
    estimated_input: u64,
    estimated_output: u64,
) -> CanonicalUsage {
    let input_tokens = match upstream.input_tokens {
        Some(v) if v > 0 => Some(v),
        _ => Some(estimated_input),
    };

    let output_tokens = match upstream.output_tokens {
        Some(v) if v > 0 => Some(v),
        _ => Some(estimated_output),
    };

    let total_tokens = match upstream.total_tokens {
        Some(v) if v > 0 => Some(v),
        _ => {
            // Compute from the resolved input + output
            let i = input_tokens.unwrap_or(0);
            let o = output_tokens.unwrap_or(0);
            Some(i + o)
        }
    };

    CanonicalUsage {
        input_tokens,
        output_tokens,
        total_tokens,
    }
}

/// Log token usage for a completed request at INFO level.
pub fn log_request_usage(model: &str, usage: &CanonicalUsage, duration: Duration) {
    info!(
        model = model,
        input_tokens = usage.input_tokens.unwrap_or(0),
        output_tokens = usage.output_tokens.unwrap_or(0),
        total_tokens = usage.total_tokens.unwrap_or(0),
        duration_seconds = duration.as_secs_f64(),
        "request completed"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_nonempty() {
        let count = estimate_tokens("Hello, world!", "gpt-4");
        assert!(count > 0, "should estimate at least 1 token");
    }

    #[test]
    fn test_estimate_tokens_empty() {
        let count = estimate_tokens("", "gpt-4");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_estimate_tokens_unknown_model() {
        let count = estimate_tokens("Hello, world!", "totally-unknown-model-xyz");
        assert!(count > 0, "fallback should still produce tokens");
    }

    #[test]
    fn test_merge_usage_prefers_upstream() {
        let upstream = CanonicalUsage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            total_tokens: Some(150),
        };
        let merged = merge_usage(&upstream, 999, 999);
        assert_eq!(merged.input_tokens, Some(100));
        assert_eq!(merged.output_tokens, Some(50));
        assert_eq!(merged.total_tokens, Some(150));
    }

    #[test]
    fn test_merge_usage_fills_missing() {
        let upstream = CanonicalUsage {
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
        };
        let merged = merge_usage(&upstream, 40, 20);
        assert_eq!(merged.input_tokens, Some(40));
        assert_eq!(merged.output_tokens, Some(20));
        assert_eq!(merged.total_tokens, Some(60));
    }

    #[test]
    fn test_merge_usage_fills_zero() {
        let upstream = CanonicalUsage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            total_tokens: Some(0),
        };
        let merged = merge_usage(&upstream, 30, 10);
        assert_eq!(merged.input_tokens, Some(30));
        assert_eq!(merged.output_tokens, Some(10));
        assert_eq!(merged.total_tokens, Some(40));
    }

    #[test]
    fn test_merge_usage_partial_upstream() {
        let upstream = CanonicalUsage {
            input_tokens: Some(100),
            output_tokens: None,
            total_tokens: None,
        };
        let merged = merge_usage(&upstream, 50, 25);
        assert_eq!(merged.input_tokens, Some(100));
        assert_eq!(merged.output_tokens, Some(25));
        assert_eq!(merged.total_tokens, Some(125));
    }
}
