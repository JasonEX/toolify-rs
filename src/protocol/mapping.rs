use super::canonical::{CanonicalRole, CanonicalStopReason, CanonicalUsage};

// ---------------------------------------------------------------------------
// Role mappings
// ---------------------------------------------------------------------------

#[must_use]
pub fn canonical_role_to_openai(role: CanonicalRole) -> &'static str {
    match role {
        CanonicalRole::System => "system",
        CanonicalRole::User => "user",
        CanonicalRole::Assistant => "assistant",
        CanonicalRole::Tool => "tool",
    }
}

#[must_use]
pub fn openai_role_to_canonical(s: &str) -> CanonicalRole {
    match s {
        "system" | "developer" => CanonicalRole::System,
        "assistant" => CanonicalRole::Assistant,
        "tool" => CanonicalRole::Tool,
        _ => CanonicalRole::User, // fallback
    }
}

#[must_use]
pub fn canonical_role_to_anthropic(role: CanonicalRole) -> &'static str {
    match role {
        // Anthropic has no system role in messages; system is top-level.
        CanonicalRole::System | CanonicalRole::User | CanonicalRole::Tool => "user", // system messages are handled separately; tool results are user messages in Anthropic
        CanonicalRole::Assistant => "assistant",
    }
}

#[must_use]
pub fn anthropic_role_to_canonical(s: &str) -> CanonicalRole {
    match s {
        "assistant" => CanonicalRole::Assistant,
        _ => CanonicalRole::User,
    }
}

#[must_use]
pub fn canonical_role_to_gemini(role: CanonicalRole) -> &'static str {
    match role {
        CanonicalRole::System | CanonicalRole::User => "user", // system is handled via systemInstruction
        CanonicalRole::Assistant => "model",
        CanonicalRole::Tool => "function", // tool results use "function" role
    }
}

#[must_use]
pub fn gemini_role_to_canonical(s: &str) -> CanonicalRole {
    match s {
        "model" => CanonicalRole::Assistant,
        "function" => CanonicalRole::Tool,
        _ => CanonicalRole::User,
    }
}

// ---------------------------------------------------------------------------
// Stop reason mappings
// ---------------------------------------------------------------------------

#[must_use]
pub fn canonical_stop_to_openai(reason: CanonicalStopReason) -> &'static str {
    match reason {
        CanonicalStopReason::EndOfTurn => "stop",
        CanonicalStopReason::ToolCalls => "tool_calls",
        CanonicalStopReason::MaxTokens => "length",
        CanonicalStopReason::ContentFilter => "content_filter",
    }
}

#[must_use]
pub fn openai_stop_to_canonical(s: &str) -> CanonicalStopReason {
    match s {
        "tool_calls" => CanonicalStopReason::ToolCalls,
        "length" => CanonicalStopReason::MaxTokens,
        "content_filter" => CanonicalStopReason::ContentFilter,
        _ => CanonicalStopReason::EndOfTurn,
    }
}

#[must_use]
pub fn canonical_stop_to_anthropic(reason: CanonicalStopReason) -> &'static str {
    match reason {
        CanonicalStopReason::EndOfTurn | CanonicalStopReason::ContentFilter => "end_turn", // Anthropic has no content_filter reason
        CanonicalStopReason::ToolCalls => "tool_use",
        CanonicalStopReason::MaxTokens => "max_tokens",
    }
}

#[must_use]
pub fn anthropic_stop_to_canonical(s: &str) -> CanonicalStopReason {
    match s {
        "tool_use" => CanonicalStopReason::ToolCalls,
        "max_tokens" => CanonicalStopReason::MaxTokens,
        _ => CanonicalStopReason::EndOfTurn,
    }
}

#[must_use]
pub fn canonical_stop_to_gemini(reason: CanonicalStopReason) -> &'static str {
    match reason {
        CanonicalStopReason::EndOfTurn | CanonicalStopReason::ToolCalls => "STOP", // Gemini uses STOP even for tool calls
        CanonicalStopReason::MaxTokens => "MAX_TOKENS",
        CanonicalStopReason::ContentFilter => "SAFETY",
    }
}

#[must_use]
pub fn gemini_stop_to_canonical(s: &str) -> CanonicalStopReason {
    match s {
        "MAX_TOKENS" => CanonicalStopReason::MaxTokens,
        "SAFETY" | "RECITATION" => CanonicalStopReason::ContentFilter,
        _ => CanonicalStopReason::EndOfTurn,
    }
}

// ---------------------------------------------------------------------------
// Usage mappings
// ---------------------------------------------------------------------------

/// Convert canonical usage to OpenAI-style usage fields.
#[must_use]
pub fn canonical_usage_to_openai(usage: &CanonicalUsage) -> serde_json::Value {
    serde_json::json!({
        "prompt_tokens": usage.input_tokens.unwrap_or(0),
        "completion_tokens": usage.output_tokens.unwrap_or(0),
        "total_tokens": usage.total_tokens.unwrap_or(0),
    })
}

/// Convert OpenAI-style usage JSON to canonical usage.
#[must_use]
pub fn openai_usage_to_canonical(val: &serde_json::Value) -> CanonicalUsage {
    CanonicalUsage {
        input_tokens: val.get("prompt_tokens").and_then(serde_json::Value::as_u64),
        output_tokens: val
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64),
        total_tokens: val.get("total_tokens").and_then(serde_json::Value::as_u64),
    }
}

/// Convert canonical usage to Anthropic-style usage fields.
#[must_use]
pub fn canonical_usage_to_anthropic(usage: &CanonicalUsage) -> serde_json::Value {
    serde_json::json!({
        "input_tokens": usage.input_tokens.unwrap_or(0),
        "output_tokens": usage.output_tokens.unwrap_or(0),
    })
}

/// Convert Anthropic-style usage JSON to canonical usage.
#[must_use]
pub fn anthropic_usage_to_canonical(val: &serde_json::Value) -> CanonicalUsage {
    let input = val.get("input_tokens").and_then(serde_json::Value::as_u64);
    let output = val.get("output_tokens").and_then(serde_json::Value::as_u64);
    CanonicalUsage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: match (input, output) {
            (Some(i), Some(o)) => Some(i + o),
            _ => None,
        },
    }
}

/// Convert canonical usage to Gemini-style usage fields.
#[must_use]
pub fn canonical_usage_to_gemini(usage: &CanonicalUsage) -> serde_json::Value {
    serde_json::json!({
        "promptTokenCount": usage.input_tokens.unwrap_or(0),
        "candidatesTokenCount": usage.output_tokens.unwrap_or(0),
        "totalTokenCount": usage.total_tokens.unwrap_or(0),
    })
}

/// Convert Gemini-style usage JSON to canonical usage.
#[must_use]
pub fn gemini_usage_to_canonical(val: &serde_json::Value) -> CanonicalUsage {
    CanonicalUsage {
        input_tokens: val
            .get("promptTokenCount")
            .and_then(serde_json::Value::as_u64),
        output_tokens: val
            .get("candidatesTokenCount")
            .and_then(serde_json::Value::as_u64),
        total_tokens: val
            .get("totalTokenCount")
            .and_then(serde_json::Value::as_u64),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Role bijectivity tests ---

    #[test]
    fn test_openai_role_roundtrip() {
        for role in [
            CanonicalRole::System,
            CanonicalRole::User,
            CanonicalRole::Assistant,
            CanonicalRole::Tool,
        ] {
            let wire = canonical_role_to_openai(role);
            let back = openai_role_to_canonical(wire);
            assert_eq!(role, back, "OpenAI role roundtrip failed for {role:?}");
        }
    }

    #[test]
    fn test_gemini_role_roundtrip_user_assistant() {
        // Only user and assistant roundtrip cleanly; system and tool are special-cased.
        assert_eq!(
            gemini_role_to_canonical(canonical_role_to_gemini(CanonicalRole::User)),
            CanonicalRole::User
        );
        assert_eq!(
            gemini_role_to_canonical(canonical_role_to_gemini(CanonicalRole::Assistant)),
            CanonicalRole::Assistant
        );
        assert_eq!(
            gemini_role_to_canonical(canonical_role_to_gemini(CanonicalRole::Tool)),
            CanonicalRole::Tool
        );
    }

    #[test]
    fn test_anthropic_role_roundtrip_user_assistant() {
        assert_eq!(
            anthropic_role_to_canonical(canonical_role_to_anthropic(CanonicalRole::User)),
            CanonicalRole::User
        );
        assert_eq!(
            anthropic_role_to_canonical(canonical_role_to_anthropic(CanonicalRole::Assistant)),
            CanonicalRole::Assistant
        );
    }

    #[test]
    fn test_openai_developer_role() {
        assert_eq!(openai_role_to_canonical("developer"), CanonicalRole::System);
    }

    // --- Stop reason bijectivity tests ---

    #[test]
    fn test_openai_stop_roundtrip() {
        for reason in [
            CanonicalStopReason::EndOfTurn,
            CanonicalStopReason::ToolCalls,
            CanonicalStopReason::MaxTokens,
            CanonicalStopReason::ContentFilter,
        ] {
            let wire = canonical_stop_to_openai(reason);
            let back = openai_stop_to_canonical(wire);
            assert_eq!(reason, back, "OpenAI stop roundtrip failed for {reason:?}");
        }
    }

    #[test]
    fn test_anthropic_stop_roundtrip() {
        // ContentFilter maps to end_turn in Anthropic, so it won't roundtrip.
        for reason in [
            CanonicalStopReason::EndOfTurn,
            CanonicalStopReason::ToolCalls,
            CanonicalStopReason::MaxTokens,
        ] {
            let wire = canonical_stop_to_anthropic(reason);
            let back = anthropic_stop_to_canonical(wire);
            assert_eq!(
                reason, back,
                "Anthropic stop roundtrip failed for {reason:?}"
            );
        }
    }

    #[test]
    fn test_gemini_stop_roundtrip_end_and_max() {
        assert_eq!(
            gemini_stop_to_canonical(canonical_stop_to_gemini(CanonicalStopReason::EndOfTurn)),
            CanonicalStopReason::EndOfTurn
        );
        assert_eq!(
            gemini_stop_to_canonical(canonical_stop_to_gemini(CanonicalStopReason::MaxTokens)),
            CanonicalStopReason::MaxTokens
        );
        assert_eq!(
            gemini_stop_to_canonical(canonical_stop_to_gemini(CanonicalStopReason::ContentFilter)),
            CanonicalStopReason::ContentFilter
        );
    }

    // --- Usage roundtrip tests ---

    #[test]
    fn test_openai_usage_roundtrip() {
        let usage = CanonicalUsage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            total_tokens: Some(150),
        };
        let json = canonical_usage_to_openai(&usage);
        let back = openai_usage_to_canonical(&json);
        assert_eq!(back.input_tokens, Some(100));
        assert_eq!(back.output_tokens, Some(50));
        assert_eq!(back.total_tokens, Some(150));
    }

    #[test]
    fn test_anthropic_usage_roundtrip() {
        let usage = CanonicalUsage {
            input_tokens: Some(200),
            output_tokens: Some(80),
            total_tokens: Some(280),
        };
        let json = canonical_usage_to_anthropic(&usage);
        let back = anthropic_usage_to_canonical(&json);
        assert_eq!(back.input_tokens, Some(200));
        assert_eq!(back.output_tokens, Some(80));
        assert_eq!(back.total_tokens, Some(280));
    }

    #[test]
    fn test_gemini_usage_roundtrip() {
        let usage = CanonicalUsage {
            input_tokens: Some(300),
            output_tokens: Some(120),
            total_tokens: Some(420),
        };
        let json = canonical_usage_to_gemini(&usage);
        let back = gemini_usage_to_canonical(&json);
        assert_eq!(back.input_tokens, Some(300));
        assert_eq!(back.output_tokens, Some(120));
        assert_eq!(back.total_tokens, Some(420));
    }

    #[test]
    fn test_default_usage() {
        let usage = CanonicalUsage::default();
        assert!(usage.input_tokens.is_none());
        assert!(usage.output_tokens.is_none());
        assert!(usage.total_tokens.is_none());
    }
}
