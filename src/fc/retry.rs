use crate::config::FeaturesConfig;
use crate::protocol::canonical::{CanonicalMessage, CanonicalPart, CanonicalRole};

// ---------------------------------------------------------------------------
// Default retry prompt template
// ---------------------------------------------------------------------------

const DEFAULT_RETRY_TEMPLATE: &str = "\
Your previous response attempted to make a function call but the format was invalid or could not be parsed.

**Your original response:**
```
{original_response}
```

**Error details:**
{error_details}

**Instructions:**
Please retry and output the function call in the correct XML format. Remember:
1. Start with the trigger signal on its own line
2. Immediately follow with the <function_calls> XML block
3. Use <args_json> with valid JSON for parameters
4. Do not add any text after </function_calls>

Please provide the corrected function call now. DO NOT OUTPUT ANYTHING ELSE.";

// ---------------------------------------------------------------------------
// Retry decision
// ---------------------------------------------------------------------------

/// Decide whether a retry should be attempted.
///
/// All four conditions must hold:
/// - FC error retry is enabled in the config
/// - `attempt` has not reached `max_attempts`
/// - The trigger signal was found in the response (`has_trigger`)
/// - Parsing or validation failed (`parse_error`)
#[must_use]
pub fn should_retry(
    features: &FeaturesConfig,
    attempt: u32,
    has_trigger: bool,
    parse_error: bool,
) -> bool {
    features.enable_fc_error_retry
        && attempt < features.fc_error_retry_max_attempts
        && has_trigger
        && parse_error
}

// ---------------------------------------------------------------------------
// Build retry prompt
// ---------------------------------------------------------------------------

/// Build the text prompt sent back to the model asking it to fix its output.
///
/// If `custom_template` is provided, `{error_details}` and `{original_response}`
/// placeholders are interpolated. Otherwise the built-in default template is used.
#[must_use]
pub fn build_retry_prompt(
    error_details: &str,
    original_response: &str,
    custom_template: Option<&str>,
) -> String {
    let template = custom_template.unwrap_or(DEFAULT_RETRY_TEMPLATE);
    template
        .replace("{error_details}", error_details)
        .replace("{original_response}", original_response)
}

// ---------------------------------------------------------------------------
// Build retry messages
// ---------------------------------------------------------------------------

/// Append retry context to the original conversation.
///
/// Returns a new message list that contains:
/// 1. All `original_messages` (cloned)
/// 2. An assistant message with the failed response as plain text
/// 3. A user message with the retry prompt as plain text
#[must_use]
pub fn build_retry_messages(
    original_messages: &[CanonicalMessage],
    assistant_response: &str,
    retry_prompt: &str,
) -> Vec<CanonicalMessage> {
    let mut messages: Vec<CanonicalMessage> = original_messages.to_vec();

    // Assistant message echoing back the failed response
    messages.push(CanonicalMessage {
        role: CanonicalRole::Assistant,
        parts: vec![CanonicalPart::Text(assistant_response.to_owned())].into(),
        name: None,
        tool_call_id: None,
        provider_extensions: None,
    });

    // User message with the retry prompt
    messages.push(CanonicalMessage {
        role: CanonicalRole::User,
        parts: vec![CanonicalPart::Text(retry_prompt.to_owned())].into(),
        name: None,
        tool_call_id: None,
        provider_extensions: None,
    });

    messages
}

// ---------------------------------------------------------------------------
// RetryContext — stateful helper for the retry loop
// ---------------------------------------------------------------------------

/// Tracks retry state across attempts.
///
/// Created once from the feature configuration and mutated in place as each
/// attempt is made. The actual upstream call is **not** performed here — that
/// responsibility belongs to the FC middleware. `RetryContext` only answers
/// "should I keep going?" and keeps count.
pub struct RetryContext {
    pub max_attempts: u32,
    pub current_attempt: u32,
    pub enable_retry: bool,
    pub retry_template: Option<String>,
}

impl RetryContext {
    /// Construct from the current feature flags.
    #[must_use]
    pub fn new(features: &FeaturesConfig) -> Self {
        Self {
            max_attempts: features.fc_error_retry_max_attempts,
            current_attempt: 0,
            enable_retry: features.enable_fc_error_retry,
            retry_template: features.fc_error_retry_prompt_template.clone(),
        }
    }

    /// Returns `true` if another retry attempt should be made.
    ///
    /// Mirrors the same four-condition check from [`should_retry`] but uses
    /// the internal counter.
    #[must_use]
    pub fn should_continue(&self, has_trigger: bool, parse_failed: bool) -> bool {
        self.enable_retry && self.current_attempt < self.max_attempts && has_trigger && parse_failed
    }

    /// Advance the attempt counter by one.
    pub fn increment(&mut self) {
        self.current_attempt += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_features() -> FeaturesConfig {
        FeaturesConfig {
            enable_fc_error_retry: true,
            fc_error_retry_max_attempts: 3,
            ..FeaturesConfig::default()
        }
    }

    // -- should_retry ---------------------------------------------------------

    #[test]
    fn test_should_retry_all_conditions_met() {
        let f = default_features();
        assert!(should_retry(&f, 0, true, true));
        assert!(should_retry(&f, 2, true, true));
    }

    #[test]
    fn test_should_retry_disabled() {
        let mut f = default_features();
        f.enable_fc_error_retry = false;
        assert!(!should_retry(&f, 0, true, true));
    }

    #[test]
    fn test_should_retry_at_max_attempts() {
        let f = default_features();
        assert!(!should_retry(&f, 3, true, true));
        assert!(!should_retry(&f, 10, true, true));
    }

    #[test]
    fn test_should_retry_no_trigger() {
        let f = default_features();
        assert!(!should_retry(&f, 0, false, true));
    }

    #[test]
    fn test_should_retry_no_parse_error() {
        let f = default_features();
        assert!(!should_retry(&f, 0, true, false));
    }

    // -- build_retry_prompt ---------------------------------------------------

    #[test]
    fn test_build_retry_prompt_default_template() {
        let prompt = build_retry_prompt("bad xml", "response text", None);
        assert!(prompt.contains("bad xml"));
        assert!(prompt.contains("response text"));
        assert!(prompt.contains("DO NOT OUTPUT ANYTHING ELSE"));
    }

    #[test]
    fn test_build_retry_prompt_custom_template() {
        let tpl = "Error: {error_details} | Response: {original_response}";
        let prompt = build_retry_prompt("oops", "hello", Some(tpl));
        assert_eq!(prompt, "Error: oops | Response: hello");
    }

    // -- build_retry_messages -------------------------------------------------

    #[test]
    fn test_build_retry_messages_appends_two() {
        let original = vec![CanonicalMessage {
            role: CanonicalRole::User,
            parts: vec![CanonicalPart::Text("hi".into())].into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        }];
        let msgs = build_retry_messages(&original, "bad response", "please fix");
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1].role, CanonicalRole::Assistant);
        assert_eq!(msgs[2].role, CanonicalRole::User);

        // Verify text content
        match &msgs[1].parts[0] {
            CanonicalPart::Text(t) => assert_eq!(t, "bad response"),
            _ => panic!("expected Text part"),
        }
        match &msgs[2].parts[0] {
            CanonicalPart::Text(t) => assert_eq!(t, "please fix"),
            _ => panic!("expected Text part"),
        }
    }

    // -- RetryContext ----------------------------------------------------------

    #[test]
    fn test_retry_context_new() {
        let f = default_features();
        let ctx = RetryContext::new(&f);
        assert_eq!(ctx.max_attempts, 3);
        assert_eq!(ctx.current_attempt, 0);
        assert!(ctx.enable_retry);
        assert!(ctx.retry_template.is_none());
    }

    #[test]
    fn test_retry_context_should_continue() {
        let f = default_features();
        let mut ctx = RetryContext::new(&f);
        assert!(ctx.should_continue(true, true));

        ctx.increment();
        assert!(ctx.should_continue(true, true)); // attempt 1 < 3

        ctx.increment();
        assert!(ctx.should_continue(true, true)); // attempt 2 < 3

        ctx.increment();
        assert!(!ctx.should_continue(true, true)); // attempt 3 == 3
    }

    #[test]
    fn test_retry_context_disabled() {
        let mut f = default_features();
        f.enable_fc_error_retry = false;
        let ctx = RetryContext::new(&f);
        assert!(!ctx.should_continue(true, true));
    }

    #[test]
    fn test_retry_context_with_custom_template() {
        let mut f = default_features();
        f.fc_error_retry_prompt_template = Some("custom {error_details}".into());
        let ctx = RetryContext::new(&f);
        assert_eq!(
            ctx.retry_template.as_deref(),
            Some("custom {error_details}")
        );
    }
}
