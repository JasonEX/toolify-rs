use crate::config::{FcMode, FeaturesConfig, UpstreamServiceConfig};
use crate::error::CanonicalError;

/// The action the FC middleware should take for a given request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FcAction {
    /// Use FC injection (prompt-based tool calling).
    Inject,
    /// Pass tools natively to the upstream.
    Native,
    /// No tools in request — skip FC entirely.
    Skip,
}

/// Determine the FC action for a request based on upstream config and whether
/// the request carries tools.
///
/// - If no tools in request -> Skip
/// - If upstream `fc_mode` is Inject -> Inject
/// - If upstream `fc_mode` is Native -> Native
/// - If upstream `fc_mode` is Auto -> Native (caller handles fallback to Inject on failure)
#[must_use]
pub fn get_fc_mode(upstream: &UpstreamServiceConfig, has_tools: bool) -> FcAction {
    if !has_tools {
        return FcAction::Skip;
    }
    match upstream.fc_mode {
        FcMode::Inject => FcAction::Inject,
        FcMode::Native | FcMode::Auto => FcAction::Native,
    }
}

/// Determine FC action with global feature gate applied.
///
/// When `enable_function_calling` is false, injected FC logic is disabled and
/// requests with tools are treated as native passthrough.
#[must_use]
pub fn decide_fc_action(
    features: &FeaturesConfig,
    upstream: &UpstreamServiceConfig,
    has_tools: bool,
) -> FcAction {
    if !has_tools {
        return FcAction::Skip;
    }
    if !features.enable_function_calling {
        return FcAction::Native;
    }
    get_fc_mode(upstream, true)
}

/// Whether `fc_mode=auto` can fallback from native tool passing to inject mode.
#[must_use]
pub fn allow_auto_inject_fallback(
    features: &FeaturesConfig,
    upstream: &UpstreamServiceConfig,
    has_tools: bool,
) -> bool {
    has_tools && features.enable_function_calling && matches!(upstream.fc_mode, FcMode::Auto)
}

/// Detect if an upstream error likely means native tool calling is unsupported.
///
/// This is used for `fc_mode=auto` one-shot fallback:
/// 1) try native tool call
/// 2) on capability error, retry once with FC inject mode
#[must_use]
pub fn should_auto_fallback_to_inject(err: &CanonicalError) -> bool {
    let crate::error::CanonicalError::Upstream { status, message } = err else {
        return false;
    };
    if !matches!(*status, 400 | 404 | 422 | 501) {
        return false;
    }

    let msg = message.to_ascii_lowercase();
    let mentions_tools = [
        "tool",
        "tools",
        "tool_choice",
        "function call",
        "function_call",
        "function calling",
    ]
    .iter()
    .any(|kw| msg.contains(kw));
    if !mentions_tools {
        return false;
    }

    [
        "unsupported",
        "does not support",
        "doesn't support",
        "not support",
        "not implemented",
        "unrecognized request argument",
        "unknown field",
        "unknown parameter",
        "invalid parameter",
        "not available",
    ]
    .iter()
    .any(|kw| msg.contains(kw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_upstream(fc_mode: FcMode) -> UpstreamServiceConfig {
        UpstreamServiceConfig {
            name: "test".into(),
            provider: "openai".into(),
            base_url: "http://localhost".into(),
            api_key: "key".into(),
            models: vec![],
            description: String::new(),
            is_default: false,
            fc_mode,
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        }
    }

    #[test]
    fn test_fc_mode_skip_no_tools() {
        let upstream = make_upstream(FcMode::Inject);
        assert_eq!(get_fc_mode(&upstream, false), FcAction::Skip);
    }

    #[test]
    fn test_fc_mode_inject() {
        let upstream = make_upstream(FcMode::Inject);
        assert_eq!(get_fc_mode(&upstream, true), FcAction::Inject);
    }

    #[test]
    fn test_fc_mode_native() {
        let upstream = make_upstream(FcMode::Native);
        assert_eq!(get_fc_mode(&upstream, true), FcAction::Native);
    }

    #[test]
    fn test_fc_mode_auto_returns_native() {
        let upstream = make_upstream(FcMode::Auto);
        assert_eq!(get_fc_mode(&upstream, true), FcAction::Native);
    }

    #[test]
    fn test_decide_fc_action_disabled_feature_forces_native() {
        let upstream = make_upstream(FcMode::Inject);
        let features = FeaturesConfig {
            enable_function_calling: false,
            ..FeaturesConfig::default()
        };
        assert_eq!(
            decide_fc_action(&features, &upstream, true),
            FcAction::Native
        );
    }

    #[test]
    fn test_allow_auto_inject_fallback() {
        let upstream = make_upstream(FcMode::Auto);
        assert!(allow_auto_inject_fallback(
            &FeaturesConfig::default(),
            &upstream,
            true
        ));
    }

    #[test]
    fn test_should_auto_fallback_to_inject_positive() {
        let err = CanonicalError::Upstream {
            status: 400,
            message: "This model does not support tools".to_string(),
        };
        assert!(should_auto_fallback_to_inject(&err));
    }

    #[test]
    fn test_should_auto_fallback_to_inject_negative_status() {
        let err = CanonicalError::Upstream {
            status: 500,
            message: "This model does not support tools".to_string(),
        };
        assert!(!should_auto_fallback_to_inject(&err));
    }

    #[test]
    fn test_should_auto_fallback_to_inject_negative_message() {
        let err = CanonicalError::Upstream {
            status: 400,
            message: "rate limit exceeded".to_string(),
        };
        assert!(!should_auto_fallback_to_inject(&err));
    }
}
