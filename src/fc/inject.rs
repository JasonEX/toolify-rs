use crate::config::FeaturesConfig;
use crate::error::CanonicalError;
use crate::protocol::canonical::{
    CanonicalRequest, CanonicalToolChoice, CanonicalToolSpec, IngressApi,
};
use std::sync::Arc;

use super::preprocess::preprocess_messages_owned;
use super::prompt;

/// Prepare a canonical request for FC injection.
///
/// - Generates the FC system prompt and prepends/appends to existing `system_prompt`.
/// - Preprocesses messages (Tool->User, ToolCall->XML).
/// - **Removes** tools and sets `tool_choice = none` for FC-active requests (S2-I5).
/// - Keeps request unchanged when caller explicitly sets `tool_choice = none`.
///
/// # Errors
///
/// Returns [`CanonicalError`] when prompt generation fails.
pub fn apply_fc_inject(
    canonical: &mut CanonicalRequest,
    features: &FeaturesConfig,
) -> Result<(), CanonicalError> {
    let _ = apply_fc_inject_take_tools(canonical, features)?;
    Ok(())
}

/// Same as [`apply_fc_inject`], but returns the original tools instead of cloning.
///
/// This allows hot paths to reuse tool specs for post-processing without an
/// extra `tools.clone()` allocation.
///
/// # Errors
///
/// Returns [`CanonicalError`] when prompt generation fails.
pub fn apply_fc_inject_take_tools(
    canonical: &mut CanonicalRequest,
    features: &FeaturesConfig,
) -> Result<Arc<[CanonicalToolSpec]>, CanonicalError> {
    if request_prefers_structured_output(canonical) {
        return Ok(Arc::<[CanonicalToolSpec]>::from([]));
    }

    if matches!(canonical.tool_choice, CanonicalToolChoice::None) {
        return Ok(Arc::<[CanonicalToolSpec]>::from([]));
    }

    let saved_tools = std::mem::take(&mut canonical.tools);

    let fc_prompt = prompt::generate_fc_prompt(
        saved_tools.as_ref(),
        &canonical.tool_choice,
        features.prompt_template.as_deref(),
    )?;

    canonical.system_prompt = Some(match &canonical.system_prompt {
        Some(existing) => format!("{existing}\n{fc_prompt}"),
        None => fc_prompt,
    });

    let messages = std::mem::take(&mut canonical.messages);
    canonical.messages = preprocess_messages_owned(messages, features.convert_developer_to_system);

    canonical.tool_choice = CanonicalToolChoice::None;

    Ok(saved_tools)
}

#[inline]
fn request_prefers_structured_output(canonical: &CanonicalRequest) -> bool {
    match canonical.ingress_api {
        IngressApi::OpenAiChat => canonical
            .provider_extensions_ref()
            .get("response_format")
            .is_some_and(response_format_is_json_mode),
        IngressApi::OpenAiResponses => {
            let extensions = canonical.provider_extensions_ref();
            extensions
                .get("response_format")
                .is_some_and(response_format_is_json_mode)
                || extensions
                    .get("text")
                    .is_some_and(responses_text_format_is_json_mode)
        }
        IngressApi::Anthropic | IngressApi::Gemini => false,
    }
}

#[inline]
fn response_format_is_json_mode(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(mode) => {
            mode.eq_ignore_ascii_case("json_object") || mode.eq_ignore_ascii_case("json_schema")
        }
        serde_json::Value::Object(obj) => obj
            .get("type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|mode| {
                mode.eq_ignore_ascii_case("json_object") || mode.eq_ignore_ascii_case("json_schema")
            }),
        _ => false,
    }
}

#[inline]
fn responses_text_format_is_json_mode(value: &serde_json::Value) -> bool {
    value
        .as_object()
        .and_then(|obj| obj.get("format"))
        .is_some_and(response_format_is_json_mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::{
        CanonicalMessage, CanonicalPart, CanonicalRole, CanonicalToolFunction, GenerationParams,
        IngressApi,
    };
    use serde_json::json;

    fn make_tool(name: &str, desc: &str, params: serde_json::Value) -> CanonicalToolSpec {
        CanonicalToolSpec {
            function: CanonicalToolFunction {
                name: name.to_string(),
                description: if desc.is_empty() {
                    None
                } else {
                    Some(desc.to_string())
                },
                parameters: params,
            },
        }
    }

    fn make_message(role: CanonicalRole, text: &str) -> CanonicalMessage {
        CanonicalMessage {
            role,
            parts: vec![CanonicalPart::Text(text.to_string())].into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        }
    }

    #[test]
    fn test_apply_fc_inject_sets_system_prompt() {
        let tools = vec![make_tool(
            "get_weather",
            "Get weather",
            json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        )];

        let mut request = CanonicalRequest {
            request_id: uuid::Uuid::from_u128(1),
            ingress_api: IngressApi::OpenAiChat,
            model: "test-model".into(),
            stream: false,
            system_prompt: Some("You are helpful.".into()),
            messages: vec![make_message(CanonicalRole::User, "hi")],
            tools: tools.into(),
            tool_choice: CanonicalToolChoice::Auto,
            generation: GenerationParams::default(),
            provider_extensions: None,
        };

        let features = FeaturesConfig::default();
        apply_fc_inject(&mut request, &features).unwrap();

        let sp = request.system_prompt.as_ref().unwrap();
        assert!(sp.starts_with("You are helpful.\n"));
        assert!(sp.contains(prompt::get_trigger_signal()));

        assert!(request.tools.is_empty());
        assert_eq!(request.tool_choice, CanonicalToolChoice::None);
    }

    #[test]
    fn test_apply_fc_inject_no_existing_system_prompt() {
        let tools = vec![make_tool(
            "f",
            "desc",
            json!({"type": "object", "properties": {}}),
        )];

        let mut request = CanonicalRequest {
            request_id: uuid::Uuid::from_u128(1),
            ingress_api: IngressApi::OpenAiChat,
            model: "test-model".into(),
            stream: false,
            system_prompt: None,
            messages: vec![make_message(CanonicalRole::User, "hi")],
            tools: tools.into(),
            tool_choice: CanonicalToolChoice::Auto,
            generation: GenerationParams::default(),
            provider_extensions: None,
        };

        let features = FeaturesConfig::default();
        apply_fc_inject(&mut request, &features).unwrap();

        assert!(request.system_prompt.is_some());
        assert!(request.tools.is_empty());
    }

    #[test]
    fn test_apply_fc_inject_skips_when_tool_choice_none() {
        let tools = vec![make_tool(
            "noop",
            "desc",
            json!({"type": "object", "properties": {}}),
        )];
        let original_tools = tools.clone();

        let mut request = CanonicalRequest {
            request_id: uuid::Uuid::from_u128(1),
            ingress_api: IngressApi::OpenAiChat,
            model: "test-model".into(),
            stream: false,
            system_prompt: Some("base".into()),
            messages: vec![make_message(CanonicalRole::User, "hi")],
            tools: tools.into(),
            tool_choice: CanonicalToolChoice::None,
            generation: GenerationParams::default(),
            provider_extensions: None,
        };

        let features = FeaturesConfig::default();
        let saved_tools = apply_fc_inject_take_tools(&mut request, &features).unwrap();

        assert!(saved_tools.is_empty());
        assert_eq!(request.system_prompt.as_deref(), Some("base"));
        assert_eq!(request.tool_choice, CanonicalToolChoice::None);
        assert_eq!(request.tools.as_ref(), original_tools.as_slice());
    }

    #[test]
    fn test_apply_fc_inject_skips_when_response_format_json_mode() {
        let tools = vec![make_tool(
            "noop",
            "desc",
            json!({"type": "object", "properties": {}}),
        )];
        let original_tools = tools.clone();
        let mut provider_extensions = serde_json::Map::new();
        provider_extensions.insert(
            "response_format".to_string(),
            json!({"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}),
        );

        let mut request = CanonicalRequest {
            request_id: uuid::Uuid::from_u128(1),
            ingress_api: IngressApi::OpenAiChat,
            model: "test-model".into(),
            stream: false,
            system_prompt: Some("base".into()),
            messages: vec![make_message(CanonicalRole::User, "hi")],
            tools: tools.into(),
            tool_choice: CanonicalToolChoice::Auto,
            generation: GenerationParams::default(),
            provider_extensions: Some(Box::new(provider_extensions)),
        };

        let features = FeaturesConfig::default();
        let saved_tools = apply_fc_inject_take_tools(&mut request, &features).unwrap();

        assert!(saved_tools.is_empty());
        assert_eq!(request.system_prompt.as_deref(), Some("base"));
        assert_eq!(request.tool_choice, CanonicalToolChoice::Auto);
        assert_eq!(request.tools.as_ref(), original_tools.as_slice());
    }

    #[test]
    fn test_apply_fc_inject_skips_for_responses_text_json_mode() {
        let tools = vec![make_tool(
            "noop",
            "desc",
            json!({"type": "object", "properties": {}}),
        )];
        let original_tools = tools.clone();
        let mut provider_extensions = serde_json::Map::new();
        provider_extensions.insert("text".to_string(), json!({"format":{"type":"json_object"}}));

        let mut request = CanonicalRequest {
            request_id: uuid::Uuid::from_u128(1),
            ingress_api: IngressApi::OpenAiResponses,
            model: "test-model".into(),
            stream: false,
            system_prompt: Some("base".into()),
            messages: vec![make_message(CanonicalRole::User, "hi")],
            tools: tools.into(),
            tool_choice: CanonicalToolChoice::Auto,
            generation: GenerationParams::default(),
            provider_extensions: Some(Box::new(provider_extensions)),
        };

        let features = FeaturesConfig::default();
        let saved_tools = apply_fc_inject_take_tools(&mut request, &features).unwrap();

        assert!(saved_tools.is_empty());
        assert_eq!(request.system_prompt.as_deref(), Some("base"));
        assert_eq!(request.tool_choice, CanonicalToolChoice::Auto);
        assert_eq!(request.tools.as_ref(), original_tools.as_slice());
    }
}
