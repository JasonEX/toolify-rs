use toolify_rs::config::FeaturesConfig;
use toolify_rs::fc::{apply_fc_inject_take_tools, process_fc_response, FcResult};
use toolify_rs::protocol::canonical::{
    CanonicalMessage, CanonicalPart, CanonicalRequest, CanonicalRole, CanonicalToolChoice,
    CanonicalToolFunction, CanonicalToolSpec, GenerationParams, IngressApi,
};

fn sample_request() -> CanonicalRequest {
    CanonicalRequest {
        request_id: uuid::Uuid::from_u128(1),
        ingress_api: IngressApi::OpenAiChat,
        model: "gpt-4o-mini".to_string(),
        stream: false,
        system_prompt: Some("You are helpful.".to_string()),
        messages: vec![CanonicalMessage {
            role: CanonicalRole::User,
            parts: vec![CanonicalPart::Text("Check weather in SF".to_string())].into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        }],
        tools: vec![CanonicalToolSpec {
            function: CanonicalToolFunction {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }),
            },
        }]
        .into(),
        tool_choice: CanonicalToolChoice::Required,
        generation: GenerationParams::default(),
        provider_extensions: None,
    }
}

#[test]
fn test_fc_injection_preprocesses_request() {
    let mut request = sample_request();
    let features = FeaturesConfig::default();

    let saved_tools = apply_fc_inject_take_tools(&mut request, &features).expect("apply inject");
    assert_eq!(saved_tools.len(), 1);
    assert!(request.tools.is_empty());
    assert!(matches!(request.tool_choice, CanonicalToolChoice::None));

    let system_prompt = request.system_prompt.as_deref().expect("system prompt");
    assert!(system_prompt.contains("get_weather"));
    assert!(system_prompt.contains(toolify_rs::fc::prompt::get_trigger_signal()));
}

#[test]
fn test_fc_response_without_trigger_is_passthrough_text() {
    let tools = sample_request().tools;
    let result = process_fc_response("This is a normal text response without tool calls.", &tools)
        .expect("process");
    assert!(matches!(result, FcResult::NoToolCalls));
}
