use serde_json::json;
use toolify_rs::protocol::canonical::{
    CanonicalPart, CanonicalRequest, CanonicalResponse, CanonicalStopReason, CanonicalToolChoice,
    CanonicalUsage, ProviderKind,
};
use toolify_rs::protocol::{anthropic, gemini, openai_chat, openai_responses};
use uuid::Uuid;

fn ingress_openai_chat_request() -> CanonicalRequest {
    let wire: openai_chat::OpenAiChatRequest = serde_json::from_value(json!({
        "model": "ingress-model",
        "messages": [
            {"role":"system","content":"You are helpful."},
            {"role":"user","content":"What's the weather in SF?"},
            {
                "role":"assistant",
                "content": null,
                "tool_calls": [{
                    "id":"call_1",
                    "type":"function",
                    "function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}
                }]
            },
            {"role":"tool","tool_call_id":"call_1","content":"{\"temp\":72}"}
        ],
        "tools": [{
            "type":"function",
            "function":{
                "name":"get_weather",
                "description":"Get weather",
                "parameters":{
                    "type":"object",
                    "properties":{"city":{"type":"string"}},
                    "required":["city"]
                }
            }
        }],
        "tool_choice":{"type":"function","function":{"name":"get_weather"}}
    }))
    .expect("openai chat wire parse");
    openai_chat::decoder::decode_openai_chat_request(&wire, Uuid::from_u128(1))
        .expect("openai decode")
}

fn ingress_openai_responses_request() -> CanonicalRequest {
    let wire: openai_responses::ResponsesRequest = serde_json::from_value(json!({
        "model": "ingress-model",
        "instructions": "You are helpful.",
        "input": [
            {
                "type":"message",
                "role":"user",
                "content":[{"type":"input_text","text":"What's the weather in SF?"}]
            },
            {
                "type":"function_call",
                "id":"fc_1",
                "call_id":"call_1",
                "name":"get_weather",
                "arguments":"{\"city\":\"SF\"}"
            },
            {
                "type":"function_call_output",
                "call_id":"call_1",
                "output":"{\"temp\":72}"
            }
        ],
        "tools": [{
            "type":"function",
            "name":"get_weather",
            "description":"Get weather",
            "parameters":{
                "type":"object",
                "properties":{"city":{"type":"string"}},
                "required":["city"]
            }
        }],
        "tool_choice":{"type":"function","name":"get_weather"},
        "parallel_tool_calls": false
    }))
    .expect("openai responses wire parse");
    openai_responses::decoder::decode_responses_request(&wire, Uuid::from_u128(1))
        .expect("responses decode")
}

fn ingress_anthropic_request() -> CanonicalRequest {
    let wire: anthropic::AnthropicRequest = serde_json::from_value(json!({
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "system": "You are helpful.",
        "messages": [
            {"role":"user","content":[{"type":"text","text":"What's the weather in SF?"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"get_weather","input":{"city":"SF"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"{\"temp\":72}"}]}
        ],
        "tools": [{
            "name":"get_weather",
            "description":"Get weather",
            "input_schema":{
                "type":"object",
                "properties":{"city":{"type":"string"}},
                "required":["city"]
            }
        }],
        "tool_choice":{"type":"tool","name":"get_weather"}
    }))
    .expect("anthropic wire parse");
    anthropic::decoder::decode_anthropic_request(&wire, Uuid::from_u128(1))
        .expect("anthropic decode")
}

fn ingress_gemini_request() -> CanonicalRequest {
    let wire: gemini::GeminiRequest = serde_json::from_value(json!({
        "contents": [
            {"role":"user","parts":[{"text":"What's the weather in SF?"}]},
            {"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"SF"}}}]},
            {"role":"function","parts":[{"functionResponse":{"name":"get_weather","response":{"temp":72}}}]}
        ],
        "tools": [{
            "functionDeclarations": [{
                "name":"get_weather",
                "description":"Get weather",
                "parameters":{
                    "type":"object",
                    "properties":{"city":{"type":"string"}},
                    "required":["city"]
                }
            }]
        }],
        "toolConfig":{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["get_weather"]}},
        "systemInstruction":{"parts":[{"text":"You are helpful."}]}
    }))
    .expect("gemini wire parse");
    gemini::decoder::decode_gemini_request(&wire, "gemini-2.5-pro", Uuid::from_u128(1))
        .expect("gemini decode")
}

fn roundtrip_request_via_provider(
    request: &CanonicalRequest,
    provider: ProviderKind,
) -> CanonicalRequest {
    match provider {
        ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => {
            let wire = openai_chat::encoder::encode_openai_chat_request(request).expect("encode");
            openai_chat::decoder::decode_openai_chat_request(&wire, Uuid::from_u128(1))
                .expect("decode")
        }
        ProviderKind::OpenAiResponses => {
            let wire =
                openai_responses::encoder::encode_responses_request(request).expect("encode");
            openai_responses::decoder::decode_responses_request(&wire, Uuid::from_u128(1))
                .expect("decode")
        }
        ProviderKind::Anthropic => {
            let wire = anthropic::encoder::encode_anthropic_request(request).expect("encode");
            anthropic::decoder::decode_anthropic_request(&wire, Uuid::from_u128(1)).expect("decode")
        }
        ProviderKind::Gemini => {
            let wire = gemini::encoder::encode_gemini_request(request).expect("encode");
            gemini::decoder::decode_gemini_request(&wire, &request.model, Uuid::from_u128(1))
                .expect("decode")
        }
    }
}

fn assert_request_tool_semantics(request: &CanonicalRequest) {
    assert!(
        request.messages.iter().any(|msg| msg
            .parts
            .iter()
            .any(|part| matches!(part, CanonicalPart::Text(text) if !text.is_empty()))),
        "expected at least one text part"
    );
    assert!(
        request
            .messages
            .iter()
            .any(|msg| msg.parts.iter().any(|part| matches!(
                part,
                CanonicalPart::ToolCall { name, .. } if name == "get_weather"
            ))),
        "expected tool call for get_weather"
    );
    assert!(
        request.messages.iter().any(|msg| msg
            .parts
            .iter()
            .any(|part| matches!(part, CanonicalPart::ToolResult { .. }))),
        "expected at least one tool result"
    );
    assert!(
        request
            .tools
            .iter()
            .any(|tool| tool.function.name == "get_weather"),
        "expected get_weather tool declaration"
    );
    assert!(
        !matches!(request.tool_choice, CanonicalToolChoice::None),
        "tool_choice should keep tool-enabled semantics"
    );
}

fn base_canonical_response() -> CanonicalResponse {
    CanonicalResponse {
        id: "resp_matrix".to_string(),
        model: "test-model".to_string(),
        content: vec![
            CanonicalPart::Text("Let me check.".to_string()),
            CanonicalPart::ToolCall {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                arguments: serde_json::value::RawValue::from_string(
                    "{\"city\":\"SF\"}".to_string(),
                )
                .expect("raw args"),
            },
        ],
        stop_reason: CanonicalStopReason::ToolCalls,
        usage: CanonicalUsage::default(),
        provider_extensions: serde_json::Map::new(),
    }
}

fn assert_response_tool_semantics(response: &CanonicalResponse) {
    assert!(
        response.content.iter().any(|part| {
            matches!(
                part,
                CanonicalPart::ToolCall { name, .. } if name == "get_weather"
            )
        }),
        "expected tool call part in response"
    );
    assert_eq!(response.stop_reason, CanonicalStopReason::ToolCalls);
}

#[test]
fn test_request_matrix_4x5_tool_flow() {
    let ingress_cases = vec![
        ingress_openai_chat_request(),
        ingress_openai_responses_request(),
        ingress_anthropic_request(),
        ingress_gemini_request(),
    ];
    let providers = vec![
        ProviderKind::OpenAi,
        ProviderKind::OpenAiResponses,
        ProviderKind::Anthropic,
        ProviderKind::Gemini,
        ProviderKind::GeminiOpenAi,
    ];

    for ingress_request in ingress_cases {
        assert_request_tool_semantics(&ingress_request);
        for provider in &providers {
            let roundtripped = roundtrip_request_via_provider(&ingress_request, *provider);
            assert_request_tool_semantics(&roundtripped);
        }
    }
}

#[test]
fn test_response_matrix_across_ingress_formats() {
    let canonical = base_canonical_response();

    let chat_wire =
        openai_chat::response_encoder::encode_openai_chat_response(&canonical, "gpt-4o")
            .expect("chat encode");
    let chat_back = openai_chat::response_decoder::decode_openai_chat_response(&chat_wire)
        .expect("chat decode");
    assert_response_tool_semantics(&chat_back);

    let resp_wire =
        openai_responses::response_encoder::encode_responses_output(&canonical, "gpt-4o")
            .expect("responses encode");
    let resp_back = openai_responses::response_decoder::decode_responses_output(&resp_wire)
        .expect("responses decode");
    assert_response_tool_semantics(&resp_back);

    let anthropic_wire =
        anthropic::response_encoder::encode_anthropic_response(&canonical, "claude-sonnet-4-5")
            .expect("anthropic encode");
    let anthropic_back = anthropic::response_decoder::decode_anthropic_response(&anthropic_wire)
        .expect("anthropic decode");
    assert_response_tool_semantics(&anthropic_back);

    let gemini_wire =
        gemini::response_encoder::encode_gemini_response(&canonical).expect("gemini encode");
    let gemini_back =
        gemini::response_decoder::decode_gemini_response(&gemini_wire, "gemini-2.5-pro")
            .expect("gemini decode");
    assert_response_tool_semantics(&gemini_back);
}
