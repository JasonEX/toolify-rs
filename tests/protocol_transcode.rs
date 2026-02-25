use serde_json::json;
use toolify_rs::protocol::canonical::{
    CanonicalPart, CanonicalRequest, CanonicalToolChoice, ProviderKind,
};
use toolify_rs::protocol::{anthropic, gemini, openai_chat, openai_responses};
use uuid::Uuid;

fn ingress_openai_chat_request() -> CanonicalRequest {
    let wire: openai_chat::OpenAiChatRequest = serde_json::from_value(json!({
        "model": "ingress-model",
        "messages": [
            { "role":"user", "content":"What is the weather in SF?" },
            {
                "role":"assistant",
                "content": null,
                "tool_calls": [{
                    "id":"call_1",
                    "type":"function",
                    "function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}
                }]
            }
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
    .expect("wire parse");

    openai_chat::decoder::decode_openai_chat_request(&wire, Uuid::from_u128(1)).expect("decode")
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

fn assert_tool_semantics(request: &CanonicalRequest) {
    assert!(
        request.messages.iter().any(|message| {
            message
                .parts
                .iter()
                .any(|part| matches!(part, CanonicalPart::Text(text) if !text.is_empty()))
        }),
        "expected at least one text part"
    );
    assert!(
        request.messages.iter().any(|message| {
            message.parts.iter().any(
                |part| matches!(part, CanonicalPart::ToolCall { name, .. } if name == "get_weather"),
            )
        }),
        "expected get_weather tool call"
    );
    assert!(
        request
            .tools
            .iter()
            .any(|tool| tool.function.name == "get_weather"),
        "expected tool declaration"
    );
    assert!(
        !matches!(request.tool_choice, CanonicalToolChoice::None),
        "tool_choice should keep tool-enabled semantics"
    );
}

#[test]
fn test_protocol_transcode_request_matrix() {
    let request = ingress_openai_chat_request();
    assert_tool_semantics(&request);

    let providers = [
        ProviderKind::OpenAi,
        ProviderKind::OpenAiResponses,
        ProviderKind::Anthropic,
        ProviderKind::Gemini,
        ProviderKind::GeminiOpenAi,
    ];

    for provider in providers {
        let roundtripped = roundtrip_request_via_provider(&request, provider);
        assert_tool_semantics(&roundtripped);
    }
}
