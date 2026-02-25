use criterion::{black_box, criterion_group, criterion_main, Criterion};
use toolify_rs::protocol::canonical::{
    CanonicalMessage, CanonicalPart, CanonicalRequest, CanonicalRole, CanonicalToolChoice,
    CanonicalToolFunction, CanonicalToolSpec, GenerationParams, IngressApi, ProviderKind,
};
use toolify_rs::protocol::{anthropic, gemini, openai_chat, openai_responses};
use toolify_rs::stream::{SseEvent, StreamTranscoder};
use uuid::Uuid;

fn sample_request() -> CanonicalRequest {
    CanonicalRequest {
        request_id: Uuid::from_u128(1),
        ingress_api: IngressApi::OpenAiChat,
        model: "gpt-4o-mini".to_string(),
        stream: false,
        system_prompt: Some("You are a helpful assistant".to_string()),
        messages: vec![
            CanonicalMessage {
                role: CanonicalRole::User,
                parts: vec![CanonicalPart::Text(
                    "What is the weather in SF?".to_string(),
                )]
                .into(),
                name: None,
                tool_call_id: None,
                provider_extensions: None,
            },
            CanonicalMessage {
                role: CanonicalRole::Assistant,
                parts: vec![CanonicalPart::ToolCall {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: serde_json::value::RawValue::from_string(
                        "{\"city\":\"SF\"}".to_string(),
                    )
                    .expect("raw args"),
                }]
                .into(),
                name: None,
                tool_call_id: None,
                provider_extensions: None,
            },
        ],
        tools: vec![CanonicalToolSpec {
            function: CanonicalToolFunction {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"]
                }),
            },
        }]
        .into(),
        tool_choice: CanonicalToolChoice::Auto,
        generation: GenerationParams::default(),
        provider_extensions: None,
    }
}

fn sample_large_request(message_count: usize, total_bytes: usize) -> CanonicalRequest {
    let per_message = (total_bytes / message_count.max(1)).max(1);
    let messages = (0..message_count)
        .map(|idx| CanonicalMessage {
            role: if idx % 2 == 0 {
                CanonicalRole::User
            } else {
                CanonicalRole::Assistant
            },
            parts: vec![CanonicalPart::Text("x".repeat(per_message))].into(),
            name: None,
            tool_call_id: None,
            provider_extensions: None,
        })
        .collect::<Vec<_>>();

    CanonicalRequest {
        request_id: Uuid::from_u128(2),
        ingress_api: IngressApi::OpenAiChat,
        model: "gpt-4o-mini".to_string(),
        stream: false,
        system_prompt: Some("You are a helpful assistant".to_string()),
        messages,
        tools: Vec::<CanonicalToolSpec>::new().into(),
        tool_choice: CanonicalToolChoice::None,
        generation: GenerationParams::default(),
        provider_extensions: None,
    }
}

fn bench_transcode(c: &mut Criterion) {
    let canonical = sample_request();
    let large = sample_large_request(50, 100_000);

    c.bench_function("transcode_openai_chat", |b| {
        b.iter(|| {
            let wire =
                openai_chat::encoder::encode_openai_chat_request(black_box(&canonical)).unwrap();
            let decoded =
                openai_chat::decoder::decode_openai_chat_request(&wire, Uuid::from_u128(1))
                    .unwrap();
            black_box(decoded);
        });
    });

    c.bench_function("transcode_openai_responses", |b| {
        b.iter(|| {
            let wire =
                openai_responses::encoder::encode_responses_request(black_box(&canonical)).unwrap();
            let decoded =
                openai_responses::decoder::decode_responses_request(&wire, Uuid::from_u128(1))
                    .unwrap();
            black_box(decoded);
        });
    });

    c.bench_function("transcode_anthropic", |b| {
        b.iter(|| {
            let wire = anthropic::encoder::encode_anthropic_request(black_box(&canonical)).unwrap();
            let decoded =
                anthropic::decoder::decode_anthropic_request(&wire, Uuid::from_u128(1)).unwrap();
            black_box(decoded);
        });
    });

    c.bench_function("transcode_gemini", |b| {
        b.iter(|| {
            let wire = gemini::encoder::encode_gemini_request(black_box(&canonical)).unwrap();
            let decoded =
                gemini::decoder::decode_gemini_request(&wire, &canonical.model, Uuid::from_u128(1))
                    .unwrap();
            black_box(decoded);
        });
    });

    c.bench_function("transcode_openai_chat_large_50msg_100k", |b| {
        b.iter(|| {
            let wire = openai_chat::encoder::encode_openai_chat_request(black_box(&large)).unwrap();
            let decoded =
                openai_chat::decoder::decode_openai_chat_request(&wire, Uuid::from_u128(2))
                    .unwrap();
            black_box(decoded);
        });
    });

    c.bench_function("transcode_openai_responses_large_50msg_100k", |b| {
        b.iter(|| {
            let wire =
                openai_responses::encoder::encode_responses_request(black_box(&large)).unwrap();
            let decoded =
                openai_responses::decoder::decode_responses_request(&wire, Uuid::from_u128(2))
                    .unwrap();
            black_box(decoded);
        });
    });
}

fn sample_stream_text_delta_frame(provider: ProviderKind) -> SseEvent {
    match provider {
        ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => SseEvent {
            event: None,
            data: serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "bench"},
                    "finish_reason": null
                }]
            })
            .to_string(),
            id: None,
            retry: None,
        },
        ProviderKind::OpenAiResponses => SseEvent {
            event: Some("response.output_text.delta".into()),
            data: serde_json::json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "bench"
            })
            .to_string(),
            id: None,
            retry: None,
        },
        ProviderKind::Anthropic => SseEvent {
            event: Some("content_block_delta".into()),
            data: serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "bench"}
            })
            .to_string(),
            id: None,
            retry: None,
        },
        ProviderKind::Gemini => SseEvent {
            event: None,
            data: serde_json::json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "bench"}]
                    },
                    "index": 0
                }]
            })
            .to_string(),
            id: None,
            retry: None,
        },
    }
}

#[derive(Clone, Copy)]
enum StreamScenario {
    TextDelta,
    ToolCall,
    Usage,
    Done,
    Reasoning,
    Error,
}

impl StreamScenario {
    const fn name(self) -> &'static str {
        match self {
            Self::TextDelta => "text_delta",
            Self::ToolCall => "tool_call",
            Self::Usage => "usage",
            Self::Done => "done",
            Self::Reasoning => "reasoning",
            Self::Error => "error",
        }
    }
}

fn sample_stream_tool_call_frame(provider: ProviderKind) -> SseEvent {
    match provider {
        ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => SseEvent {
            event: None,
            data: serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}
                        }]
                    },
                    "finish_reason": null
                }]
            })
            .to_string(),
            id: None,
            retry: None,
        },
        ProviderKind::OpenAiResponses => SseEvent {
            event: Some("response.output_item.added".into()),
            data: serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_0",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{\"q\":\"x\"}"
                }
            })
            .to_string(),
            id: None,
            retry: None,
        },
        ProviderKind::Anthropic => SseEvent {
            event: Some("content_block_start".into()),
            data: serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "lookup",
                    "input": {"q": "x"}
                }
            })
            .to_string(),
            id: None,
            retry: None,
        },
        ProviderKind::Gemini => SseEvent {
            event: None,
            data: serde_json::json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{
                            "functionCall": {
                                "name": "lookup",
                                "args": {"q": "x"}
                            }
                        }]
                    },
                    "index": 0
                }]
            })
            .to_string(),
            id: None,
            retry: None,
        },
    }
}

fn sample_stream_usage_frame(provider: ProviderKind) -> SseEvent {
    match provider {
        ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => SseEvent {
            event: None,
            data: serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": "m1",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                }
            })
            .to_string(),
            id: None,
            retry: None,
        },
        ProviderKind::OpenAiResponses => SseEvent {
            event: Some("response.completed".into()),
            data: serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "model": "m1",
                    "output": [],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15
                    },
                    "status": "completed"
                }
            })
            .to_string(),
            id: None,
            retry: None,
        },
        ProviderKind::Anthropic => SseEvent {
            event: Some("message_delta".into()),
            data: serde_json::json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": null,
                    "stop_sequence": null
                },
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5
                }
            })
            .to_string(),
            id: None,
            retry: None,
        },
        ProviderKind::Gemini => SseEvent {
            event: None,
            data: serde_json::json!({
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15
                }
            })
            .to_string(),
            id: None,
            retry: None,
        },
    }
}

fn sample_stream_done_frame(provider: ProviderKind) -> SseEvent {
    match provider {
        ProviderKind::Anthropic => SseEvent {
            event: Some("message_stop".into()),
            data: serde_json::json!({
                "type": "message_stop"
            })
            .to_string(),
            id: None,
            retry: None,
        },
        ProviderKind::OpenAi
        | ProviderKind::OpenAiResponses
        | ProviderKind::Gemini
        | ProviderKind::GeminiOpenAi => SseEvent {
            event: None,
            data: "[DONE]".into(),
            id: None,
            retry: None,
        },
    }
}

fn sample_stream_reasoning_frame(provider: ProviderKind) -> Option<SseEvent> {
    if provider != ProviderKind::Anthropic {
        return None;
    }
    Some(SseEvent {
        event: Some("content_block_delta".into()),
        data: serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "bench-reasoning"}
        })
        .to_string(),
        id: None,
        retry: None,
    })
}

fn sample_stream_error_frame(provider: ProviderKind) -> Option<SseEvent> {
    match provider {
        ProviderKind::Anthropic => Some(SseEvent {
            event: Some("error".into()),
            data: serde_json::json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "boom"
                }
            })
            .to_string(),
            id: None,
            retry: None,
        }),
        ProviderKind::OpenAiResponses => Some(SseEvent {
            event: Some("error".into()),
            data: serde_json::json!({
                "type": "error",
                "message": "boom"
            })
            .to_string(),
            id: None,
            retry: None,
        }),
        ProviderKind::OpenAi | ProviderKind::Gemini | ProviderKind::GeminiOpenAi => None,
    }
}

fn sample_stream_frame(provider: ProviderKind, scenario: StreamScenario) -> Option<SseEvent> {
    match scenario {
        StreamScenario::TextDelta => Some(sample_stream_text_delta_frame(provider)),
        StreamScenario::ToolCall => Some(sample_stream_tool_call_frame(provider)),
        StreamScenario::Usage => Some(sample_stream_usage_frame(provider)),
        StreamScenario::Done => Some(sample_stream_done_frame(provider)),
        StreamScenario::Reasoning => sample_stream_reasoning_frame(provider),
        StreamScenario::Error => sample_stream_error_frame(provider),
    }
}

fn sample_stream_tool_cycle_frames(provider: ProviderKind) -> Vec<SseEvent> {
    match provider {
        ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => vec![
            sample_stream_tool_call_frame(provider),
            SseEvent {
                event: None,
                data: serde_json::json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "model": "m1",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls"
                    }]
                })
                .to_string(),
                id: None,
                retry: None,
            },
            sample_stream_done_frame(provider),
        ],
        ProviderKind::OpenAiResponses => vec![
            sample_stream_tool_call_frame(provider),
            SseEvent {
                event: Some("response.function_call_arguments.delta".into()),
                data: serde_json::json!({
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "delta": "{\"q\":\"x\"}"
                })
                .to_string(),
                id: None,
                retry: None,
            },
            SseEvent {
                event: Some("response.output_item.done".into()),
                data: serde_json::json!({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_0",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": "{\"q\":\"x\"}"
                    }
                })
                .to_string(),
                id: None,
                retry: None,
            },
            SseEvent {
                event: Some("response.output_item.done".into()),
                data: serde_json::json!({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "function_call_output",
                        "id": "fco_0",
                        "call_id": "call_1",
                        "output": "{\"ok\":true}"
                    }
                })
                .to_string(),
                id: None,
                retry: None,
            },
            sample_stream_usage_frame(provider),
            sample_stream_done_frame(provider),
        ],
        ProviderKind::Anthropic => vec![
            sample_stream_tool_call_frame(provider),
            SseEvent {
                event: Some("content_block_delta".into()),
                data: serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": "{\"q\":\"x\"}"
                    }
                })
                .to_string(),
                id: None,
                retry: None,
            },
            SseEvent {
                event: Some("content_block_stop".into()),
                data: serde_json::json!({
                    "type": "content_block_stop",
                    "index": 0
                })
                .to_string(),
                id: None,
                retry: None,
            },
            SseEvent {
                event: Some("message_delta".into()),
                data: serde_json::json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "tool_use",
                        "stop_sequence": null
                    },
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5
                    }
                })
                .to_string(),
                id: None,
                retry: None,
            },
            sample_stream_done_frame(provider),
        ],
        ProviderKind::Gemini => vec![
            sample_stream_tool_call_frame(provider),
            sample_stream_usage_frame(provider),
            sample_stream_done_frame(provider),
        ],
    }
}

fn sample_stream_text_usage_done_frames(provider: ProviderKind) -> Vec<SseEvent> {
    match provider {
        ProviderKind::Anthropic => vec![
            sample_stream_text_delta_frame(provider),
            SseEvent {
                event: Some("message_delta".into()),
                data: serde_json::json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "end_turn",
                        "stop_sequence": null
                    },
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5
                    }
                })
                .to_string(),
                id: None,
                retry: None,
            },
            sample_stream_done_frame(provider),
        ],
        ProviderKind::OpenAi
        | ProviderKind::OpenAiResponses
        | ProviderKind::Gemini
        | ProviderKind::GeminiOpenAi => vec![
            sample_stream_text_delta_frame(provider),
            sample_stream_usage_frame(provider),
            sample_stream_done_frame(provider),
        ],
    }
}

fn sample_stream_error_sequence_frames(provider: ProviderKind) -> Option<Vec<SseEvent>> {
    sample_stream_error_frame(provider).map(|error| vec![error])
}

fn sample_stream_success_combo_frames(provider: ProviderKind) -> Vec<SseEvent> {
    let mut frames = Vec::with_capacity(5);
    frames.push(sample_stream_text_delta_frame(provider));
    if let Some(reasoning) = sample_stream_reasoning_frame(provider) {
        frames.push(reasoning);
    }
    frames.push(sample_stream_tool_call_frame(provider));
    frames.push(sample_stream_usage_frame(provider));
    frames.push(sample_stream_done_frame(provider));
    frames
}

fn sample_stream_error_combo_frames(provider: ProviderKind) -> Option<Vec<SseEvent>> {
    let error = sample_stream_error_frame(provider)?;
    let mut frames = Vec::with_capacity(4);
    frames.push(sample_stream_text_delta_frame(provider));
    if let Some(reasoning) = sample_stream_reasoning_frame(provider) {
        frames.push(reasoning);
    }
    frames.push(sample_stream_tool_call_frame(provider));
    frames.push(error);
    Some(frames)
}

fn bench_stream_transcode_matrix(c: &mut Criterion) {
    let providers = [
        ("openai", ProviderKind::OpenAi),
        ("responses", ProviderKind::OpenAiResponses),
        ("anthropic", ProviderKind::Anthropic),
        ("gemini", ProviderKind::Gemini),
        ("gemini_openai", ProviderKind::GeminiOpenAi),
    ];
    let apis = [
        ("chat", IngressApi::OpenAiChat),
        ("responses", IngressApi::OpenAiResponses),
        ("anthropic", IngressApi::Anthropic),
        ("gemini", IngressApi::Gemini),
    ];
    let scenarios = [
        StreamScenario::TextDelta,
        StreamScenario::ToolCall,
        StreamScenario::Usage,
        StreamScenario::Done,
        StreamScenario::Reasoning,
        StreamScenario::Error,
    ];

    for (provider_name, provider) in providers {
        for (api_name, api) in apis {
            for scenario in scenarios {
                let Some(frame) = sample_stream_frame(provider, scenario) else {
                    continue;
                };
                let name = format!(
                    "stream_transcode_{}_{}_to_{}",
                    scenario.name(),
                    provider_name,
                    api_name
                );
                let mut transcoder =
                    StreamTranscoder::new(provider, api, "m1".to_string(), "id-1".to_string());
                c.bench_function(&name, |b| {
                    b.iter(|| black_box(transcoder.transcode_frame(black_box(&frame))));
                });
            }
        }
    }
}

fn bench_stream_transcode_sequence_matrix(c: &mut Criterion) {
    let providers = [
        ("openai", ProviderKind::OpenAi),
        ("responses", ProviderKind::OpenAiResponses),
        ("anthropic", ProviderKind::Anthropic),
        ("gemini", ProviderKind::Gemini),
        ("gemini_openai", ProviderKind::GeminiOpenAi),
    ];
    let apis = [
        ("chat", IngressApi::OpenAiChat),
        ("responses", IngressApi::OpenAiResponses),
        ("anthropic", IngressApi::Anthropic),
        ("gemini", IngressApi::Gemini),
    ];

    for (provider_name, provider) in providers {
        for (api_name, api) in apis {
            let fixed_sequences = [
                ("tool_cycle", sample_stream_tool_cycle_frames(provider)),
                (
                    "text_usage_done",
                    sample_stream_text_usage_done_frames(provider),
                ),
                (
                    "success_combo",
                    sample_stream_success_combo_frames(provider),
                ),
            ];
            for (sequence_name, frames) in fixed_sequences {
                let name =
                    format!("stream_transcode_seq_{sequence_name}_{provider_name}_to_{api_name}");
                c.bench_function(&name, |b| {
                    b.iter(|| {
                        let mut transcoder = StreamTranscoder::new(
                            provider,
                            api,
                            "m1".to_string(),
                            "id-1".to_string(),
                        );
                        let mut produced_frames = 0usize;
                        for frame in &frames {
                            produced_frames +=
                                black_box(transcoder.transcode_frame(black_box(frame))).len();
                        }
                        black_box(produced_frames);
                    });
                });
            }

            if let Some(frames) = sample_stream_error_sequence_frames(provider) {
                let name = format!("stream_transcode_seq_error_{provider_name}_to_{api_name}");
                c.bench_function(&name, |b| {
                    b.iter(|| {
                        let mut transcoder = StreamTranscoder::new(
                            provider,
                            api,
                            "m1".to_string(),
                            "id-1".to_string(),
                        );
                        let mut produced_frames = 0usize;
                        for frame in &frames {
                            produced_frames +=
                                black_box(transcoder.transcode_frame(black_box(frame))).len();
                        }
                        black_box(produced_frames);
                    });
                });
            }

            if let Some(frames) = sample_stream_error_combo_frames(provider) {
                let name =
                    format!("stream_transcode_seq_error_combo_{provider_name}_to_{api_name}");
                c.bench_function(&name, |b| {
                    b.iter(|| {
                        let mut transcoder = StreamTranscoder::new(
                            provider,
                            api,
                            "m1".to_string(),
                            "id-1".to_string(),
                        );
                        let mut produced_frames = 0usize;
                        for frame in &frames {
                            produced_frames +=
                                black_box(transcoder.transcode_frame(black_box(frame))).len();
                        }
                        black_box(produced_frames);
                    });
                });
            }
        }
    }
}

criterion_group!(
    benches,
    bench_transcode,
    bench_stream_transcode_matrix,
    bench_stream_transcode_sequence_matrix
);
criterion_main!(benches);
