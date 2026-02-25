use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::Arc;

pub type ProviderExtensions = serde_json::Map<String, serde_json::Value>;

/// Which ingress API the request arrived on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IngressApi {
    OpenAiChat,
    OpenAiResponses,
    Anthropic,
    Gemini,
}

/// The kind of provider an upstream service speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderKind {
    OpenAi,
    OpenAiResponses,
    Anthropic,
    Gemini,
    GeminiOpenAi,
}

/// Canonical message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CanonicalRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Reason the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanonicalStopReason {
    EndOfTurn,
    ToolCalls,
    MaxTokens,
    ContentFilter,
}

/// Tool choice specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalToolChoice {
    Auto,
    None,
    Required,
    Specific(String),
}

/// Token usage information.
#[derive(Debug, Clone, Default)]
pub struct CanonicalUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// Generation parameters passed through to the upstream.
#[derive(Debug, Clone, Default)]
pub struct GenerationParams {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub top_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub n: Option<u32>,
    pub stop: Option<Vec<String>>,
}

/// A single part of a message's content.
#[derive(Debug, Clone)]
pub enum CanonicalPart {
    Text(String),
    ReasoningText(String),
    ImageUrl {
        url: String,
        detail: Option<String>,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Box<serde_json::value::RawValue>,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
    },
    Refusal(String),
}

/// A single message in the canonical conversation.
#[derive(Debug, Clone)]
pub struct CanonicalMessage {
    pub role: CanonicalRole,
    pub parts: SmallVec<[CanonicalPart; 1]>,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
    pub provider_extensions: Option<Box<ProviderExtensions>>,
}

/// A tool's function declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalToolFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

/// A tool specification in the request.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalToolSpec {
    pub function: CanonicalToolFunction,
}

/// The fully-decoded, provider-agnostic request.
#[derive(Debug, Clone)]
pub struct CanonicalRequest {
    pub request_id: uuid::Uuid,
    pub ingress_api: IngressApi,
    pub model: String,
    pub stream: bool,
    pub system_prompt: Option<String>,
    pub messages: Vec<CanonicalMessage>,
    pub tools: Arc<[CanonicalToolSpec]>,
    pub tool_choice: CanonicalToolChoice,
    pub generation: GenerationParams,
    pub provider_extensions: Option<Box<ProviderExtensions>>,
}

/// The fully-decoded, provider-agnostic non-streaming response.
#[derive(Debug, Clone)]
pub struct CanonicalResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<CanonicalPart>,
    pub stop_reason: CanonicalStopReason,
    pub usage: CanonicalUsage,
    pub provider_extensions: ProviderExtensions,
}

#[must_use]
pub fn provider_extensions_from_map(map: ProviderExtensions) -> Option<Box<ProviderExtensions>> {
    if map.is_empty() {
        None
    } else {
        Some(Box::new(map))
    }
}

#[must_use]
pub fn provider_extensions_to_map(value: &Option<Box<ProviderExtensions>>) -> ProviderExtensions {
    value.as_deref().cloned().unwrap_or_default()
}

impl CanonicalRequest {
    #[must_use]
    pub fn provider_extensions_ref(&self) -> &ProviderExtensions {
        match self.provider_extensions.as_deref() {
            Some(ext) => ext,
            None => empty_extensions(),
        }
    }

    #[must_use]
    pub fn provider_extensions_mut(&mut self) -> &mut ProviderExtensions {
        self.provider_extensions
            .get_or_insert_with(|| Box::new(ProviderExtensions::new()))
    }
}

fn empty_extensions() -> &'static ProviderExtensions {
    static EMPTY: std::sync::LazyLock<ProviderExtensions> =
        std::sync::LazyLock::new(ProviderExtensions::new);
    &EMPTY
}

/// A single event in a canonical stream.
#[derive(Debug, Clone)]
pub enum CanonicalStreamEvent {
    MessageStart {
        role: CanonicalRole,
    },
    TextDelta(String),
    ReasoningDelta(String),
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgsDelta {
        index: usize,
        delta: String,
    },
    ToolCallEnd {
        index: usize,
        call_id: Option<String>,
        call_name: Option<String>,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
    },
    Usage(CanonicalUsage),
    MessageEnd {
        stop_reason: CanonicalStopReason,
    },
    Done,
    Error {
        status: u16,
        message: String,
    },
}

/// Tracks canonical call IDs to provider-specific IDs.
#[derive(Debug, Clone, Default)]
pub struct CallIdBindings {
    bindings: FxHashMap<String, ProviderCallIds>,
}

impl CallIdBindings {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, canonical_id: String, ids: ProviderCallIds) {
        self.bindings.insert(canonical_id, ids);
    }

    #[must_use]
    pub fn get(&self, canonical_id: &str) -> Option<&ProviderCallIds> {
        self.bindings.get(canonical_id)
    }
}

/// Provider-specific call IDs for a single canonical tool call.
#[derive(Debug, Clone)]
pub enum ProviderCallIds {
    OpenAi(String),
    Anthropic(String),
    Gemini(String),
}
