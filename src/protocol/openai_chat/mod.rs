pub mod decoder;
pub mod encoder;
pub mod response_decoder;
pub mod response_encoder;
pub mod stream;

use serde::{Deserialize, Serialize};

/// `OpenAI` Chat Completion request wire type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiChatRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAiToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<OpenAiStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<OpenAiStop>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// `tool_choice` field in `OpenAI` Chat requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAiToolChoice {
    Mode(String),
    Function(OpenAiToolChoiceFunctionCall),
}

/// `tool_choice` object form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolChoiceFunctionCall {
    #[serde(rename = "type")]
    pub type_: String,
    pub function: OpenAiToolChoiceFunction,
}

/// Nested function selector in `tool_choice`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolChoiceFunction {
    pub name: String,
}

/// `stop` field in `OpenAI` Chat requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAiStop {
    Single(String),
    Multi(Vec<String>),
}

/// Stream options for `include_usage`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiStreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// `OpenAI` Chat Completion response wire type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiChatResponse {
    pub id: String,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
    pub model: String,
    pub choices: Vec<OpenAiChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAiUsage>,
}

/// A single choice in the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiChoice {
    pub index: u32,
    pub message: OpenAiMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// `OpenAI` message wire type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

/// A tool call within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub function: OpenAiToolCallFunction,
}

/// The function part of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// A tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiTool {
    #[serde(rename = "type")]
    pub type_: String,
    pub function: OpenAiToolFunction,
}

/// A function declaration within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Usage info in the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// A streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiStreamChunk {
    pub id: String,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
    pub model: String,
    pub choices: Vec<OpenAiStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAiUsage>,
}

/// A choice within a stream chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiStreamChoice {
    pub index: u32,
    pub delta: OpenAiDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Delta content within a stream choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiStreamToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

/// A tool call delta in streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiStreamToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<OpenAiStreamToolCallFunction>,
}

/// Function delta within a streaming tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiStreamToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}
