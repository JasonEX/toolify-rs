use crate::error::CanonicalError;
use crate::protocol::anthropic::{AnthropicContentBlock, AnthropicResponse, AnthropicUsage};
use crate::protocol::canonical::{CanonicalPart, CanonicalResponse};
use crate::protocol::mapping::canonical_stop_to_anthropic;
use crate::util::next_generated_id;
use std::sync::atomic::AtomicU64;

static GENERATED_MSG_ID_SEQ: AtomicU64 = AtomicU64::new(1);

fn next_generated_msg_id() -> String {
    next_generated_id("msg", &GENERATED_MSG_ID_SEQ)
}

/// Encode a canonical response into the Anthropic Messages API wire format.
///
/// # Errors
///
/// Returns [`CanonicalError`] when canonical tool-call arguments cannot be
/// decoded into JSON values.
pub fn encode_anthropic_response(
    canonical: &CanonicalResponse,
    model: &str,
) -> Result<AnthropicResponse, CanonicalError> {
    // --- content blocks ---
    let mut content = Vec::new();
    for part in &canonical.content {
        match part {
            CanonicalPart::ReasoningText(text) => {
                content.push(AnthropicContentBlock::Thinking {
                    thinking: text.clone(),
                });
            }
            CanonicalPart::ToolCall {
                id,
                name,
                arguments,
            } => {
                let input: serde_json::Value =
                    serde_json::from_str(arguments.get()).unwrap_or(serde_json::json!({}));
                content.push(AnthropicContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input,
                });
            }
            CanonicalPart::ToolResult {
                tool_call_id,
                content: result_content,
            } => {
                content.push(AnthropicContentBlock::ToolResult {
                    tool_use_id: tool_call_id.clone(),
                    content: serde_json::Value::String(result_content.clone()),
                });
            }
            CanonicalPart::ImageUrl { .. } => {
                // Images are not part of response content blocks — skip
            }
            CanonicalPart::Text(text) | CanonicalPart::Refusal(text) => {
                content.push(AnthropicContentBlock::Text { text: text.clone() });
            }
        }
    }

    // --- stop reason ---
    let stop_reason = Some(canonical_stop_to_anthropic(canonical.stop_reason).to_string());

    // --- usage ---
    let usage = AnthropicUsage {
        input_tokens: canonical.usage.input_tokens.unwrap_or(0),
        output_tokens: canonical.usage.output_tokens.unwrap_or(0),
    };

    // --- generate id if empty ---
    let id = if canonical.id.is_empty() {
        next_generated_msg_id()
    } else {
        canonical.id.clone()
    };

    Ok(AnthropicResponse {
        id,
        type_: "message".to_string(),
        role: "assistant".to_string(),
        model: model.to_string(),
        content,
        stop_reason,
        stop_sequence: None,
        usage,
    })
}
