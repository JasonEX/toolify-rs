use serde::Deserialize;

use crate::api::engine::pipeline::{find_top_level_field_value_range, raw_tools_field_has_items};
use crate::error::CanonicalError;

#[derive(Deserialize)]
struct GeminiProbe<'a> {
    #[serde(default, borrow)]
    tools: Option<&'a serde_json::value::RawValue>,
}

pub(crate) fn parse_gemini_has_tools(body: &bytes::Bytes) -> Result<bool, CanonicalError> {
    match find_top_level_field_value_range(body.as_ref(), b"tools") {
        Ok(Some(range)) => return Ok(raw_gemini_tools_token_has_items(&body[range])),
        Ok(None) => return Ok(false),
        Err(()) => {}
    }

    let probe = parse_gemini_probe(body)?;
    Ok(raw_tools_field_has_items(probe.tools))
}

fn parse_gemini_probe(body: &bytes::Bytes) -> Result<GeminiProbe<'_>, CanonicalError> {
    serde_json::from_slice(body)
        .map_err(|e| CanonicalError::InvalidRequest(format!("Invalid Gemini request body: {e}")))
}

fn raw_gemini_tools_token_has_items(token: &[u8]) -> bool {
    if token.first() != Some(&b'[') || token.last() != Some(&b']') {
        return true;
    }
    !token[1..token.len() - 1]
        .iter()
        .all(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
}
