use crate::api::engine::pipeline::{
    find_top_level_field_value_range, parse_common_request_probe, CommonRequestProbe,
};
use crate::api::ingress::openai_responses::fc::responses_tools_token_has_function;
use crate::error::CanonicalError;

pub(crate) fn parse_openai_responses_probe(
    body: &bytes::Bytes,
) -> Result<CommonRequestProbe<'_>, CanonicalError> {
    let mut probe = parse_common_request_probe(body, "OpenAI Responses request")?;
    if !probe.has_tools {
        return Ok(probe);
    }

    if let Ok(Some(range)) = find_top_level_field_value_range(body.as_ref(), b"tools") {
        probe.has_tools = responses_tools_token_has_function(&body[range]);
    }

    Ok(probe)
}
