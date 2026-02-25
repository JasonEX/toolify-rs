use crate::api::engine::pipeline::{parse_common_request_probe, CommonRequestProbe};
use crate::error::CanonicalError;

pub(crate) fn parse_anthropic_probe(
    body: &bytes::Bytes,
) -> Result<CommonRequestProbe<'_>, CanonicalError> {
    parse_common_request_probe(body, "Anthropic request")
}
