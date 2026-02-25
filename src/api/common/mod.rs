//! Shared API helpers reused across ingress handlers.

mod codec;
mod io;
mod non_streaming;
mod passthrough;
mod probe;
mod streaming;

pub(crate) use crate::json_scan::{
    find_top_level_field_value_range, parse_json_string_end, parse_json_value_end, skip_ws,
};
pub(crate) use codec::{decode_response_from_provider, encode_for_provider};
pub(crate) use io::{
    prepare_upstream_io_request, send_non_streaming_bytes, PreparedUpstreamIoRequest,
    UpstreamIoRequest,
};
pub(crate) use non_streaming::{
    handle_non_streaming_common, handle_non_streaming_preencoded_common,
};
pub(crate) use passthrough::{
    is_protocol_passthrough, passthrough_non_streaming_bytes, passthrough_non_streaming_uri_bytes,
    passthrough_non_streaming_url_bytes, passthrough_streaming_bytes,
    passthrough_streaming_uri_bytes, passthrough_streaming_url_bytes, sanitize_upstream_error,
};
pub(crate) use probe::{
    find_common_probe_field_ranges, parse_common_request_probe, parse_optional_bool_token,
    raw_tools_field_has_items, raw_tools_token_has_items,
    rewrite_model_field_in_json_body_with_range, CommonProbeRanges, CommonRequestProbe,
};
pub(crate) use streaming::handle_streaming_request;
