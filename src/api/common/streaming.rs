use axum::response::Response;
use futures_util::StreamExt;
use smallvec::SmallVec;
use std::sync::LazyLock;

use crate::api::common::io::UpstreamIoRequest;
use crate::api::common::passthrough::{is_protocol_passthrough, sanitize_upstream_error};
use crate::error::CanonicalError;
use crate::fc;
use crate::protocol::canonical::{CanonicalToolSpec, IngressApi, ProviderKind};
use crate::stream::sse::{sse_frame_stream, sse_raw_frame_stream};
use crate::stream::transcoder::StreamTranscoder;
use crate::stream::{parse_sse_frame_bytes, StreamingFcProcessor};

const FUNCTION_CALLS_OPEN_TAG_BYTES: &[u8] = b"<function_calls>";
static TRIGGER_SIGNAL_FINDER: LazyLock<memchr::memmem::Finder<'static>> =
    LazyLock::new(|| memchr::memmem::Finder::new(fc::prompt::get_trigger_signal().as_bytes()));

#[inline]
fn move_byte_chunks_to_pending(frame_chunks: &mut Vec<bytes::Bytes>, pending: &mut PendingBytes) {
    pending.extend_from_bytes(frame_chunks);
}

#[inline]
fn emit_from_byte_chunks(
    frame_chunks: &mut Vec<bytes::Bytes>,
    pending: &mut PendingBytes,
) -> Option<bytes::Bytes> {
    match frame_chunks.len() {
        0 => None,
        1 => frame_chunks.pop(),
        _ => {
            move_byte_chunks_to_pending(frame_chunks, pending);
            pending.pop_front()
        }
    }
}

#[inline]
fn parse_raw_sse_frame_bytes(raw: &[u8]) -> Option<crate::stream::SseEvent> {
    parse_sse_frame_bytes(raw)
}

#[inline]
fn parse_openai_raw_sse_data_bounds(bytes: &[u8]) -> Option<(usize, usize)> {
    let end = if bytes.ends_with(b"\r\n\r\n") {
        bytes.len().saturating_sub(4)
    } else if bytes.ends_with(b"\n\n") {
        bytes.len().saturating_sub(2)
    } else {
        return None;
    };

    if end < 5 || !bytes[..end].starts_with(b"data:") {
        return None;
    }

    let start = 5 + usize::from(bytes[5] == b' ');
    let data_bytes = &bytes[start..end];
    if memchr::memchr2(b'\n', b'\r', data_bytes).is_some() {
        return None;
    }

    Some((start, end))
}

#[inline]
fn parse_openai_raw_sse_data_bytes(raw: &[u8]) -> Option<&[u8]> {
    let (start, end) = parse_openai_raw_sse_data_bounds(raw)?;
    raw.get(start..end)
}

#[inline]
fn text_contains_fc_patterns_bytes(text: &[u8], trigger_signal: &str) -> bool {
    memchr::memmem::find(text, FUNCTION_CALLS_OPEN_TAG_BYTES).is_some()
        || trigger_signal_in_bytes(text, trigger_signal)
}

#[inline]
fn trigger_signal_in_bytes(text: &[u8], trigger_signal: &str) -> bool {
    if std::ptr::eq(trigger_signal, fc::prompt::get_trigger_signal()) {
        TRIGGER_SIGNAL_FINDER.find(text).is_some()
    } else {
        memchr::memmem::find(text, trigger_signal.as_bytes()).is_some()
    }
}

#[inline]
fn frame_might_start_fc_with_parsed_bytes(
    raw: &[u8],
    parsed_data: Option<&[u8]>,
    trigger_signal: &str,
) -> bool {
    if let Some(data) = parsed_data {
        memchr::memchr(b'<', data).is_some()
            && text_contains_fc_patterns_bytes(data, trigger_signal)
    } else {
        text_contains_fc_patterns_bytes(raw, trigger_signal)
    }
}

fn try_start_passthrough_fc_processor(
    raw_frame: &bytes::Bytes,
    provider_kind: ProviderKind,
    ingress_api: IngressApi,
    model: &str,
    response_id: &str,
    frame_chunks: &mut Vec<bytes::Bytes>,
) -> Option<StreamingFcProcessor> {
    let openai_chat_passthrough_fast = ingress_api == IngressApi::OpenAiChat
        && matches!(
            provider_kind,
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi
        );
    let trigger_signal = fc::prompt::get_trigger_signal();

    let transcoder = StreamTranscoder::new(
        provider_kind,
        ingress_api,
        model.to_owned(),
        response_id.to_owned(),
    );
    let mut proc = StreamingFcProcessor::new(transcoder, true, &[], trigger_signal);

    if openai_chat_passthrough_fast {
        let parsed_data = parse_openai_raw_sse_data_bytes(raw_frame.as_ref());
        let should_activate =
            frame_might_start_fc_with_parsed_bytes(raw_frame.as_ref(), parsed_data, trigger_signal);
        if !should_activate {
            return None;
        }

        if let Some(data) = parsed_data {
            if !proc.try_process_openai_data_frame_bytes_into_bytes(data, frame_chunks) {
                return None;
            }
        } else if !proc.try_process_raw_frame_into_bytes(raw_frame.as_ref(), frame_chunks) {
            return None;
        }
    } else {
        let frame = parse_raw_sse_frame_bytes(raw_frame.as_ref())?;
        if !frame.data.as_bytes().contains(&b'<') {
            return None;
        }
        proc.process_frame_into_bytes(&frame, frame_chunks);
    }

    Some(proc)
}

#[inline]
fn sse_ok_response(body: axum::body::Body) -> Response {
    sse_ok_response_with_content_type(body, http::HeaderValue::from_static("text/event-stream"))
}

#[inline]
fn sse_ok_response_with_content_type(
    body: axum::body::Body,
    content_type: http::HeaderValue,
) -> Response {
    let mut response = Response::new(body);
    *response.status_mut() = http::StatusCode::OK;
    let headers = response.headers_mut();
    headers.insert(http::header::CONTENT_TYPE, content_type);
    headers.insert(
        http::header::CACHE_CONTROL,
        http::HeaderValue::from_static("no-cache"),
    );
    headers.insert(
        http::header::CONNECTION,
        http::HeaderValue::from_static("keep-alive"),
    );
    response
}

struct PendingBytes {
    chunks: SmallVec<[bytes::Bytes; 8]>,
    head: usize,
}

impl PendingBytes {
    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        let mut chunks = SmallVec::new();
        chunks.reserve(capacity);
        Self { chunks, head: 0 }
    }

    #[inline]
    fn pop_front(&mut self) -> Option<bytes::Bytes> {
        if self.head >= self.chunks.len() {
            return None;
        }
        let chunk = std::mem::take(&mut self.chunks[self.head]);
        self.head += 1;
        if self.head == self.chunks.len() {
            self.chunks.clear();
            self.head = 0;
        }
        Some(chunk)
    }

    #[inline]
    fn extend_from_bytes(&mut self, frame_chunks: &mut Vec<bytes::Bytes>) {
        if frame_chunks.is_empty() {
            return;
        }
        self.chunks.reserve(frame_chunks.len());
        self.chunks.extend(frame_chunks.drain(..));
    }
}

pub(crate) async fn handle_streaming_request(
    ctx: UpstreamIoRequest<'_>,
    upstream_body: bytes::Bytes,
    ingress: IngressApi,
    response_id: String,
    fc_active: bool,
    saved_tools: &[CanonicalToolSpec],
) -> Result<Response, CanonicalError> {
    if ctx
        .state
        .transport
        .hyper_passthrough_enabled_for(ctx.proxy_url)
    {
        use http_body_util::BodyExt as _;

        let response = if let Some(parsed_hyper_uri) = ctx.parsed_hyper_uri {
            ctx.state
                .transport
                .send_stream_uri(
                    parsed_hyper_uri,
                    http::Method::POST,
                    ctx.upstream_headers,
                    upstream_body,
                )
                .await?
        } else {
            ctx.state
                .transport
                .send_stream_uri_str(
                    ctx.url,
                    http::Method::POST,
                    ctx.upstream_headers,
                    upstream_body,
                )
                .await?
        };
        let status = response.status();
        let content_type = response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .cloned()
            .unwrap_or_else(|| http::HeaderValue::from_static("text/event-stream"));
        let (_, body) = response.into_parts();

        if !status.is_success() {
            let body_bytes = body
                .collect()
                .await
                .map(http_body_util::Collected::to_bytes)
                .map_err(|e| {
                    CanonicalError::Transport(format!("Failed to read error body: {e}"))
                })?;
            return Err(CanonicalError::Upstream {
                status: status.as_u16(),
                message: sanitize_upstream_error(&body_bytes),
            });
        }

        if !fc_active && is_protocol_passthrough(ctx.provider, ingress) {
            return Ok(sse_ok_response_with_content_type(
                axum::body::Body::new(body),
                content_type,
            ));
        }

        return Ok(build_transcoded_stream_response(
            body.into_data_stream(),
            ctx.provider,
            ingress,
            ctx.client_model,
            response_id,
            fc_active,
            saved_tools,
        ));
    }

    let response = if let Some(parsed_url) = ctx.parsed_url {
        ctx.state
            .transport
            .send_stream_url_with_client(
                parsed_url,
                http::Method::POST,
                ctx.upstream_headers,
                upstream_body,
                ctx.proxy_url,
                ctx.preconfigured_proxy_client,
            )
            .await?
    } else {
        ctx.state
            .transport
            .send_stream_with_client(
                ctx.url,
                http::Method::POST,
                ctx.upstream_headers,
                upstream_body,
                ctx.proxy_url,
                ctx.preconfigured_proxy_client,
            )
            .await?
    };

    let status = response.status();
    if !status.is_success() {
        let body_bytes = response
            .bytes()
            .await
            .map_err(|e| CanonicalError::Transport(format!("Failed to read error body: {e}")))?;
        return Err(CanonicalError::Upstream {
            status: status.as_u16(),
            message: sanitize_upstream_error(&body_bytes),
        });
    }

    let byte_stream = response.bytes_stream();
    if !fc_active && is_protocol_passthrough(ctx.provider, ingress) {
        let body = axum::body::Body::from_stream(byte_stream);
        return Ok(sse_ok_response(body));
    }

    Ok(build_transcoded_stream_response(
        byte_stream,
        ctx.provider,
        ingress,
        ctx.client_model,
        response_id,
        fc_active,
        saved_tools,
    ))
}

pub(crate) fn build_transcoded_stream_response<E>(
    byte_stream: impl futures_util::Stream<Item = Result<bytes::Bytes, E>> + Send + 'static,
    provider: ProviderKind,
    ingress: IngressApi,
    client_model: &str,
    response_id: String,
    fc_active: bool,
    saved_tools: &[CanonicalToolSpec],
) -> Response
where
    E: std::fmt::Debug + Send + 'static,
{
    if fc_active {
        return build_fc_transcoded_stream_response(
            byte_stream,
            provider,
            ingress,
            client_model,
            response_id,
            saved_tools,
        );
    }

    build_non_fc_transcoded_stream_response(
        byte_stream,
        provider,
        ingress,
        client_model,
        response_id,
    )
}

fn build_fc_transcoded_stream_response<E>(
    byte_stream: impl futures_util::Stream<Item = Result<bytes::Bytes, E>> + Send + 'static,
    provider: ProviderKind,
    ingress: IngressApi,
    client_model: &str,
    response_id: String,
    saved_tools: &[CanonicalToolSpec],
) -> Response
where
    E: std::fmt::Debug + Send + 'static,
{
    if is_protocol_passthrough(provider, ingress) {
        let output_stream = futures_util::stream::unfold(
            (
                Box::pin(sse_raw_frame_stream(byte_stream)),
                None::<StreamingFcProcessor>,
                Vec::<bytes::Bytes>::with_capacity(8),
                PendingBytes::with_capacity(8),
                false,
                provider,
                ingress,
                client_model.to_string(),
                response_id,
            ),
            |(
                mut sse_stream,
                mut processor,
                mut frame_chunks,
                mut pending,
                mut finalized,
                provider_kind,
                ingress_api,
                model,
                response_id,
            )| async move {
                loop {
                    if let Some(chunk) = pending.pop_front() {
                        return Some((
                            chunk,
                            (
                                sse_stream,
                                processor,
                                frame_chunks,
                                pending,
                                finalized,
                                provider_kind,
                                ingress_api,
                                model,
                                response_id,
                            ),
                        ));
                    }
                    if finalized {
                        return None;
                    }
                    if let Some(raw_frame) = sse_stream.as_mut().next().await {
                        if let Some(proc) = processor.as_mut() {
                            if proc.try_process_raw_frame_into_bytes(
                                raw_frame.as_ref(),
                                &mut frame_chunks,
                            ) {
                                move_byte_chunks_to_pending(&mut frame_chunks, &mut pending);
                                continue;
                            }
                            return Some((
                                raw_frame,
                                (
                                    sse_stream,
                                    processor,
                                    frame_chunks,
                                    pending,
                                    finalized,
                                    provider_kind,
                                    ingress_api,
                                    model,
                                    response_id,
                                ),
                            ));
                        }

                        if !raw_frame.as_ref().contains(&b'<') {
                            return Some((
                                raw_frame,
                                (
                                    sse_stream,
                                    processor,
                                    frame_chunks,
                                    pending,
                                    finalized,
                                    provider_kind,
                                    ingress_api,
                                    model,
                                    response_id,
                                ),
                            ));
                        }

                        if let Some(proc) = try_start_passthrough_fc_processor(
                            &raw_frame,
                            provider_kind,
                            ingress_api,
                            &model,
                            &response_id,
                            &mut frame_chunks,
                        ) {
                            move_byte_chunks_to_pending(&mut frame_chunks, &mut pending);
                            processor = Some(proc);
                        } else {
                            return Some((
                                raw_frame,
                                (
                                    sse_stream,
                                    processor,
                                    frame_chunks,
                                    pending,
                                    finalized,
                                    provider_kind,
                                    ingress_api,
                                    model,
                                    response_id,
                                ),
                            ));
                        }
                    } else {
                        if let Some(proc) = processor.as_mut() {
                            proc.finalize_into_bytes(&mut frame_chunks);
                            move_byte_chunks_to_pending(&mut frame_chunks, &mut pending);
                        }
                        finalized = true;
                    }
                }
            },
        );

        let body = axum::body::Body::from_stream(
            output_stream.map(Ok::<bytes::Bytes, std::convert::Infallible>),
        );
        return sse_ok_response(body);
    }

    if matches!(provider, ProviderKind::OpenAi | ProviderKind::GeminiOpenAi) {
        return build_fc_transcoded_stream_response_openai_upstream(
            byte_stream,
            provider,
            ingress,
            client_model,
            response_id,
            saved_tools,
        );
    }

    build_fc_transcoded_stream_response_generic(
        byte_stream,
        provider,
        ingress,
        client_model,
        response_id,
        saved_tools,
    )
}

fn build_fc_transcoded_stream_response_openai_upstream<E>(
    byte_stream: impl futures_util::Stream<Item = Result<bytes::Bytes, E>> + Send + 'static,
    provider: ProviderKind,
    ingress: IngressApi,
    client_model: &str,
    response_id: String,
    saved_tools: &[CanonicalToolSpec],
) -> Response
where
    E: std::fmt::Debug + Send + 'static,
{
    let transcoder =
        StreamTranscoder::new(provider, ingress, client_model.to_string(), response_id);
    let processor = StreamingFcProcessor::new(
        transcoder,
        true,
        saved_tools,
        fc::prompt::get_trigger_signal(),
    );
    let output_stream = futures_util::stream::unfold(
        (
            Box::pin(sse_raw_frame_stream(byte_stream)),
            processor,
            Vec::<bytes::Bytes>::with_capacity(8),
            PendingBytes::with_capacity(8),
            false,
        ),
        |(mut sse_stream, mut proc, mut frame_chunks, mut pending, mut finalized)| async move {
            loop {
                if let Some(chunk) = pending.pop_front() {
                    return Some((chunk, (sse_stream, proc, frame_chunks, pending, finalized)));
                }
                if finalized {
                    return None;
                }
                if let Some(raw_frame) = sse_stream.as_mut().next().await {
                    proc.process_raw_frame_into_bytes(raw_frame.as_ref(), &mut frame_chunks);
                } else {
                    proc.finalize_into_bytes(&mut frame_chunks);
                    finalized = true;
                }
                if let Some(chunk) = emit_from_byte_chunks(&mut frame_chunks, &mut pending) {
                    return Some((chunk, (sse_stream, proc, frame_chunks, pending, finalized)));
                }
            }
        },
    );

    let body = axum::body::Body::from_stream(
        output_stream.map(Ok::<bytes::Bytes, std::convert::Infallible>),
    );
    sse_ok_response(body)
}

fn build_fc_transcoded_stream_response_generic<E>(
    byte_stream: impl futures_util::Stream<Item = Result<bytes::Bytes, E>> + Send + 'static,
    provider: ProviderKind,
    ingress: IngressApi,
    client_model: &str,
    response_id: String,
    saved_tools: &[CanonicalToolSpec],
) -> Response
where
    E: std::fmt::Debug + Send + 'static,
{
    let transcoder =
        StreamTranscoder::new(provider, ingress, client_model.to_string(), response_id);
    let sse_events = sse_frame_stream(byte_stream);
    let processor = StreamingFcProcessor::new(
        transcoder,
        true,
        saved_tools,
        fc::prompt::get_trigger_signal(),
    );

    let output_stream = futures_util::stream::unfold(
        (
            Box::pin(sse_events),
            processor,
            Vec::<bytes::Bytes>::with_capacity(8),
            PendingBytes::with_capacity(8),
            false,
        ),
        |(mut sse_stream, mut proc, mut frame_chunks, mut pending, mut finalized)| async move {
            loop {
                if let Some(chunk) = pending.pop_front() {
                    return Some((chunk, (sse_stream, proc, frame_chunks, pending, finalized)));
                }
                if finalized {
                    return None;
                }
                if let Some(frame) = sse_stream.as_mut().next().await {
                    proc.process_frame_into_bytes(&frame, &mut frame_chunks);
                } else {
                    proc.finalize_into_bytes(&mut frame_chunks);
                    finalized = true;
                }
                if let Some(chunk) = emit_from_byte_chunks(&mut frame_chunks, &mut pending) {
                    return Some((chunk, (sse_stream, proc, frame_chunks, pending, finalized)));
                }
            }
        },
    );

    let body = axum::body::Body::from_stream(
        output_stream.map(Ok::<bytes::Bytes, std::convert::Infallible>),
    );
    sse_ok_response(body)
}

fn build_non_fc_transcoded_stream_response<E>(
    byte_stream: impl futures_util::Stream<Item = Result<bytes::Bytes, E>> + Send + 'static,
    provider: ProviderKind,
    ingress: IngressApi,
    client_model: &str,
    response_id: String,
) -> Response
where
    E: std::fmt::Debug + Send + 'static,
{
    if matches!(provider, ProviderKind::OpenAi | ProviderKind::GeminiOpenAi) {
        let transcoder =
            StreamTranscoder::new(provider, ingress, client_model.to_string(), response_id);
        let output_stream = futures_util::stream::unfold(
            (
                Box::pin(sse_raw_frame_stream(byte_stream)),
                transcoder,
                Vec::<crate::protocol::canonical::CanonicalStreamEvent>::with_capacity(8),
                Vec::<bytes::Bytes>::with_capacity(8),
                PendingBytes::with_capacity(8),
                false,
            ),
            |(
                mut sse_stream,
                mut transcoder,
                mut decode_buffer,
                mut frame_chunks,
                mut pending,
                mut done,
            )| async move {
                loop {
                    if let Some(chunk) = pending.pop_front() {
                        return Some((
                            chunk,
                            (
                                sse_stream,
                                transcoder,
                                decode_buffer,
                                frame_chunks,
                                pending,
                                done,
                            ),
                        ));
                    }
                    if done {
                        return None;
                    }
                    if let Some(raw_frame) = sse_stream.as_mut().next().await {
                        let _ = transcoder.transcode_raw_frame_into_bytes_with_decode_buffer(
                            raw_frame.as_ref(),
                            &mut decode_buffer,
                            &mut frame_chunks,
                        );
                        if let Some(chunk) = emit_from_byte_chunks(&mut frame_chunks, &mut pending)
                        {
                            return Some((
                                chunk,
                                (
                                    sse_stream,
                                    transcoder,
                                    decode_buffer,
                                    frame_chunks,
                                    pending,
                                    done,
                                ),
                            ));
                        }
                    } else {
                        done = true;
                    }
                }
            },
        );

        let body = axum::body::Body::from_stream(
            output_stream.map(Ok::<bytes::Bytes, std::convert::Infallible>),
        );
        return sse_ok_response(body);
    }

    let transcoder =
        StreamTranscoder::new(provider, ingress, client_model.to_string(), response_id);
    let sse_events = Box::pin(sse_frame_stream(byte_stream));
    let output_stream = futures_util::stream::unfold(
        (
            sse_events,
            transcoder,
            Vec::<crate::protocol::canonical::CanonicalStreamEvent>::with_capacity(8),
            Vec::<bytes::Bytes>::with_capacity(8),
            PendingBytes::with_capacity(8),
            false,
        ),
        |(
            mut sse_stream,
            mut transcoder,
            mut decode_buffer,
            mut frame_chunks,
            mut pending,
            mut done,
        )| async move {
            loop {
                if let Some(chunk) = pending.pop_front() {
                    return Some((
                        chunk,
                        (
                            sse_stream,
                            transcoder,
                            decode_buffer,
                            frame_chunks,
                            pending,
                            done,
                        ),
                    ));
                }
                if done {
                    return None;
                }
                if let Some(frame) = sse_stream.as_mut().next().await {
                    transcoder.transcode_frame_into_bytes_with_decode_buffer(
                        &frame,
                        &mut decode_buffer,
                        &mut frame_chunks,
                    );
                    if let Some(chunk) = emit_from_byte_chunks(&mut frame_chunks, &mut pending) {
                        return Some((
                            chunk,
                            (
                                sse_stream,
                                transcoder,
                                decode_buffer,
                                frame_chunks,
                                pending,
                                done,
                            ),
                        ));
                    }
                } else {
                    done = true;
                }
            }
        },
    );

    let body = axum::body::Body::from_stream(
        output_stream.map(Ok::<bytes::Bytes, std::convert::Infallible>),
    );
    sse_ok_response(body)
}
