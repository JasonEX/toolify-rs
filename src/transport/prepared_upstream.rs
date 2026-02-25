use std::borrow::Cow;

use crate::config::UpstreamServiceConfig;
use crate::protocol::canonical::ProviderKind;
use rustc_hash::{FxHashMap, FxHashSet};

/// Precomputed upstream metadata used by hot request paths.
#[derive(Debug, Clone)]
pub struct PreparedUpstream {
    provider_kind: ProviderKind,
    openai_chat_url: String,
    openai_chat_url_parsed: Option<url::Url>,
    openai_chat_uri_parsed: Option<http::Uri>,
    responses_url: String,
    responses_url_parsed: Option<url::Url>,
    responses_uri_parsed: Option<http::Uri>,
    anthropic_messages_url: String,
    anthropic_messages_url_parsed: Option<url::Url>,
    anthropic_messages_uri_parsed: Option<http::Uri>,
    gemini_model_prefix: String,
    gemini_non_stream_urls: FxHashMap<String, String>,
    gemini_non_stream_urls_parsed: FxHashMap<String, url::Url>,
    gemini_non_stream_uris_parsed: FxHashMap<String, http::Uri>,
    gemini_stream_urls: FxHashMap<String, String>,
    gemini_stream_urls_parsed: FxHashMap<String, url::Url>,
    gemini_stream_uris_parsed: FxHashMap<String, http::Uri>,
    static_headers: http::HeaderMap,
    proxy_default: Option<String>,
    proxy_stream: Option<String>,
    proxy_non_stream: Option<String>,
}

impl PreparedUpstream {
    /// Build a prepared upstream cache from configuration.
    #[must_use]
    pub fn new(upstream: &UpstreamServiceConfig) -> Self {
        let base = upstream.base_url.trim_end_matches('/').to_string();
        let provider_kind = match upstream.provider.as_str() {
            "openai" => ProviderKind::OpenAi,
            "openai-responses" => ProviderKind::OpenAiResponses,
            "anthropic" => ProviderKind::Anthropic,
            "gemini" => ProviderKind::Gemini,
            "gemini-openai" => ProviderKind::GeminiOpenAi,
            _ => unreachable!("provider is validated at config load time"),
        };

        let static_headers = Self::build_provider_headers(upstream);
        let mut openai_chat_url = String::new();
        let mut openai_chat_url_parsed: Option<url::Url> = None;
        let mut openai_chat_uri_parsed: Option<http::Uri> = None;
        let mut responses_url = String::new();
        let mut responses_url_parsed: Option<url::Url> = None;
        let mut responses_uri_parsed: Option<http::Uri> = None;
        let mut anthropic_messages_url = String::new();
        let mut anthropic_messages_url_parsed: Option<url::Url> = None;
        let mut anthropic_messages_uri_parsed: Option<http::Uri> = None;
        let mut gemini_model_prefix = String::new();
        let mut gemini_non_stream_urls = FxHashMap::default();
        let mut gemini_non_stream_urls_parsed = FxHashMap::default();
        let mut gemini_non_stream_uris_parsed = FxHashMap::default();
        let mut gemini_stream_urls = FxHashMap::default();
        let mut gemini_stream_urls_parsed = FxHashMap::default();
        let mut gemini_stream_uris_parsed = FxHashMap::default();
        let proxy_default = normalize_proxy(upstream.proxy.as_deref());
        let proxy_stream = normalize_proxy(upstream.proxy_stream.as_deref());
        let proxy_non_stream = normalize_proxy(upstream.proxy_non_stream.as_deref());

        match provider_kind {
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => {
                openai_chat_url = format!("{base}/chat/completions");
                openai_chat_url_parsed = url::Url::parse(&openai_chat_url).ok();
                openai_chat_uri_parsed = openai_chat_url.parse().ok();
            }
            ProviderKind::OpenAiResponses => {
                responses_url = format!("{base}/responses");
                responses_url_parsed = url::Url::parse(&responses_url).ok();
                responses_uri_parsed = responses_url.parse().ok();
            }
            ProviderKind::Anthropic => {
                anthropic_messages_url = format!("{base}/messages");
                anthropic_messages_url_parsed = url::Url::parse(&anthropic_messages_url).ok();
                anthropic_messages_uri_parsed = anthropic_messages_url.parse().ok();
            }
            ProviderKind::Gemini => {
                gemini_model_prefix = format!("{base}/models/");
                let mut gemini_models = FxHashSet::default();
                for model_entry in &upstream.models {
                    let actual_model = model_entry
                        .split_once(':')
                        .map_or(model_entry.as_str(), |(_, real_model)| real_model)
                        .trim();
                    if !actual_model.is_empty() {
                        gemini_models.insert(actual_model.to_owned());
                    }
                }

                for model in gemini_models {
                    let non_stream_url = format!("{gemini_model_prefix}{model}:generateContent");
                    let stream_url = format!("{gemini_model_prefix}{model}:streamGenerateContent");

                    if let Ok(parsed) = url::Url::parse(&non_stream_url) {
                        gemini_non_stream_urls_parsed.insert(model.clone(), parsed);
                    }
                    if let Ok(parsed) = non_stream_url.parse() {
                        gemini_non_stream_uris_parsed.insert(model.clone(), parsed);
                    }
                    if let Ok(parsed) = url::Url::parse(&stream_url) {
                        gemini_stream_urls_parsed.insert(model.clone(), parsed);
                    }
                    if let Ok(parsed) = stream_url.parse() {
                        gemini_stream_uris_parsed.insert(model.clone(), parsed);
                    }

                    gemini_non_stream_urls.insert(model.clone(), non_stream_url);
                    gemini_stream_urls.insert(model, stream_url);
                }
            }
        }

        Self {
            provider_kind,
            openai_chat_url,
            openai_chat_url_parsed,
            openai_chat_uri_parsed,
            responses_url,
            responses_url_parsed,
            responses_uri_parsed,
            anthropic_messages_url,
            anthropic_messages_url_parsed,
            anthropic_messages_uri_parsed,
            gemini_model_prefix,
            gemini_non_stream_urls,
            gemini_non_stream_urls_parsed,
            gemini_non_stream_uris_parsed,
            gemini_stream_urls,
            gemini_stream_urls_parsed,
            gemini_stream_uris_parsed,
            static_headers,
            proxy_default,
            proxy_stream,
            proxy_non_stream,
        }
    }

    #[must_use]
    pub fn provider_kind(&self) -> ProviderKind {
        self.provider_kind
    }

    #[must_use]
    pub fn openai_chat_url_parsed(&self) -> Option<&url::Url> {
        self.openai_chat_url_parsed.as_ref()
    }

    #[must_use]
    pub fn openai_chat_uri_parsed(&self) -> Option<&http::Uri> {
        self.openai_chat_uri_parsed.as_ref()
    }

    #[must_use]
    pub fn responses_url_parsed(&self) -> Option<&url::Url> {
        self.responses_url_parsed.as_ref()
    }

    #[must_use]
    pub fn responses_uri_parsed(&self) -> Option<&http::Uri> {
        self.responses_uri_parsed.as_ref()
    }

    #[must_use]
    pub fn anthropic_messages_url_parsed(&self) -> Option<&url::Url> {
        self.anthropic_messages_url_parsed.as_ref()
    }

    #[must_use]
    pub fn anthropic_messages_uri_parsed(&self) -> Option<&http::Uri> {
        self.anthropic_messages_uri_parsed.as_ref()
    }

    /// Build the target request URL for this prepared upstream.
    #[must_use]
    pub fn request_url<'a>(&'a self, model: &str, stream: bool) -> Cow<'a, str> {
        match self.provider_kind {
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => {
                Cow::Borrowed(&self.openai_chat_url)
            }
            ProviderKind::OpenAiResponses => Cow::Borrowed(&self.responses_url),
            ProviderKind::Anthropic => Cow::Borrowed(&self.anthropic_messages_url),
            ProviderKind::Gemini => {
                if stream {
                    if let Some(url) = self.gemini_stream_urls.get(model) {
                        Cow::Borrowed(url)
                    } else {
                        Cow::Owned(format!(
                            "{}{}:streamGenerateContent",
                            self.gemini_model_prefix, model
                        ))
                    }
                } else if let Some(url) = self.gemini_non_stream_urls.get(model) {
                    Cow::Borrowed(url)
                } else {
                    Cow::Owned(format!(
                        "{}{}:generateContent",
                        self.gemini_model_prefix, model
                    ))
                }
            }
        }
    }

    /// Return a pre-parsed static URL when the endpoint path does not depend on model/action.
    #[must_use]
    pub fn static_url(&self) -> Option<&url::Url> {
        match self.provider_kind {
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => self.openai_chat_url_parsed(),
            ProviderKind::OpenAiResponses => self.responses_url_parsed(),
            ProviderKind::Anthropic => self.anthropic_messages_url_parsed(),
            ProviderKind::Gemini => None,
        }
    }

    /// Return a pre-parsed request URL when available for the given model/action.
    #[must_use]
    pub fn request_url_parsed(&self, model: &str, stream: bool) -> Option<&url::Url> {
        match self.provider_kind {
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => self.openai_chat_url_parsed(),
            ProviderKind::OpenAiResponses => self.responses_url_parsed(),
            ProviderKind::Anthropic => self.anthropic_messages_url_parsed(),
            ProviderKind::Gemini => {
                if stream {
                    self.gemini_stream_urls_parsed.get(model)
                } else {
                    self.gemini_non_stream_urls_parsed.get(model)
                }
            }
        }
    }

    /// Return a pre-parsed static URI when the endpoint path does not depend on model/action.
    #[must_use]
    pub fn static_uri(&self) -> Option<&http::Uri> {
        match self.provider_kind {
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => self.openai_chat_uri_parsed(),
            ProviderKind::OpenAiResponses => self.responses_uri_parsed(),
            ProviderKind::Anthropic => self.anthropic_messages_uri_parsed(),
            ProviderKind::Gemini => None,
        }
    }

    /// Return a pre-parsed request URI when available for the given model/action.
    #[must_use]
    pub fn request_uri_parsed(&self, model: &str, stream: bool) -> Option<&http::Uri> {
        match self.provider_kind {
            ProviderKind::OpenAi | ProviderKind::GeminiOpenAi => self.openai_chat_uri_parsed(),
            ProviderKind::OpenAiResponses => self.responses_uri_parsed(),
            ProviderKind::Anthropic => self.anthropic_messages_uri_parsed(),
            ProviderKind::Gemini => {
                if stream {
                    self.gemini_stream_uris_parsed.get(model)
                } else {
                    self.gemini_non_stream_uris_parsed.get(model)
                }
            }
        }
    }

    #[must_use]
    pub(crate) fn static_headers(&self) -> &http::HeaderMap {
        &self.static_headers
    }

    #[must_use]
    pub fn proxy_for(&self, stream: bool) -> Option<&str> {
        if stream {
            self.proxy_stream
                .as_deref()
                .or(self.proxy_default.as_deref())
        } else {
            self.proxy_non_stream
                .as_deref()
                .or(self.proxy_default.as_deref())
        }
    }

    fn build_provider_headers(upstream: &UpstreamServiceConfig) -> http::HeaderMap {
        let key = upstream.api_key.as_str();

        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        match upstream.provider.as_str() {
            "openai" | "openai-responses" | "gemini-openai" => {
                if let Ok(val) = http::HeaderValue::from_str(&format!("Bearer {key}")) {
                    headers.insert(http::header::AUTHORIZATION, val);
                }
            }
            "anthropic" => {
                if let Ok(val) = http::HeaderValue::from_str(key) {
                    headers.insert("x-api-key", val);
                }
                let version = upstream.api_version.as_deref().unwrap_or("2023-06-01");
                if let Ok(val) = http::HeaderValue::from_str(version) {
                    headers.insert("anthropic-version", val);
                }
            }
            "gemini" => {
                if let Ok(val) = http::HeaderValue::from_str(key) {
                    headers.insert("x-goog-api-key", val);
                }
            }
            _ => unreachable!("provider is validated at config load time"),
        }

        headers
    }
}

fn normalize_proxy(proxy: Option<&str>) -> Option<String> {
    proxy.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

/// Build upstream URL using startup-prepared endpoint templates.
#[must_use]
pub fn build_upstream_url_prepared<'a>(
    prepared: &'a PreparedUpstream,
    model: &str,
    stream: bool,
) -> Cow<'a, str> {
    prepared.request_url(model, stream)
}

/// Return a pre-parsed static upstream URL when the provider endpoint does not
/// depend on model/action-specific path segments.
#[must_use]
pub fn static_parsed_upstream_url<'a>(
    prepared: &'a PreparedUpstream,
    model: &str,
    stream: bool,
) -> Option<&'a url::Url> {
    prepared.request_url_parsed(model, stream)
}

/// Return a pre-parsed static upstream URI when the provider endpoint does not
/// depend on model/action-specific path segments.
#[must_use]
pub fn static_parsed_upstream_uri<'a>(
    prepared: &'a PreparedUpstream,
    model: &str,
    stream: bool,
) -> Option<&'a http::Uri> {
    prepared.request_uri_parsed(model, stream)
}

/// Build provider headers while reusing startup-precomputed static headers when possible.
#[must_use]
pub fn build_provider_headers_prepared(prepared: &PreparedUpstream) -> &http::HeaderMap {
    prepared.static_headers()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_upstream(provider: &str) -> UpstreamServiceConfig {
        UpstreamServiceConfig {
            name: "test".to_string(),
            provider: provider.to_string(),
            base_url: "https://api.example.com/v1".to_string(),
            api_key: "sk-test-key".to_string(),
            models: vec![],
            description: String::new(),
            is_default: false,
            fc_mode: crate::config::FcMode::default(),
            api_version: None,
            proxy: None,
            proxy_stream: None,
            proxy_non_stream: None,
        }
    }

    #[test]
    fn test_build_url_openai() {
        let upstream = make_upstream("openai");
        let prepared = PreparedUpstream::new(&upstream);
        let url = prepared.request_url("gpt-4", false);
        assert_eq!(url.as_ref(), "https://api.example.com/v1/chat/completions");
    }

    #[test]
    fn test_build_url_openai_responses() {
        let upstream = make_upstream("openai-responses");
        let prepared = PreparedUpstream::new(&upstream);
        let url = prepared.request_url("gpt-4", false);
        assert_eq!(url.as_ref(), "https://api.example.com/v1/responses");
    }

    #[test]
    fn test_build_url_anthropic() {
        let upstream = make_upstream("anthropic");
        let prepared = PreparedUpstream::new(&upstream);
        let url = prepared.request_url("claude-3", false);
        assert_eq!(url.as_ref(), "https://api.example.com/v1/messages");
    }

    #[test]
    fn test_build_url_gemini_non_stream() {
        let upstream = make_upstream("gemini");
        let prepared = PreparedUpstream::new(&upstream);
        let url = prepared.request_url("gemini-pro", false);
        assert_eq!(
            url.as_ref(),
            "https://api.example.com/v1/models/gemini-pro:generateContent"
        );
    }

    #[test]
    fn test_build_url_gemini_stream() {
        let upstream = make_upstream("gemini");
        let prepared = PreparedUpstream::new(&upstream);
        let url = prepared.request_url("gemini-pro", true);
        assert_eq!(
            url.as_ref(),
            "https://api.example.com/v1/models/gemini-pro:streamGenerateContent"
        );
    }

    #[test]
    fn test_gemini_preparsed_endpoints_for_configured_models() {
        let mut upstream = make_upstream("gemini");
        upstream.models = vec!["smart:gemini-pro".to_string(), "gemini-flash".to_string()];
        let prepared = PreparedUpstream::new(&upstream);

        let non_stream = static_parsed_upstream_url(&prepared, "gemini-pro", false)
            .expect("configured gemini model should have parsed URL");
        assert_eq!(
            non_stream.as_str(),
            "https://api.example.com/v1/models/gemini-pro:generateContent"
        );
        let stream = static_parsed_upstream_url(&prepared, "gemini-pro", true)
            .expect("configured gemini model should have parsed stream URL");
        assert_eq!(
            stream.as_str(),
            "https://api.example.com/v1/models/gemini-pro:streamGenerateContent"
        );

        let non_stream_uri = static_parsed_upstream_uri(&prepared, "gemini-pro", false)
            .expect("configured gemini model should have parsed URI");
        assert_eq!(
            non_stream_uri.to_string(),
            "https://api.example.com/v1/models/gemini-pro:generateContent"
        );
        let stream_uri = static_parsed_upstream_uri(&prepared, "gemini-pro", true)
            .expect("configured gemini model should have parsed stream URI");
        assert_eq!(
            stream_uri.to_string(),
            "https://api.example.com/v1/models/gemini-pro:streamGenerateContent"
        );
    }

    #[test]
    fn test_gemini_preparsed_endpoints_missing_model() {
        let mut upstream = make_upstream("gemini");
        upstream.models = vec!["gemini-pro".to_string()];
        let prepared = PreparedUpstream::new(&upstream);
        assert!(static_parsed_upstream_url(&prepared, "gemini-other", false).is_none());
        assert!(static_parsed_upstream_uri(&prepared, "gemini-other", true).is_none());
    }

    #[test]
    fn test_build_url_gemini_openai() {
        let upstream = make_upstream("gemini-openai");
        let prepared = PreparedUpstream::new(&upstream);
        let url = prepared.request_url("gemini-pro", false);
        assert_eq!(url.as_ref(), "https://api.example.com/v1/chat/completions");
    }

    #[test]
    fn test_build_url_trailing_slash() {
        let mut upstream = make_upstream("openai");
        upstream.base_url = "https://api.example.com/v1/".to_string();
        let prepared = PreparedUpstream::new(&upstream);
        let url = prepared.request_url("gpt-4", false);
        assert_eq!(url.as_ref(), "https://api.example.com/v1/chat/completions");
    }

    #[test]
    fn test_headers_openai() {
        let upstream = make_upstream("openai");
        let prepared = PreparedUpstream::new(&upstream);
        let headers = prepared.static_headers();
        assert_eq!(
            headers.get(http::header::AUTHORIZATION).unwrap(),
            "Bearer sk-test-key"
        );
        assert_eq!(
            headers.get(http::header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
    }

    #[test]
    fn test_headers_anthropic() {
        let upstream = make_upstream("anthropic");
        let prepared = PreparedUpstream::new(&upstream);
        let headers = prepared.static_headers();
        assert_eq!(headers.get("x-api-key").unwrap(), "sk-test-key");
        assert_eq!(headers.get("anthropic-version").unwrap(), "2023-06-01");
        assert!(headers.get(http::header::AUTHORIZATION).is_none());
    }

    #[test]
    fn test_headers_anthropic_custom_version() {
        let mut upstream = make_upstream("anthropic");
        upstream.api_version = Some("2024-01-01".to_string());
        let prepared = PreparedUpstream::new(&upstream);
        let headers = prepared.static_headers();
        assert_eq!(headers.get("anthropic-version").unwrap(), "2024-01-01");
    }

    #[test]
    fn test_headers_gemini() {
        let upstream = make_upstream("gemini");
        let prepared = PreparedUpstream::new(&upstream);
        let headers = prepared.static_headers();
        assert_eq!(headers.get("x-goog-api-key").unwrap(), "sk-test-key");
        assert!(headers.get(http::header::AUTHORIZATION).is_none());
    }

    #[test]
    fn test_prepared_upstream_parsed_static_urls() {
        let openai = PreparedUpstream::new(&make_upstream("openai"));
        assert!(openai.openai_chat_url_parsed().is_some());
        assert!(openai.responses_url_parsed().is_none());
        assert!(openai.anthropic_messages_url_parsed().is_none());

        let responses = PreparedUpstream::new(&make_upstream("openai-responses"));
        assert!(responses.openai_chat_url_parsed().is_none());
        assert!(responses.responses_url_parsed().is_some());
        assert!(responses.anthropic_messages_url_parsed().is_none());

        let anthropic = PreparedUpstream::new(&make_upstream("anthropic"));
        assert!(anthropic.openai_chat_url_parsed().is_none());
        assert!(anthropic.responses_url_parsed().is_none());
        assert!(anthropic.anthropic_messages_url_parsed().is_some());
    }

    #[test]
    fn test_proxy_selection() {
        let mut upstream = make_upstream("openai");
        upstream.proxy = Some("http://default.proxy:8080".to_string());
        upstream.proxy_stream = Some("http://stream.proxy:8080".to_string());
        upstream.proxy_non_stream = Some("http://nonstream.proxy:8080".to_string());
        let prepared = PreparedUpstream::new(&upstream);
        assert_eq!(
            prepared.proxy_for(false),
            Some("http://nonstream.proxy:8080")
        );
        assert_eq!(prepared.proxy_for(true), Some("http://stream.proxy:8080"));
    }

    #[test]
    fn test_proxy_default_fallback() {
        let mut upstream = make_upstream("openai");
        upstream.proxy = Some("http://default.proxy:8080".to_string());
        let prepared = PreparedUpstream::new(&upstream);
        assert_eq!(prepared.proxy_for(false), Some("http://default.proxy:8080"));
        assert_eq!(prepared.proxy_for(true), Some("http://default.proxy:8080"));
    }
}
