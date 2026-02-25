use std::sync::Arc;
use std::sync::{Once, OnceLock};
use std::time::Duration;

use http_body_util::Full;
use hyper::body::Incoming;
use hyper_rustls::HttpsConnector;
use hyper_rustls::HttpsConnectorBuilder;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client as HyperClient;
use hyper_util::rt::{TokioExecutor, TokioTimer};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::config::ServerConfig;
use crate::error::CanonicalError;

use super::retry_policy::{
    retry_delay, retry_transport_delay, should_retry_transport_message,
    should_retry_upstream_status, PARSED_ENDPOINT_CACHE_MAX_ENTRIES, RETRY_MAX_ATTEMPTS,
};

static RUSTLS_PROVIDER_INIT: Once = Once::new();
const REQWEST_PROXY_CLIENT_CACHE_MAX_ENTRIES: usize = 64;
const H2_KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(60);
const H2_KEEP_ALIVE_TIMEOUT: Duration = Duration::from_secs(10);
const H2_INITIAL_WINDOW_SIZE: u32 = 1_572_864;

type HyperPassthroughHttpsClient = HyperClient<HttpsConnector<HttpConnector>, Full<bytes::Bytes>>;
type HyperPassthroughHttpClient = HyperClient<HttpConnector, Full<bytes::Bytes>>;

fn build_reqwest_client(
    pool_max_idle_per_host: usize,
    pool_idle_timeout: Option<Duration>,
    timeout: Duration,
    use_env_proxy: bool,
    proxy_url: Option<&str>,
) -> Result<reqwest::Client, CanonicalError> {
    let mut builder = reqwest::Client::builder()
        .pool_max_idle_per_host(pool_max_idle_per_host)
        .pool_idle_timeout(pool_idle_timeout)
        .tcp_nodelay(true)
        .connect_timeout(Duration::from_secs(5))
        .redirect(reqwest::redirect::Policy::none())
        .timeout(timeout);

    if let Some(proxy_url) = proxy_url {
        let proxy = reqwest::Proxy::all(proxy_url)
            .map_err(|err| CanonicalError::Transport(format!("Invalid proxy URL: {err}")))?;
        builder = builder.no_proxy().proxy(proxy);
    } else if !use_env_proxy {
        builder = builder.no_proxy();
    }

    builder
        .build()
        .map_err(|err| CanonicalError::Transport(format!("Failed to build HTTP client: {err}")))
}

/// HTTP transport client for sending requests to upstream providers.
pub struct HttpTransport {
    base_client: OnceLock<Arc<reqwest::Client>>,
    preconfigured_proxy_clients: FxHashMap<String, Arc<reqwest::Client>>,
    dynamic_proxy_clients: RwLock<FxHashMap<String, Arc<reqwest::Client>>>,
    parsed_url_cache: RwLock<FxHashMap<String, Arc<url::Url>>>,
    parsed_uri_cache: RwLock<FxHashMap<String, Arc<http::Uri>>>,
    reqwest_pool_max_idle_per_host: usize,
    reqwest_pool_idle_timeout: Option<Duration>,
    reqwest_timeout: Duration,
    reqwest_use_env_proxy: bool,
    hyper_passthrough_enabled: bool,
    hyper_passthrough_force_h2c_upstream: bool,
    hyper_passthrough_pool_max_idle_per_host: usize,
    hyper_passthrough_pool_idle_timeout: Option<Duration>,
    hyper_passthrough_https_client: OnceLock<HyperPassthroughHttpsClient>,
    hyper_passthrough_http_client: OnceLock<HyperPassthroughHttpClient>,
    hyper_passthrough_h2c_client: OnceLock<HyperPassthroughHttpClient>,
}

impl HttpTransport {
    #[inline]
    fn effective_pool_max_idle_per_host(config: &ServerConfig, upstream_count: usize) -> usize {
        let configured = config.http_pool_max_idle_per_host.max(1);
        let Some(worker_threads) = config.runtime_worker_threads else {
            return configured;
        };
        if worker_threads != 1 {
            return configured;
        }

        // Single-worker runtimes target low RSS; cap idle pool budget by upstream fanout.
        let dynamic_cap = match upstream_count {
            0..=2 => 16,
            3..=4 => 12,
            _ => 8,
        };
        configured.min(dynamic_cap)
    }

    /// Create a new transport with connection pooling and timeouts from the given server config.
    #[must_use]
    pub fn new(config: &ServerConfig) -> Self {
        Self::new_with_upstream_count(config, 1)
    }

    /// Create a new transport with upstream-count-aware pool budgeting.
    #[must_use]
    pub fn new_with_upstream_count(config: &ServerConfig, upstream_count: usize) -> Self {
        Self::new_with_upstream_count_and_proxies(
            config,
            upstream_count,
            std::iter::empty::<&str>(),
        )
    }

    /// Create a new transport with upstream-count-aware pool budgeting and
    /// eagerly-built per-proxy reqwest clients.
    #[must_use]
    pub fn new_with_upstream_count_and_proxies<I, S>(
        config: &ServerConfig,
        upstream_count: usize,
        proxy_urls: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        RUSTLS_PROVIDER_INIT.call_once(|| {
            let _ = rustls::crypto::ring::default_provider().install_default();
        });

        let pool_idle_timeout = if config.http_pool_idle_timeout_secs == 0 {
            None
        } else {
            Some(Duration::from_secs(config.http_pool_idle_timeout_secs))
        };

        let reqwest_timeout = Duration::from_secs(config.timeout);
        let effective_pool_max_idle_per_host =
            Self::effective_pool_max_idle_per_host(config, upstream_count);
        let reqwest_use_env_proxy = config.http_use_env_proxy;
        let preconfigured_proxy_clients = Self::build_preconfigured_proxy_clients(
            proxy_urls,
            effective_pool_max_idle_per_host,
            pool_idle_timeout,
            reqwest_timeout,
            reqwest_use_env_proxy,
        );
        Self {
            base_client: OnceLock::new(),
            preconfigured_proxy_clients,
            dynamic_proxy_clients: RwLock::new(FxHashMap::default()),
            parsed_url_cache: RwLock::new(FxHashMap::default()),
            parsed_uri_cache: RwLock::new(FxHashMap::default()),
            reqwest_pool_max_idle_per_host: effective_pool_max_idle_per_host,
            reqwest_pool_idle_timeout: pool_idle_timeout,
            reqwest_timeout,
            reqwest_use_env_proxy,
            hyper_passthrough_enabled: !reqwest_use_env_proxy,
            hyper_passthrough_force_h2c_upstream: config.http_force_h2c_upstream,
            hyper_passthrough_pool_max_idle_per_host: effective_pool_max_idle_per_host,
            hyper_passthrough_pool_idle_timeout: pool_idle_timeout,
            hyper_passthrough_https_client: OnceLock::new(),
            hyper_passthrough_http_client: OnceLock::new(),
            hyper_passthrough_h2c_client: OnceLock::new(),
        }
    }

    fn build_base_reqwest_client(&self) -> Arc<reqwest::Client> {
        match build_reqwest_client(
            self.reqwest_pool_max_idle_per_host,
            self.reqwest_pool_idle_timeout,
            self.reqwest_timeout,
            self.reqwest_use_env_proxy,
            None,
        ) {
            Ok(client) => Arc::new(client),
            Err(err) => {
                tracing::error!(error = %err, "failed to build configured reqwest client, falling back to default client");
                Arc::new(reqwest::Client::new())
            }
        }
    }

    fn base_reqwest_client(&self) -> Arc<reqwest::Client> {
        if let Some(existing) = self.base_client.get() {
            return existing.clone();
        }

        let built = self.build_base_reqwest_client();
        let _ = self.base_client.set(built.clone());
        self.base_client.get().cloned().unwrap_or(built)
    }

    fn build_preconfigured_proxy_clients<I, S>(
        proxy_urls: I,
        pool_max_idle_per_host: usize,
        pool_idle_timeout: Option<Duration>,
        timeout: Duration,
        use_env_proxy: bool,
    ) -> FxHashMap<String, Arc<reqwest::Client>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut clients = FxHashMap::default();
        for proxy_url in proxy_urls {
            let proxy_url = proxy_url.as_ref();
            if proxy_url.is_empty() || clients.contains_key(proxy_url) {
                continue;
            }
            match build_reqwest_client(
                pool_max_idle_per_host,
                pool_idle_timeout,
                timeout,
                use_env_proxy,
                Some(proxy_url),
            ) {
                Ok(client) => {
                    clients.insert(proxy_url.to_owned(), Arc::new(client));
                }
                Err(err) => {
                    tracing::error!(
                        proxy_url,
                        error = %err,
                        "failed to prebuild proxy HTTP client, falling back to lazy build"
                    );
                }
            }
        }
        clients
    }

    #[must_use]
    pub fn hyper_passthrough_enabled(&self) -> bool {
        self.hyper_passthrough_enabled
    }

    #[must_use]
    pub fn hyper_passthrough_enabled_for(&self, proxy_url: Option<&str>) -> bool {
        self.hyper_passthrough_enabled && proxy_url.is_none()
    }

    #[must_use]
    pub fn preconfigured_proxy_client(&self, proxy_url: Option<&str>) -> Option<&reqwest::Client> {
        let proxy_url = proxy_url?;
        self.preconfigured_proxy_clients
            .get(proxy_url)
            .map(std::convert::AsRef::as_ref)
    }

    fn hyper_passthrough_https_client(&self) -> Option<&HyperPassthroughHttpsClient> {
        if !self.hyper_passthrough_enabled {
            return None;
        }

        Some(self.hyper_passthrough_https_client.get_or_init(|| {
            let mut connector = HttpConnector::new();
            connector.enforce_http(false);
            connector.set_nodelay(true);
            connector.set_connect_timeout(Some(Duration::from_secs(5)));
            let https = HttpsConnectorBuilder::new()
                .with_webpki_roots()
                .https_or_http()
                .enable_http1()
                .enable_http2()
                .wrap_connector(connector);
            let mut builder = HyperClient::builder(TokioExecutor::new());
            builder.pool_max_idle_per_host(self.hyper_passthrough_pool_max_idle_per_host);
            builder.pool_idle_timeout(self.hyper_passthrough_pool_idle_timeout);
            builder.pool_timer(TokioTimer::new());
            builder.timer(TokioTimer::new());
            builder.http2_adaptive_window(true);
            builder.http2_keep_alive_interval(H2_KEEP_ALIVE_INTERVAL);
            builder.http2_keep_alive_timeout(H2_KEEP_ALIVE_TIMEOUT);
            builder.http2_keep_alive_while_idle(true);
            builder.build(https)
        }))
    }

    fn hyper_passthrough_http_client(&self) -> Option<&HyperPassthroughHttpClient> {
        if !self.hyper_passthrough_enabled {
            return None;
        }

        Some(self.hyper_passthrough_http_client.get_or_init(|| {
            let mut connector = HttpConnector::new();
            connector.enforce_http(true);
            connector.set_nodelay(true);
            connector.set_connect_timeout(Some(Duration::from_secs(5)));
            let mut builder = HyperClient::builder(TokioExecutor::new());
            builder.pool_max_idle_per_host(self.hyper_passthrough_pool_max_idle_per_host);
            builder.pool_idle_timeout(self.hyper_passthrough_pool_idle_timeout);
            builder.pool_timer(TokioTimer::new());
            builder.build(connector)
        }))
    }

    fn hyper_passthrough_h2c_client(&self) -> Option<&HyperPassthroughHttpClient> {
        if !self.hyper_passthrough_enabled {
            return None;
        }

        Some(self.hyper_passthrough_h2c_client.get_or_init(|| {
            let mut connector = HttpConnector::new();
            connector.enforce_http(true);
            connector.set_nodelay(true);
            connector.set_connect_timeout(Some(Duration::from_secs(5)));
            let mut builder = HyperClient::builder(TokioExecutor::new());
            builder.pool_max_idle_per_host(self.hyper_passthrough_pool_max_idle_per_host);
            builder.pool_idle_timeout(self.hyper_passthrough_pool_idle_timeout);
            builder.pool_timer(TokioTimer::new());
            builder.timer(TokioTimer::new());
            builder.http2_only(true);
            builder.http2_adaptive_window(false);
            builder.http2_initial_connection_window_size(H2_INITIAL_WINDOW_SIZE);
            builder.http2_initial_stream_window_size(H2_INITIAL_WINDOW_SIZE);
            builder.http2_keep_alive_interval(H2_KEEP_ALIVE_INTERVAL);
            builder.http2_keep_alive_timeout(H2_KEEP_ALIVE_TIMEOUT);
            builder.http2_keep_alive_while_idle(true);
            builder.build(connector)
        }))
    }

    fn reqwest_client_for_proxy(
        &self,
        proxy_url: Option<&str>,
    ) -> Result<Arc<reqwest::Client>, CanonicalError> {
        let Some(proxy_url) = proxy_url else {
            return Ok(self.base_reqwest_client());
        };

        if let Some(existing) = self.preconfigured_proxy_clients.get(proxy_url) {
            return Ok(existing.clone());
        }

        if let Some(existing) = self.dynamic_proxy_clients.read().get(proxy_url) {
            return Ok(existing.clone());
        }

        let client = build_reqwest_client(
            self.reqwest_pool_max_idle_per_host,
            self.reqwest_pool_idle_timeout,
            self.reqwest_timeout,
            self.reqwest_use_env_proxy,
            Some(proxy_url),
        )
        .map(Arc::new)?;

        let mut cache = self.dynamic_proxy_clients.write();
        if let Some(existing) = cache.get(proxy_url) {
            return Ok(existing.clone());
        }
        if cache.len() >= REQWEST_PROXY_CLIENT_CACHE_MAX_ENTRIES {
            cache.clear();
        }
        cache.insert(proxy_url.to_string(), client.clone());
        Ok(client)
    }

    fn parsed_url(&self, url: &str) -> Result<Arc<url::Url>, CanonicalError> {
        if let Some(cached) = self.parsed_url_cache.read().get(url) {
            return Ok(cached.clone());
        }

        let parsed = url::Url::parse(url)
            .map_err(|e| CanonicalError::Transport(format!("Invalid upstream URL: {e}")))?;

        let mut cache = self.parsed_url_cache.write();
        if let Some(existing) = cache.get(url) {
            return Ok(existing.clone());
        }
        if cache.len() >= PARSED_ENDPOINT_CACHE_MAX_ENTRIES {
            cache.clear();
        }
        let parsed = Arc::new(parsed);
        cache.insert(url.to_string(), parsed.clone());
        Ok(parsed)
    }

    fn parsed_uri(&self, url: &str) -> Result<Arc<http::Uri>, CanonicalError> {
        if let Some(cached) = self.parsed_uri_cache.read().get(url) {
            return Ok(cached.clone());
        }

        let parsed: http::Uri = url
            .parse()
            .map_err(|e| CanonicalError::Transport(format!("Invalid upstream URI: {e}")))?;

        let mut cache = self.parsed_uri_cache.write();
        if let Some(existing) = cache.get(url) {
            return Ok(existing.clone());
        }
        if cache.len() >= PARSED_ENDPOINT_CACHE_MAX_ENTRIES {
            cache.clear();
        }
        let parsed = Arc::new(parsed);
        cache.insert(url.to_string(), parsed.clone());
        Ok(parsed)
    }

    /// Send a non-streaming request to an upstream provider.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::Transport`] when URL parsing fails, request
    /// execution fails, or retries are exhausted.
    pub async fn send_request(
        &self,
        url: &str,
        method: http::Method,
        headers: &http::HeaderMap,
        body: bytes::Bytes,
        proxy_url: Option<&str>,
    ) -> Result<reqwest::Response, CanonicalError> {
        self.send_request_with_client(url, method, headers, body, proxy_url, None)
            .await
    }

    /// Send a non-streaming request to an upstream provider using an optional
    /// pre-selected proxy client.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::Transport`] when URL parsing fails, request
    /// execution fails, or retries are exhausted.
    pub async fn send_request_with_client(
        &self,
        url: &str,
        method: http::Method,
        headers: &http::HeaderMap,
        body: bytes::Bytes,
        proxy_url: Option<&str>,
        preconfigured_proxy_client: Option<&reqwest::Client>,
    ) -> Result<reqwest::Response, CanonicalError> {
        let parsed_url = self.parsed_url(url)?;
        self.send_request_url_with_client(
            parsed_url.as_ref(),
            method,
            headers,
            body,
            proxy_url,
            preconfigured_proxy_client,
        )
        .await
    }

    /// Send a non-streaming request using a pre-parsed upstream URL.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::Transport`] when request execution fails or
    /// retries are exhausted.
    pub async fn send_request_url(
        &self,
        url: &url::Url,
        method: http::Method,
        headers: &http::HeaderMap,
        body: bytes::Bytes,
        proxy_url: Option<&str>,
    ) -> Result<reqwest::Response, CanonicalError> {
        self.send_request_url_with_client(url, method, headers, body, proxy_url, None)
            .await
    }

    /// Send a non-streaming request using a pre-parsed upstream URL and optional
    /// pre-selected proxy client.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::Transport`] when request execution fails or
    /// retries are exhausted.
    pub async fn send_request_url_with_client(
        &self,
        url: &url::Url,
        method: http::Method,
        headers: &http::HeaderMap,
        body: bytes::Bytes,
        proxy_url: Option<&str>,
        preconfigured_proxy_client: Option<&reqwest::Client>,
    ) -> Result<reqwest::Response, CanonicalError> {
        let dynamic_client = if preconfigured_proxy_client.is_none() {
            Some(self.reqwest_client_for_proxy(proxy_url)?)
        } else {
            None
        };
        let client = preconfigured_proxy_client
            .or(dynamic_client.as_deref())
            .ok_or_else(|| {
                CanonicalError::Transport("No HTTP client available for upstream request".into())
            })?;
        let mut attempt = 0;
        loop {
            let mut request = reqwest::Request::new(method.clone(), url.clone());
            *request.headers_mut() = headers.clone();
            *request.body_mut() = Some(reqwest::Body::from(body.clone()));

            match client.execute(request).await {
                Ok(response) => {
                    if attempt < RETRY_MAX_ATTEMPTS
                        && should_retry_upstream_status(response.status())
                    {
                        let delay = retry_delay(response.headers(), attempt);
                        tracing::debug!(
                            status = response.status().as_u16(),
                            retry_attempt = attempt + 1,
                            delay_ms = delay.as_millis(),
                            "retrying upstream request after retriable status"
                        );
                        drop(response);
                        tokio::time::sleep(delay).await;
                        attempt += 1;
                        continue;
                    }
                    return Ok(response);
                }
                Err(err) => {
                    let message = err.to_string();
                    if attempt >= RETRY_MAX_ATTEMPTS || !should_retry_transport_message(&message) {
                        return Err(CanonicalError::Transport(message));
                    }

                    let delay = retry_transport_delay(&message, attempt);
                    tracing::debug!(
                        retry_attempt = attempt + 1,
                        delay_ms = delay.as_millis(),
                        error = %message,
                        "retrying upstream request after transport error"
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
            }
        }
    }

    /// Send a streaming request to an upstream provider, returning the response for stream reading.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::Transport`] when URL parsing fails, request
    /// execution fails, or retries are exhausted.
    pub async fn send_stream(
        &self,
        url: &str,
        method: http::Method,
        headers: &http::HeaderMap,
        body: bytes::Bytes,
        proxy_url: Option<&str>,
    ) -> Result<reqwest::Response, CanonicalError> {
        self.send_stream_with_client(url, method, headers, body, proxy_url, None)
            .await
    }

    /// Send a streaming request to an upstream provider with optional
    /// pre-selected proxy client.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::Transport`] when URL parsing fails, request
    /// execution fails, or retries are exhausted.
    pub async fn send_stream_with_client(
        &self,
        url: &str,
        method: http::Method,
        headers: &http::HeaderMap,
        body: bytes::Bytes,
        proxy_url: Option<&str>,
        preconfigured_proxy_client: Option<&reqwest::Client>,
    ) -> Result<reqwest::Response, CanonicalError> {
        // Same as send_request; the caller reads the response body as a stream.
        self.send_request_with_client(
            url,
            method,
            headers,
            body,
            proxy_url,
            preconfigured_proxy_client,
        )
        .await
    }

    /// Send a streaming request using a pre-parsed upstream URL.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::Transport`] when request execution fails or
    /// retries are exhausted.
    pub async fn send_stream_url(
        &self,
        url: &url::Url,
        method: http::Method,
        headers: &http::HeaderMap,
        body: bytes::Bytes,
        proxy_url: Option<&str>,
    ) -> Result<reqwest::Response, CanonicalError> {
        self.send_stream_url_with_client(url, method, headers, body, proxy_url, None)
            .await
    }

    /// Send a streaming request using a pre-parsed upstream URL and optional
    /// pre-selected proxy client.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::Transport`] when request execution fails or
    /// retries are exhausted.
    pub async fn send_stream_url_with_client(
        &self,
        url: &url::Url,
        method: http::Method,
        headers: &http::HeaderMap,
        body: bytes::Bytes,
        proxy_url: Option<&str>,
        preconfigured_proxy_client: Option<&reqwest::Client>,
    ) -> Result<reqwest::Response, CanonicalError> {
        self.send_request_url_with_client(
            url,
            method,
            headers,
            body,
            proxy_url,
            preconfigured_proxy_client,
        )
        .await
    }

    /// Send a request using the hyper passthrough client and a pre-parsed URI.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::Transport`] when passthrough is disabled,
    /// request execution fails, or retries are exhausted.
    pub async fn send_request_uri(
        &self,
        uri: &http::Uri,
        method: http::Method,
        headers: &http::HeaderMap,
        body: bytes::Bytes,
    ) -> Result<http::Response<Incoming>, CanonicalError> {
        enum HyperClientRef<'a> {
            Http(&'a HyperPassthroughHttpClient),
            Https(&'a HyperPassthroughHttpsClient),
        }

        let client = if uri.scheme_str() == Some("http") {
            let http_client = if self.hyper_passthrough_force_h2c_upstream {
                self.hyper_passthrough_h2c_client()
            } else {
                self.hyper_passthrough_http_client()
            };
            let Some(client) = http_client else {
                return Err(CanonicalError::Transport(
                    "Hyper passthrough client is disabled".to_string(),
                ));
            };
            HyperClientRef::Http(client)
        } else {
            let Some(client) = self.hyper_passthrough_https_client() else {
                return Err(CanonicalError::Transport(
                    "Hyper passthrough client is disabled".to_string(),
                ));
            };
            HyperClientRef::Https(client)
        };

        let mut attempt = 0;
        loop {
            let mut request = http::Request::new(Full::new(body.clone()));
            *request.method_mut() = method.clone();
            *request.uri_mut() = uri.clone();
            *request.headers_mut() = headers.clone();

            let result = match client {
                HyperClientRef::Http(client) => client.request(request).await,
                HyperClientRef::Https(client) => client.request(request).await,
            };

            match result {
                Ok(response) => {
                    if attempt < RETRY_MAX_ATTEMPTS
                        && should_retry_upstream_status(response.status())
                    {
                        let delay = retry_delay(response.headers(), attempt);
                        tracing::debug!(
                            status = response.status().as_u16(),
                            retry_attempt = attempt + 1,
                            delay_ms = delay.as_millis(),
                            "retrying upstream hyper request after retriable status"
                        );
                        drop(response);
                        tokio::time::sleep(delay).await;
                        attempt += 1;
                        continue;
                    }
                    return Ok(response);
                }
                Err(err) => {
                    let message = err.to_string();
                    if attempt >= RETRY_MAX_ATTEMPTS || !should_retry_transport_message(&message) {
                        return Err(CanonicalError::Transport(message));
                    }

                    let delay = retry_transport_delay(&message, attempt);
                    tracing::debug!(
                        retry_attempt = attempt + 1,
                        delay_ms = delay.as_millis(),
                        error = %message,
                        "retrying upstream hyper request after transport error"
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
            }
        }
    }

    /// Send a streaming request using the hyper passthrough client and a pre-parsed URI.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::Transport`] when passthrough is disabled,
    /// request execution fails, or retries are exhausted.
    pub async fn send_stream_uri(
        &self,
        uri: &http::Uri,
        method: http::Method,
        headers: &http::HeaderMap,
        body: bytes::Bytes,
    ) -> Result<http::Response<Incoming>, CanonicalError> {
        self.send_request_uri(uri, method, headers, body).await
    }

    /// Send a request using the hyper passthrough client and a URL string.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::Transport`] when URI parsing fails, passthrough
    /// is disabled, request execution fails, or retries are exhausted.
    pub async fn send_request_uri_str(
        &self,
        url: &str,
        method: http::Method,
        headers: &http::HeaderMap,
        body: bytes::Bytes,
    ) -> Result<http::Response<Incoming>, CanonicalError> {
        let parsed_uri = self.parsed_uri(url)?;
        self.send_request_uri(parsed_uri.as_ref(), method, headers, body)
            .await
    }

    /// Send a streaming request using the hyper passthrough client and a URL string.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalError::Transport`] when URI parsing fails, passthrough
    /// is disabled, request execution fails, or retries are exhausted.
    pub async fn send_stream_uri_str(
        &self,
        url: &str,
        method: http::Method,
        headers: &http::HeaderMap,
        body: bytes::Bytes,
    ) -> Result<http::Response<Incoming>, CanonicalError> {
        self.send_request_uri_str(url, method, headers, body).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyper_passthrough_client_is_lazy() {
        let transport = HttpTransport::new(&ServerConfig::default());
        assert!(transport.hyper_passthrough_enabled());
        assert!(transport.hyper_passthrough_enabled_for(None));
        assert!(!transport.hyper_passthrough_enabled_for(Some("http://127.0.0.1:8080")));
        assert!(transport.hyper_passthrough_https_client.get().is_none());
        assert!(transport.hyper_passthrough_http_client.get().is_none());
        assert!(transport.hyper_passthrough_h2c_client.get().is_none());

        let passthrough_https = transport.hyper_passthrough_https_client();
        let passthrough_http = transport.hyper_passthrough_http_client();
        let passthrough_h2c = transport.hyper_passthrough_h2c_client();
        assert!(passthrough_https.is_some());
        assert!(passthrough_http.is_some());
        assert!(passthrough_h2c.is_some());
        assert!(transport.hyper_passthrough_https_client.get().is_some());
        assert!(transport.hyper_passthrough_http_client.get().is_some());
        assert!(transport.hyper_passthrough_h2c_client.get().is_some());
    }

    #[test]
    fn test_base_reqwest_client_is_lazy() {
        let transport = HttpTransport::new(&ServerConfig::default());
        assert!(transport.base_client.get().is_none());
        let _ = transport.reqwest_client_for_proxy(None).unwrap();
        assert!(transport.base_client.get().is_some());
    }

    #[test]
    fn test_hyper_passthrough_client_disabled_with_env_proxy() {
        let transport = HttpTransport::new(&ServerConfig {
            http_use_env_proxy: true,
            ..ServerConfig::default()
        });
        assert!(!transport.hyper_passthrough_enabled());
        assert!(!transport.hyper_passthrough_enabled_for(None));
        assert!(transport.hyper_passthrough_https_client().is_none());
        assert!(transport.hyper_passthrough_http_client().is_none());
        assert!(transport.hyper_passthrough_h2c_client().is_none());
        assert!(transport.hyper_passthrough_https_client.get().is_none());
        assert!(transport.hyper_passthrough_http_client.get().is_none());
        assert!(transport.hyper_passthrough_h2c_client.get().is_none());
    }

    #[test]
    fn test_parsed_url_cache_hit() {
        let transport = HttpTransport::new(&ServerConfig::default());
        let url = "https://api.example.com/v1/chat/completions";

        let first = transport.parsed_url(url).unwrap();
        let second = transport.parsed_url(url).unwrap();

        assert_eq!(first.as_ref(), second.as_ref());
        assert_eq!(transport.parsed_url_cache.read().len(), 1);
    }

    #[test]
    fn test_parsed_endpoint_cache_is_bounded() {
        let transport = HttpTransport::new(&ServerConfig::default());

        for idx in 0..(PARSED_ENDPOINT_CACHE_MAX_ENTRIES + 32) {
            let url = format!("https://api.example.com/v1/models/gemini-{idx}:generateContent");
            let _ = transport.parsed_url(&url).unwrap();
            let _ = transport.parsed_uri(&url).unwrap();
        }

        assert!(transport.parsed_url_cache.read().len() <= PARSED_ENDPOINT_CACHE_MAX_ENTRIES);
        assert!(transport.parsed_uri_cache.read().len() <= PARSED_ENDPOINT_CACHE_MAX_ENTRIES);
    }

    #[test]
    fn test_parsed_url_invalid() {
        let transport = HttpTransport::new(&ServerConfig::default());
        let err = transport.parsed_url("://bad-url").unwrap_err();
        assert!(matches!(err, CanonicalError::Transport(_)));
    }

    #[test]
    fn test_proxy_client_cache_hit() {
        let transport = HttpTransport::new(&ServerConfig::default());
        let first = transport
            .reqwest_client_for_proxy(Some("http://127.0.0.1:8080"))
            .unwrap();
        let second = transport
            .reqwest_client_for_proxy(Some("http://127.0.0.1:8080"))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(transport.dynamic_proxy_clients.read().len(), 1);
    }

    #[test]
    fn test_preconfigured_proxy_client_hit() {
        let transport = HttpTransport::new_with_upstream_count_and_proxies(
            &ServerConfig::default(),
            1,
            ["http://127.0.0.1:8080"],
        );
        let first = transport
            .reqwest_client_for_proxy(Some("http://127.0.0.1:8080"))
            .unwrap();
        let second = transport
            .reqwest_client_for_proxy(Some("http://127.0.0.1:8080"))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(transport.preconfigured_proxy_clients.len(), 1);
        assert!(transport.dynamic_proxy_clients.read().is_empty());
    }
}
