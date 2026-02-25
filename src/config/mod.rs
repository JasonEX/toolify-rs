pub mod validation;

use serde::{Deserialize, Serialize};
use std::fmt;

use self::validation::validate_config;

/// Error type for configuration loading and validation.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("Config validation error: {0}")]
    Validation(String),
}

/// Function calling mode for an upstream service.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FcMode {
    #[default]
    Inject,
    Native,
    Auto,
}

impl fmt::Display for FcMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FcMode::Inject => write!(f, "inject"),
            FcMode::Native => write!(f, "native"),
            FcMode::Auto => write!(f, "auto"),
        }
    }
}

/// Server configuration.
#[derive(Debug, Clone, Serialize)]
pub struct ServerConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(default = "default_http_pool_max_idle_per_host")]
    pub http_pool_max_idle_per_host: usize,
    #[serde(default = "default_http_pool_idle_timeout_secs")]
    pub http_pool_idle_timeout_secs: u64,
    #[serde(default = "default_models_cache_ttl_secs")]
    pub models_cache_ttl_secs: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_worker_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_max_blocking_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_thread_stack_size_kb: Option<usize>,
    #[serde(default)]
    pub base_path: String,
    #[serde(default)]
    pub trust_forwarded_headers: bool,
    #[serde(default)]
    pub http_use_env_proxy: bool,
    #[serde(default)]
    pub http_force_h2c_upstream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tcp_reuse_port_listener_count: Option<usize>,
}

fn default_port() -> u16 {
    8000
}
fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_timeout() -> u64 {
    180
}
fn default_http_pool_max_idle_per_host() -> usize {
    16
}
fn default_http_pool_idle_timeout_secs() -> u64 {
    15
}
fn default_models_cache_ttl_secs() -> u64 {
    300
}

#[derive(Debug, Deserialize)]
struct ServerConfigWire {
    #[serde(default = "default_port")]
    port: u16,
    #[serde(default = "default_host")]
    host: String,
    #[serde(default = "default_timeout")]
    timeout: u64,
    #[serde(default = "default_http_pool_max_idle_per_host")]
    http_pool_max_idle_per_host: usize,
    #[serde(default = "default_http_pool_idle_timeout_secs")]
    http_pool_idle_timeout_secs: u64,
    #[serde(default = "default_models_cache_ttl_secs")]
    models_cache_ttl_secs: u64,
    #[serde(default)]
    runtime_worker_threads: Option<RuntimeThreadsSetting>,
    #[serde(default)]
    runtime_max_blocking_threads: Option<RuntimeThreadsSetting>,
    #[serde(default)]
    runtime_thread_stack_size_kb: Option<usize>,
    #[serde(default)]
    base_path: String,
    #[serde(default)]
    trust_forwarded_headers: bool,
    #[serde(default)]
    http_use_env_proxy: bool,
    #[serde(default)]
    http_force_h2c_upstream: bool,
    #[serde(default)]
    tcp_reuse_port_listener_count: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RuntimeThreadsSetting {
    Fixed(usize),
    Auto(()),
}

fn runtime_threads_or_default(
    setting: Option<&RuntimeThreadsSetting>,
    default: Option<usize>,
) -> Option<usize> {
    match setting {
        None => default,
        Some(RuntimeThreadsSetting::Fixed(threads)) => Some(*threads),
        Some(RuntimeThreadsSetting::Auto(())) => None,
    }
}

impl<'de> Deserialize<'de> for ServerConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ServerConfigWire::deserialize(deserializer)?;
        Ok(Self {
            port: wire.port,
            host: wire.host,
            timeout: wire.timeout,
            http_pool_max_idle_per_host: wire.http_pool_max_idle_per_host,
            http_pool_idle_timeout_secs: wire.http_pool_idle_timeout_secs,
            models_cache_ttl_secs: wire.models_cache_ttl_secs,
            // missing => Some(default), explicit null => None
            runtime_worker_threads: runtime_threads_or_default(
                wire.runtime_worker_threads.as_ref(),
                None,
            ),
            runtime_max_blocking_threads: runtime_threads_or_default(
                wire.runtime_max_blocking_threads.as_ref(),
                Some(8),
            ),
            runtime_thread_stack_size_kb: wire.runtime_thread_stack_size_kb,
            base_path: wire.base_path,
            trust_forwarded_headers: wire.trust_forwarded_headers,
            http_use_env_proxy: wire.http_use_env_proxy,
            http_force_h2c_upstream: wire.http_force_h2c_upstream,
            tcp_reuse_port_listener_count: wire.tcp_reuse_port_listener_count,
        })
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            host: default_host(),
            timeout: default_timeout(),
            http_pool_max_idle_per_host: default_http_pool_max_idle_per_host(),
            http_pool_idle_timeout_secs: default_http_pool_idle_timeout_secs(),
            models_cache_ttl_secs: default_models_cache_ttl_secs(),
            runtime_worker_threads: None,
            runtime_max_blocking_threads: Some(8),
            runtime_thread_stack_size_kb: None,
            base_path: String::new(),
            trust_forwarded_headers: false,
            http_use_env_proxy: false,
            http_force_h2c_upstream: false,
            tcp_reuse_port_listener_count: None,
        }
    }
}

/// Upstream service configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamServiceConfig {
    pub name: String,
    #[serde(default = "default_provider")]
    pub provider: String,
    pub base_url: String,
    pub api_key: String,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub is_default: bool,
    #[serde(default)]
    pub fc_mode: FcMode,
    #[serde(default)]
    pub api_version: Option<String>,
    #[serde(default)]
    pub proxy: Option<String>,
    #[serde(default)]
    pub proxy_stream: Option<String>,
    #[serde(default)]
    pub proxy_non_stream: Option<String>,
}

fn default_provider() -> String {
    "openai".to_string()
}

/// Client authentication configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientAuthConfig {
    pub allowed_keys: Vec<String>,
}

/// Feature flags and settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturesConfig {
    #[serde(default = "default_true")]
    pub enable_function_calling: bool,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default = "default_true")]
    pub convert_developer_to_system: bool,
    #[serde(default)]
    pub enable_fc_error_retry: bool,
    #[serde(default = "default_fc_retry_max")]
    pub fc_error_retry_max_attempts: u32,
    #[serde(default)]
    pub prompt_template: Option<String>,
    #[serde(default)]
    pub fc_error_retry_prompt_template: Option<String>,
}

fn default_true() -> bool {
    true
}
fn default_log_level() -> String {
    "INFO".to_string()
}
fn default_fc_retry_max() -> u32 {
    3
}

impl Default for FeaturesConfig {
    fn default() -> Self {
        Self {
            enable_function_calling: true,
            log_level: default_log_level(),
            convert_developer_to_system: true,
            enable_fc_error_retry: false,
            fc_error_retry_max_attempts: default_fc_retry_max(),
            prompt_template: None,
            fc_error_retry_prompt_template: None,
        }
    }
}

/// Top-level application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    pub upstream_services: Vec<UpstreamServiceConfig>,
    pub client_authentication: ClientAuthConfig,
    #[serde(default)]
    pub features: FeaturesConfig,
}

/// Load configuration from a YAML file and validate it.
///
/// # Errors
///
/// Returns [`ConfigError::Io`] when reading the file fails, [`ConfigError::Yaml`]
/// when parsing fails, or [`ConfigError::Validation`] when semantic validation fails.
pub fn load_config(path: &str) -> Result<AppConfig, ConfigError> {
    let contents = std::fs::read_to_string(path)?;
    let config: AppConfig = serde_yaml::from_str(&contents)?;
    validate_config(&config)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_example_config() {
        // The example config should load and validate successfully
        let config = load_config("config.example.yaml");
        assert!(
            config.is_ok(),
            "Failed to load example config: {:?}",
            config.err()
        );
        let config = config.unwrap();
        assert_eq!(config.server.port, 8000);
        assert!(!config.server.http_use_env_proxy);
        assert!(!config.server.http_force_h2c_upstream);
        assert!(config.server.tcp_reuse_port_listener_count.is_none());
        assert_eq!(config.server.http_pool_max_idle_per_host, 16);
        assert!(config.upstream_services.len() >= 2);
        assert_eq!(config.client_authentication.allowed_keys.len(), 2);
        assert!(config.features.enable_function_calling);
    }

    #[test]
    fn test_fc_mode_default() {
        assert_eq!(FcMode::default(), FcMode::Inject);
    }

    #[test]
    fn test_fc_mode_serde() {
        let json = serde_json::to_string(&FcMode::Native).unwrap();
        assert_eq!(json, "\"native\"");
        let mode: FcMode = serde_json::from_str("\"auto\"").unwrap();
        assert_eq!(mode, FcMode::Auto);
    }

    #[test]
    fn test_server_config_runtime_defaults() {
        let server = ServerConfig::default();
        assert_eq!(server.runtime_worker_threads, None);
        assert_eq!(server.runtime_max_blocking_threads, Some(8));
        assert_eq!(server.runtime_thread_stack_size_kb, None);
        assert!(!server.http_force_h2c_upstream);
    }
}
