use std::collections::HashSet;

use super::{AppConfig, ConfigError};

/// Validate the full application config, returning an error if any rule is violated.
///
/// # Errors
///
/// Returns [`ConfigError::Validation`] when any configuration invariant is violated.
pub fn validate_config(config: &AppConfig) -> Result<(), ConfigError> {
    validate_server_config(config)?;
    validate_allowed_keys(config)?;
    validate_upstream_services(config)?;
    validate_log_level(config)?;
    validate_prompt_templates(config)?;
    Ok(())
}

fn validation_err(msg: impl Into<String>) -> ConfigError {
    ConfigError::Validation(msg.into())
}

fn validate_server_config(config: &AppConfig) -> Result<(), ConfigError> {
    let server = &config.server;
    if server.http_pool_max_idle_per_host == 0 {
        return Err(validation_err(
            "server.http_pool_max_idle_per_host must be greater than 0",
        ));
    }
    if let Some(worker_threads) = server.runtime_worker_threads {
        if worker_threads == 0 {
            return Err(validation_err(
                "server.runtime_worker_threads must be greater than 0 when set",
            ));
        }
    }
    if let Some(max_blocking_threads) = server.runtime_max_blocking_threads {
        if max_blocking_threads == 0 {
            return Err(validation_err(
                "server.runtime_max_blocking_threads must be greater than 0 when set",
            ));
        }
    }
    if let Some(thread_stack_size_kb) = server.runtime_thread_stack_size_kb {
        if thread_stack_size_kb == 0 {
            return Err(validation_err(
                "server.runtime_thread_stack_size_kb must be greater than 0 when set",
            ));
        }
    }
    if let Some(listener_count) = server.tcp_reuse_port_listener_count {
        if listener_count == 0 {
            return Err(validation_err(
                "server.tcp_reuse_port_listener_count must be greater than 0 when set",
            ));
        }
    }
    Ok(())
}

fn validate_allowed_keys(config: &AppConfig) -> Result<(), ConfigError> {
    if config.client_authentication.allowed_keys.is_empty() {
        return Err(validation_err("allowed_keys cannot be empty"));
    }
    for key in &config.client_authentication.allowed_keys {
        if key.trim().is_empty() {
            return Err(validation_err("allowed_keys contains an empty key"));
        }
    }
    Ok(())
}

const VALID_PROVIDERS: &[&str] = &[
    "openai",
    "openai-responses",
    "anthropic",
    "gemini",
    "gemini-openai",
];

fn validate_upstream_services(config: &AppConfig) -> Result<(), ConfigError> {
    if config.upstream_services.is_empty() {
        return Err(validation_err("upstream_services cannot be empty"));
    }

    // Validate each service individually
    for svc in &config.upstream_services {
        if !svc.base_url.starts_with("http://") && !svc.base_url.starts_with("https://") {
            return Err(validation_err(format!(
                "Service '{}': base_url must start with http:// or https://",
                svc.name
            )));
        }
        if svc.api_key.trim().is_empty() {
            return Err(validation_err(format!(
                "Service '{}': api_key cannot be empty",
                svc.name
            )));
        }
        if !VALID_PROVIDERS.contains(&svc.provider.as_str()) {
            return Err(validation_err(format!(
                "Service '{}': unknown provider '{}'. Must be one of: {}",
                svc.name,
                svc.provider,
                VALID_PROVIDERS.join(", ")
            )));
        }
        validate_proxy_url(&svc.name, "proxy", svc.proxy.as_deref())?;
        validate_proxy_url(&svc.name, "proxy_stream", svc.proxy_stream.as_deref())?;
        validate_proxy_url(
            &svc.name,
            "proxy_non_stream",
            svc.proxy_non_stream.as_deref(),
        )?;
    }

    // Every upstream must have at least one model
    for svc in &config.upstream_services {
        if svc.models.is_empty() {
            return Err(validation_err(format!(
                "Service '{}' must have at least one model",
                svc.name
            )));
        }
    }

    // Multiple upstreams can expose the same model/alias for failover.
    // Only duplicates inside the same service are rejected.
    let mut regular_models = HashSet::new();
    let mut all_aliases = HashSet::new();

    for svc in &config.upstream_services {
        let mut service_entries = HashSet::new();
        for model in &svc.models {
            if model.trim().is_empty() {
                return Err(validation_err(format!(
                    "Service '{}': model name cannot be empty",
                    svc.name
                )));
            }
            if !service_entries.insert(model.clone()) {
                return Err(validation_err(format!(
                    "Service '{}': duplicate model entry '{model}'",
                    svc.name
                )));
            }
            if let Some(colon_pos) = model.find(':') {
                let alias = &model[..colon_pos];
                let real_model = &model[colon_pos + 1..];
                if alias.trim().is_empty() || real_model.trim().is_empty() {
                    return Err(validation_err(format!(
                        "Invalid alias format in '{model}'. Both parts must not be empty."
                    )));
                }
                all_aliases.insert(alias.to_string());
            } else {
                regular_models.insert(model.clone());
            }
        }
    }

    // Alias names must not conflict with regular model names
    for alias in &all_aliases {
        if regular_models.contains(alias) {
            return Err(validation_err(format!(
                "Alias name '{alias}' conflicts with a regular model name"
            )));
        }
    }

    Ok(())
}

fn validate_proxy_url(
    service_name: &str,
    field_name: &str,
    proxy: Option<&str>,
) -> Result<(), ConfigError> {
    let Some(proxy) = proxy.map(str::trim) else {
        return Ok(());
    };
    if proxy.is_empty() {
        return Err(validation_err(format!(
            "Service '{service_name}': {field_name} cannot be empty when set"
        )));
    }
    let parsed = url::Url::parse(proxy).map_err(|err| {
        validation_err(format!(
            "Service '{service_name}': {field_name} is not a valid URL: {err}"
        ))
    })?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(validation_err(format!(
            "Service '{service_name}': {field_name} must use http:// or https://"
        )));
    }
    Ok(())
}

fn validate_log_level(config: &AppConfig) -> Result<(), ConfigError> {
    let valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "DISABLED"];
    if !valid_levels.contains(&config.features.log_level.to_uppercase().as_str()) {
        return Err(validation_err(format!(
            "log_level must be one of {valid_levels:?}"
        )));
    }
    Ok(())
}

fn validate_prompt_templates(config: &AppConfig) -> Result<(), ConfigError> {
    if let Some(ref tmpl) = config.features.prompt_template {
        if !tmpl.contains("{tools_list}") || !tmpl.contains("{trigger_signal}") {
            return Err(validation_err(
                "prompt_template must contain {tools_list} and {trigger_signal} placeholders",
            ));
        }
    }
    if let Some(ref tmpl) = config.features.fc_error_retry_prompt_template {
        if !tmpl.contains("{error_details}") || !tmpl.contains("{original_response}") {
            return Err(validation_err(
                "fc_error_retry_prompt_template must contain {error_details} and {original_response} placeholders",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn make_valid_config() -> AppConfig {
        AppConfig {
            server: ServerConfig::default(),
            upstream_services: vec![UpstreamServiceConfig {
                name: "openai".to_string(),
                provider: "openai".to_string(),
                base_url: "https://api.openai.com/v1".to_string(),
                api_key: "sk-test".to_string(),
                models: vec!["gpt-4".to_string()],
                description: String::new(),
                is_default: true,
                fc_mode: FcMode::Inject,
                api_version: None,
                proxy: None,
                proxy_stream: None,
                proxy_non_stream: None,
            }],
            client_authentication: ClientAuthConfig {
                allowed_keys: vec!["sk-client-key".to_string()],
            },
            features: FeaturesConfig::default(),
        }
    }

    #[test]
    fn test_valid_config() {
        let config = make_valid_config();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_empty_allowed_keys() {
        let mut config = make_valid_config();
        config.client_authentication.allowed_keys = vec![];
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_no_default_service() {
        let mut config = make_valid_config();
        config.upstream_services[0].is_default = false;
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_multiple_defaults() {
        let mut config = make_valid_config();
        let mut svc2 = config.upstream_services[0].clone();
        svc2.name = "second".to_string();
        svc2.models = vec!["model-b".to_string()];
        svc2.is_default = true;
        config.upstream_services.push(svc2);
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_duplicate_model() {
        let mut config = make_valid_config();
        let mut svc2 = config.upstream_services[0].clone();
        svc2.name = "second".to_string();
        svc2.is_default = false;
        // Same model across services is allowed for failover.
        svc2.models = vec!["gpt-4".to_string()];
        config.upstream_services.push(svc2);
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_duplicate_model_within_same_service() {
        let mut config = make_valid_config();
        config.upstream_services[0].models.push("gpt-4".to_string());
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_alias_conflicts_with_model() {
        let mut config = make_valid_config();
        // "gpt-4" is a regular model, add an alias "gpt-4:gpt-4-turbo"
        config.upstream_services[0]
            .models
            .push("gpt-4:gpt-4-turbo".to_string());
        let err = validate_config(&config);
        assert!(err.is_err());
    }

    #[test]
    fn test_invalid_base_url() {
        let mut config = make_valid_config();
        config.upstream_services[0].base_url = "ftp://bad.url".to_string();
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_empty_api_key() {
        let mut config = make_valid_config();
        config.upstream_services[0].api_key = "  ".to_string();
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_log_level() {
        let mut config = make_valid_config();
        config.features.log_level = "VERBOSE".to_string();
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_pool_max_idle_per_host() {
        let mut config = make_valid_config();
        config.server.http_pool_max_idle_per_host = 0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_runtime_worker_threads() {
        let mut config = make_valid_config();
        config.server.runtime_worker_threads = Some(0);
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_runtime_max_blocking_threads() {
        let mut config = make_valid_config();
        config.server.runtime_max_blocking_threads = Some(0);
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_runtime_thread_stack_size_kb() {
        let mut config = make_valid_config();
        config.server.runtime_thread_stack_size_kb = Some(0);
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_tcp_reuse_port_listener_count() {
        let mut config = make_valid_config();
        config.server.tcp_reuse_port_listener_count = Some(0);
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_prompt_template_missing_placeholders() {
        let mut config = make_valid_config();
        config.features.prompt_template = Some("no placeholders here".to_string());
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_prompt_template_valid() {
        let mut config = make_valid_config();
        config.features.prompt_template =
            Some("Tools: {tools_list}\nSignal: {trigger_signal}".to_string());
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_fc_error_retry_template_missing_placeholders() {
        let mut config = make_valid_config();
        config.features.fc_error_retry_prompt_template = Some("bad template".to_string());
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_fc_error_retry_template_valid() {
        let mut config = make_valid_config();
        config.features.fc_error_retry_prompt_template =
            Some("Error: {error_details}\nOriginal: {original_response}".to_string());
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_no_models_is_invalid() {
        let mut config = make_valid_config();
        config.upstream_services[0].models = vec![];
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_unknown_provider() {
        let mut config = make_valid_config();
        config.upstream_services[0].provider = "unknown-provider".to_string();
        let result = validate_config(&config);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("unknown provider"));
    }

    #[test]
    fn test_valid_providers() {
        for provider in &[
            "openai",
            "openai-responses",
            "anthropic",
            "gemini",
            "gemini-openai",
        ] {
            let mut config = make_valid_config();
            config.upstream_services[0].provider = (*provider).to_string();
            assert!(
                validate_config(&config).is_ok(),
                "Provider '{provider}' should be valid"
            );
        }
    }

    #[test]
    fn test_invalid_proxy_url() {
        let mut config = make_valid_config();
        config.upstream_services[0].proxy = Some("bad-proxy".to_string());
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_valid_proxy_overrides() {
        let mut config = make_valid_config();
        config.upstream_services[0].proxy = Some("http://127.0.0.1:8080".to_string());
        config.upstream_services[0].proxy_stream = Some("http://127.0.0.1:8081".to_string());
        config.upstream_services[0].proxy_non_stream = Some("http://127.0.0.1:8082".to_string());
        assert!(validate_config(&config).is_ok());
    }
}
