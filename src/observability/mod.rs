pub mod token_counter;

use crate::protocol::canonical::CanonicalUsage;
use tracing_subscriber::EnvFilter;

/// Initialize the tracing subscriber with the configured log level.
///
/// Maps config log levels to tracing levels:
/// - "DISABLED" -> no subscriber installed
/// - "WARNING" -> WARN
/// - "CRITICAL" -> ERROR
/// - Others map directly (DEBUG, INFO, ERROR)
pub fn init_tracing(log_level: &str) {
    let level = log_level.to_uppercase();

    if level == "DISABLED" {
        return;
    }

    let tracing_level = match level.as_str() {
        "WARNING" => "WARN",
        "CRITICAL" => "ERROR",
        other => other,
    };

    let filter = EnvFilter::try_new(tracing_level).unwrap_or_else(|_| EnvFilter::new("INFO"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();
}

/// Log token usage for a completed request, computing duration from start time.
pub fn log_request_complete(model: &str, usage: &CanonicalUsage, start_time: std::time::Instant) {
    token_counter::log_request_usage(model, usage, start_time.elapsed());
}
