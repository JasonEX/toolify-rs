pub mod detector;
pub mod parser;
pub mod prompt;
pub mod retry;
pub mod validator;

mod action;
mod inject;
mod postprocess;
mod preprocess;

pub use action::{
    allow_auto_inject_fallback, decide_fc_action, get_fc_mode, should_auto_fallback_to_inject,
    FcAction,
};
pub use inject::{apply_fc_inject, apply_fc_inject_take_tools};
pub use postprocess::{
    apply_fc_postprocess_once, extract_response_text, extract_response_text_if_trigger,
    process_fc_response, response_text_contains_trigger, FcResult,
};
pub use preprocess::{preprocess_messages, preprocess_messages_owned};
