use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};

use crate::error::CanonicalError;
use crate::protocol::canonical::{CanonicalToolChoice, CanonicalToolSpec};
use parking_lot::RwLock;

// ---------------------------------------------------------------------------
// Trigger signal – generated once per process (S3-I1)
// ---------------------------------------------------------------------------

static TRIGGER_SIGNAL: LazyLock<String> = LazyLock::new(|| {
    const ALNUM: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let chars: String = (0..4)
        .map(|_| {
            let idx = fastrand::usize(..ALNUM.len());
            ALNUM[idx] as char
        })
        .collect();
    format!("<Function_{chars}_Start/>")
});
static DEFAULT_PROMPT_TEMPLATE: LazyLock<String> =
    LazyLock::new(|| default_prompt_template(get_trigger_signal()));
static PROMPT_CACHE: LazyLock<RwLock<PromptCache>> =
    LazyLock::new(|| RwLock::new(PromptCache::new()));

const PROMPT_CACHE_CAPACITY: usize = 64;

#[derive(Clone)]
pub struct PromptArtifacts {
    prompt: Arc<str>,
    openai_system_message_json: Arc<[u8]>,
}

impl PromptArtifacts {
    #[must_use]
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    #[must_use]
    pub fn openai_system_message_json(&self) -> &[u8] {
        &self.openai_system_message_json
    }
}

#[derive(Clone)]
struct PromptCacheEntry {
    tools: Vec<CanonicalToolSpec>,
    tool_choice: CanonicalToolChoice,
    custom_template: Option<String>,
    prompt: Arc<str>,
    openai_system_message_json: Arc<[u8]>,
}

#[derive(Default)]
struct PromptCache {
    entries: VecDeque<PromptCacheEntry>,
}

impl PromptCache {
    fn new() -> Self {
        Self {
            entries: VecDeque::with_capacity(PROMPT_CACHE_CAPACITY),
        }
    }

    fn get(
        &self,
        tools: &[CanonicalToolSpec],
        tool_choice: &CanonicalToolChoice,
        custom_template: Option<&str>,
    ) -> Option<PromptArtifacts> {
        let entry = self.entries.iter().rfind(|entry| {
            entry.tool_choice == *tool_choice
                && entry.custom_template.as_deref() == custom_template
                && entry.tools == tools
        })?;
        Some(PromptArtifacts {
            prompt: Arc::clone(&entry.prompt),
            openai_system_message_json: Arc::clone(&entry.openai_system_message_json),
        })
    }

    fn insert(
        &mut self,
        tools: &[CanonicalToolSpec],
        tool_choice: &CanonicalToolChoice,
        custom_template: Option<&str>,
        artifacts: &PromptArtifacts,
    ) {
        if let Some(pos) = self.entries.iter().position(|entry| {
            entry.tool_choice == *tool_choice
                && entry.custom_template.as_deref() == custom_template
                && entry.tools == tools
        }) {
            self.entries.remove(pos);
        }

        if self.entries.len() >= PROMPT_CACHE_CAPACITY {
            self.entries.pop_front();
        }

        self.entries.push_back(PromptCacheEntry {
            tools: tools.to_vec(),
            tool_choice: tool_choice.clone(),
            custom_template: custom_template.map(ToOwned::to_owned),
            prompt: Arc::clone(&artifacts.prompt),
            openai_system_message_json: Arc::clone(&artifacts.openai_system_message_json),
        });
    }
}

/// Return the per-process trigger signal (`<Function_XXXX_Start/>`).
#[must_use]
pub fn get_trigger_signal() -> &'static str {
    &TRIGGER_SIGNAL
}

// ---------------------------------------------------------------------------
// Tool list formatting  (matches Python output exactly)
// ---------------------------------------------------------------------------

/// Format a single parameter's detail lines, mirroring the Python logic.
fn format_param_detail(
    p_name: &str,
    p_info: &serde_json::Value,
    required_list: &[String],
) -> Vec<String> {
    let Some(p_info) = p_info.as_object() else {
        // p_info is null / not an object – treat as empty
        let mut lines = Vec::new();
        let p_type = "any";
        let is_required = if required_list.iter().any(|r| r == p_name) {
            "Yes"
        } else {
            "No"
        };
        lines.push(format!("- {p_name}:"));
        lines.push(format!("  - type: {p_type}"));
        lines.push(format!("  - required: {is_required}"));
        return lines;
    };

    let p_type = p_info.get("type").and_then(|v| v.as_str()).unwrap_or("any");

    let is_required = if required_list.iter().any(|r| r == p_name) {
        "Yes"
    } else {
        "No"
    };

    let p_desc = p_info.get("description").and_then(|v| v.as_str());
    let enum_vals = p_info.get("enum");
    let default_val = p_info.get("default");
    let examples_val = p_info.get("examples").or_else(|| p_info.get("example"));

    // Constraints
    let constraint_keys = [
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "uniqueItems",
    ];

    let mut constraints = serde_json::Map::new();
    for key in &constraint_keys {
        if let Some(val) = p_info.get(*key) {
            constraints.insert((*key).to_string(), val.clone());
        }
    }

    // Array item type hint
    if p_type == "array" {
        if let Some(items) = p_info.get("items").and_then(|v| v.as_object()) {
            if let Some(itype) = items.get("type") {
                constraints.insert("items.type".to_string(), itype.clone());
            }
        }
    }

    let mut lines = Vec::new();
    lines.push(format!("- {p_name}:"));
    lines.push(format!("  - type: {p_type}"));
    lines.push(format!("  - required: {is_required}"));

    if let Some(desc) = p_desc {
        lines.push(format!("  - description: {desc}"));
    }
    if let Some(ev) = enum_vals {
        lines.push(format!(
            "  - enum: {}",
            serde_json::to_string(ev).unwrap_or_else(|_| format!("{ev}"))
        ));
    }
    if let Some(dv) = default_val {
        lines.push(format!(
            "  - default: {}",
            serde_json::to_string(dv).unwrap_or_else(|_| format!("{dv}"))
        ));
    }
    if let Some(ex) = examples_val {
        lines.push(format!(
            "  - examples: {}",
            serde_json::to_string(ex).unwrap_or_else(|_| format!("{ex}"))
        ));
    }
    if !constraints.is_empty() {
        let cval = serde_json::Value::Object(constraints);
        lines.push(format!(
            "  - constraints: {}",
            serde_json::to_string(&cval).unwrap_or_else(|_| format!("{cval}"))
        ));
    }

    lines
}

/// Format the list of tools into the text block used inside the prompt.
///
/// Returns `(tools_list_string, required_list_per_tool)`.
/// On validation failure returns `CanonicalError::InvalidRequest`.
fn format_tools_list(tools: &[CanonicalToolSpec]) -> Result<String, CanonicalError> {
    let mut tools_list_str: Vec<String> = Vec::new();

    for (i, tool) in tools.iter().enumerate() {
        let func = &tool.function;
        let name = &func.name;
        let description = func.description.as_deref().unwrap_or("");

        let schema = &func.parameters;

        // --- properties ---
        let props_raw = schema.get("properties");
        let props = match props_raw {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::Object(m)) => Some(m),
            Some(other) => {
                return Err(CanonicalError::InvalidRequest(format!(
                    "Tool '{name}': 'properties' must be an object, got {}",
                    json_type_name(other)
                )));
            }
        };

        // --- required ---
        let required_raw = schema.get("required");
        let required_list: Vec<String> = match required_raw {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(serde_json::Value::Array(arr)) => {
                let mut out = Vec::new();
                for item in arr {
                    match item.as_str() {
                        Some(s) => out.push(s.to_string()),
                        None => {
                            return Err(CanonicalError::InvalidRequest(format!(
                                "Tool '{name}': 'required' entries must be strings, got {item}"
                            )));
                        }
                    }
                }
                out
            }
            Some(other) => {
                return Err(CanonicalError::InvalidRequest(format!(
                    "Tool '{name}': 'required' must be a list, got {}",
                    json_type_name(other)
                )));
            }
        };

        // Validate required keys exist in properties
        let missing_keys: Vec<&str> = required_list
            .iter()
            .filter(|k| props.is_none_or(|m| !m.contains_key(k.as_str())))
            .map(std::string::String::as_str)
            .collect();
        if !missing_keys.is_empty() {
            return Err(CanonicalError::InvalidRequest(format!(
                "Tool '{name}': required parameters {missing_keys:?} are not defined in properties"
            )));
        }

        // params summary: name (type), ...
        let params_summary = props.map_or_else(
            || "None".to_string(),
            |props| {
                if props.is_empty() {
                    "None".to_string()
                } else {
                    props
                        .iter()
                        .map(|(p_name, p_info)| {
                            let ptype = p_info
                                .as_object()
                                .and_then(|o| o.get("type"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("any");
                            format!("{p_name} ({ptype})")
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            },
        );

        // detail block
        let detail_lines: Vec<String> = props
            .into_iter()
            .flat_map(|m| m.iter())
            .flat_map(|(p_name, p_info)| format_param_detail(p_name, p_info, &required_list))
            .collect();

        let detail_block = if detail_lines.is_empty() {
            "(no parameter details)".to_string()
        } else {
            detail_lines.join("\n")
        };

        let desc_block = if description.is_empty() {
            "None".to_string()
        } else {
            format!("```\n{description}\n```")
        };

        let required_str = if required_list.is_empty() {
            "None".to_string()
        } else {
            required_list.join(", ")
        };

        tools_list_str.push(format!(
            "{idx}. <tool name=\"{name}\">\n\
             \x20\x20\x20Description:\n\
             {desc_block}\n\
             \x20\x20\x20Parameters summary: {params_summary}\n\
             \x20\x20\x20Required parameters: {required_str}\n\
             \x20\x20\x20Parameter details:\n\
             {detail_block}",
            idx = i + 1,
        ));
    }

    Ok(tools_list_str.join("\n\n"))
}

/// Return a human-readable JSON type name (mirrors Python `type(x).__name__`).
fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "NoneType",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "str",
        serde_json::Value::Array(_) => "list",
        serde_json::Value::Object(_) => "dict",
    }
}

// ---------------------------------------------------------------------------
// Default prompt template  (exact parity with Python)
// ---------------------------------------------------------------------------

fn default_prompt_template(trigger_signal: &str) -> String {
    format!(
        r#"
You have access to the following available tools to help solve problems:

{{tools_list}}

**IMPORTANT CONTEXT NOTES:**
1. You can call MULTIPLE tools in a single response if needed.
2. Even though you can call multiple tools, you MUST respect the user's later constraints and preferences (e.g., the user may request no tools, only one tool, or a specific tool/workflow).
3. The conversation context may already contain tool execution results from previous function calls. Review the conversation history carefully to avoid unnecessary duplicate tool calls.
4. When tool execution results are present in the context, they will be formatted with XML tags like <tool_result>...</tool_result> for easy identification.
5. This is the ONLY format you can use for tool calls, and any deviation will result in failure.

When you need to use tools, you **MUST** strictly follow this format. Do NOT include any extra text, explanations, or dialogue on the first and second lines of the tool call syntax:

1. When starting tool calls, begin on a new line with exactly:
{trigger_signal}
No leading or trailing spaces, output exactly as shown above. The trigger signal MUST be on its own line and appear only once. Do not output a trigger signal for each tool call.

2. Starting from the second line, **immediately** follow with the complete <function_calls> XML block.

3. For multiple tool calls, include multiple <function_call> blocks within the same <function_calls> wrapper, not separate blocks. Output the trigger signal only once, then one <function_calls> with all <function_call> children.

4. Do not add any text or explanation after the closing </function_calls> tag.

STRICT ARGUMENT KEY RULES:
- You MUST use parameter keys EXACTLY as defined (case- and punctuation-sensitive). Do NOT rename, add, or remove characters.
- If a key starts with a hyphen (e.g., "-i", "-C"), you MUST keep the leading hyphen in the JSON key. Never convert "-i" to "i" or "-C" to "C".
- The <tool> tag must contain the exact name of a tool from the list. Any other tool name is invalid.
- The <args_json> tag must contain a single JSON object with all required arguments for that tool.
- You MAY wrap the JSON content inside <![CDATA[...]]> to avoid XML escaping issues.

CORRECT Example (multiple tool calls):
...response content (optional)...
{trigger_signal}
<function_calls>
    <function_call>
        <tool>Grep</tool>
        <args_json><![CDATA[{{"-i": true, "-C": 2, "path": "."}}]]></args_json>
    </function_call>
    <function_call>
        <tool>search</tool>
        <args_json><![CDATA[{{"keywords": ["Python Document", "how to use python"]}}]]></args_json>
    </function_call>
  </function_calls>

INCORRECT Example (extra text + wrong key names — DO NOT DO THIS):
...response content (optional)...
{trigger_signal}
I will call the tools for you.
<function_calls>
    <function_call>
        <tool>Grep</tool>
        <args>
            <i>true</i>
            <C>2</C>
            <path>.</path>
        </args>
    </function_call>
</function_calls>

INCORRECT Example (output non-XML format — DO NOT DO THIS):
...response content (optional)...
```json
{{"files":[{{"path":"system.py"}}]}}
```

Now please be ready to strictly follow the above specifications.
"#
    )
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate the FC system prompt from tool specs and tool choice.
///
/// This is the Rust equivalent of Python's `generate_function_prompt`.
/// It returns the full prompt text ready to be prepended/appended to the
/// system prompt.
///
/// If `custom_template` is `Some`, it is used instead of the default template.
/// The custom template must contain `{tools_list}` and `{trigger_signal}` placeholders.
///
/// # Errors
///
/// Returns `CanonicalError` when tool schema validation fails or prompt
/// serialization fails.
pub fn generate_fc_prompt_artifacts(
    tools: &[CanonicalToolSpec],
    tool_choice: &CanonicalToolChoice,
    custom_template: Option<&str>,
) -> Result<PromptArtifacts, CanonicalError> {
    if let Some(cached) = PROMPT_CACHE.read().get(tools, tool_choice, custom_template) {
        return Ok(cached);
    }

    let prompt = Arc::<str>::from(generate_fc_prompt_uncached(
        tools,
        tool_choice,
        custom_template,
    )?);
    let openai_system_message_json = encode_openai_system_message_json(prompt.as_ref())?;
    let artifacts = PromptArtifacts {
        prompt,
        openai_system_message_json,
    };
    let mut cache = PROMPT_CACHE.write();
    if let Some(cached) = cache.get(tools, tool_choice, custom_template) {
        return Ok(cached);
    }
    cache.insert(tools, tool_choice, custom_template, &artifacts);
    Ok(artifacts)
}

fn encode_openai_system_message_json(prompt: &str) -> Result<Arc<[u8]>, CanonicalError> {
    let json = serde_json::json!({
        "role": "system",
        "content": prompt,
    });
    serde_json::to_vec(&json)
        .map(Arc::<[u8]>::from)
        .map_err(|e| CanonicalError::Translation(format!("Failed to serialize FC prompt: {e}")))
}

/// Generate the FC system prompt from tool specs and tool choice.
///
/// This is the Rust equivalent of Python's `generate_function_prompt`.
/// It returns the full prompt text ready to be prepended/appended to the
/// system prompt.
///
/// If `custom_template` is `Some`, it is used instead of the default template.
/// The custom template must contain `{tools_list}` and `{trigger_signal}` placeholders.
///
/// # Errors
///
/// Returns `CanonicalError` when tool schema validation fails or prompt
/// generation fails.
pub fn generate_fc_prompt(
    tools: &[CanonicalToolSpec],
    tool_choice: &CanonicalToolChoice,
    custom_template: Option<&str>,
) -> Result<String, CanonicalError> {
    Ok(
        generate_fc_prompt_artifacts(tools, tool_choice, custom_template)?
            .prompt()
            .to_string(),
    )
}

fn generate_fc_prompt_uncached(
    tools: &[CanonicalToolSpec],
    tool_choice: &CanonicalToolChoice,
    custom_template: Option<&str>,
) -> Result<String, CanonicalError> {
    let trigger_signal = get_trigger_signal();

    // Build the tools list text
    let tools_list = format_tools_list(tools)?;

    // Build prompt from template
    let prompt = if let Some(tmpl) = custom_template {
        // Custom template: interpolate trigger_signal first (keep {tools_list} literal),
        // then replace {tools_list}.
        tmpl.replace("{trigger_signal}", trigger_signal)
            .replace("{tools_list}", &tools_list)
    } else {
        DEFAULT_PROMPT_TEMPLATE.replace("{tools_list}", &tools_list)
    };

    // Append tool_choice constraints
    let prompt = match tool_choice {
        CanonicalToolChoice::None => {
            format!("{prompt}\n\nDo NOT call any function.")
        }
        CanonicalToolChoice::Auto => prompt,
        CanonicalToolChoice::Required => {
            format!("{prompt}\n\nYou MUST call at least one function.")
        }
        CanonicalToolChoice::Specific(name) => {
            format!("{prompt}\n\nYou MUST call the function: {name}")
        }
    };

    Ok(prompt)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::{CanonicalToolFunction, CanonicalToolSpec};

    fn make_tool(name: &str, desc: &str, params: serde_json::Value) -> CanonicalToolSpec {
        CanonicalToolSpec {
            function: CanonicalToolFunction {
                name: name.to_string(),
                description: if desc.is_empty() {
                    None
                } else {
                    Some(desc.to_string())
                },
                parameters: params,
            },
        }
    }

    #[test]
    fn trigger_signal_format() {
        let sig = get_trigger_signal();
        assert!(sig.starts_with("<Function_"));
        assert!(sig.ends_with("_Start/>"));
        // 4 alphanumeric chars between the underscores
        let inner = &sig["<Function_".len()..sig.len() - "_Start/>".len()];
        assert_eq!(inner.len(), 4);
        assert!(inner.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn trigger_signal_is_stable() {
        let a = get_trigger_signal();
        let b = get_trigger_signal();
        assert_eq!(a, b, "trigger signal must be the same across calls");
    }

    #[test]
    fn basic_prompt_contains_tool() {
        let tool = make_tool(
            "get_weather",
            "Get current weather",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }),
        );

        let prompt = generate_fc_prompt(&[tool], &CanonicalToolChoice::Auto, None).unwrap();

        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("Get current weather"));
        assert!(prompt.contains("location (string)"));
        assert!(prompt.contains(get_trigger_signal()));
    }

    #[test]
    fn tool_choice_none_appends_constraint() {
        let tool = make_tool("f", "desc", serde_json::json!({}));
        let prompt = generate_fc_prompt(&[tool], &CanonicalToolChoice::None, None).unwrap();
        assert!(prompt.contains("Do NOT call any function."));
    }

    #[test]
    fn tool_choice_required_appends_constraint() {
        let tool = make_tool("f", "desc", serde_json::json!({}));
        let prompt = generate_fc_prompt(&[tool], &CanonicalToolChoice::Required, None).unwrap();
        assert!(prompt.contains("You MUST call at least one function."));
    }

    #[test]
    fn tool_choice_specific_appends_constraint() {
        let tool = make_tool("f", "desc", serde_json::json!({}));
        let prompt = generate_fc_prompt(
            &[tool],
            &CanonicalToolChoice::Specific("f".to_string()),
            None,
        )
        .unwrap();
        assert!(prompt.contains("You MUST call the function: f"));
    }

    #[test]
    fn custom_template_is_used() {
        let tool = make_tool(
            "search",
            "Search things",
            serde_json::json!({"type": "object", "properties": {}}),
        );
        let tmpl = "TOOLS: {tools_list}\nSIGNAL: {trigger_signal}";
        let prompt = generate_fc_prompt(&[tool], &CanonicalToolChoice::Auto, Some(tmpl)).unwrap();
        assert!(prompt.starts_with("TOOLS: "));
        assert!(prompt.contains(get_trigger_signal()));
    }

    #[test]
    fn missing_required_in_properties_is_error() {
        let tool = make_tool(
            "bad",
            "bad tool",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "required": ["nonexistent"]
            }),
        );
        let result = generate_fc_prompt(&[tool], &CanonicalToolChoice::Auto, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("nonexistent"));
    }

    #[test]
    fn param_detail_includes_enum_and_default() {
        let tool = make_tool(
            "set_mode",
            "Set mode",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "slow"],
                        "default": "fast",
                        "description": "The mode to use"
                    }
                },
                "required": ["mode"]
            }),
        );

        let prompt = generate_fc_prompt(&[tool], &CanonicalToolChoice::Auto, None).unwrap();
        assert!(prompt.contains("enum:"));
        assert!(prompt.contains("default:"));
        assert!(prompt.contains("The mode to use"));
    }

    #[test]
    fn no_description_shows_none() {
        let tool = make_tool(
            "f",
            "",
            serde_json::json!({"type": "object", "properties": {}}),
        );
        let prompt = generate_fc_prompt(&[tool], &CanonicalToolChoice::Auto, None).unwrap();
        // With empty description, desc_block = "None"
        assert!(prompt.contains("Description:\nNone"));
    }

    #[test]
    fn array_items_type_constraint() {
        let tool = make_tool(
            "search",
            "search",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                }
            }),
        );

        let prompt = generate_fc_prompt(&[tool], &CanonicalToolChoice::Auto, None).unwrap();
        assert!(prompt.contains("items.type"));
    }
}
