use crate::protocol::canonical::CanonicalToolSpec;
use regex_lite::Regex;

static REGEX_CACHE: std::sync::LazyLock<
    parking_lot::RwLock<rustc_hash::FxHashMap<String, Option<Regex>>>,
> = std::sync::LazyLock::new(|| parking_lot::RwLock::new(rustc_hash::FxHashMap::default()));

/// A single validation error with a JSON path indicating where it occurred.
#[derive(Debug)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

/// A parsed tool call ready for validation.
#[derive(Debug)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Maximum recursion depth to prevent pathological schemas.
const MAX_DEPTH: usize = 8;

/// Validate a single parsed tool call against the provided tool specs.
///
/// Checks that the tool name exists and that the arguments match the tool's
/// parameter schema.
///
/// # Errors
///
/// Returns `Err(Vec<ValidationError>)` when the tool does not exist, arguments
/// are not an object, or schema validation fails.
pub fn validate_tool_call(
    name: &str,
    arguments: &serde_json::Value,
    tools: &[CanonicalToolSpec],
) -> Result<(), Vec<ValidationError>> {
    let tool = match tools {
        [single] => (single.function.name == name).then_some(single),
        _ => tools.iter().find(|t| t.function.name == name),
    };
    let Some(tool) = tool else {
        let allowed: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        return Err(vec![ValidationError {
            path: name.to_string(),
            message: format!("unknown tool '{name}'. Allowed tools: {allowed:?}"),
        }]);
    };

    if !arguments.is_object() {
        return Err(vec![ValidationError {
            path: name.to_string(),
            message: format!(
                "arguments must be a JSON object, got {}",
                json_type_name(arguments)
            ),
        }]);
    }

    let schema = &tool.function.parameters;
    if schema_is_permissive_object(schema) {
        return Ok(());
    }

    let errors = validate_value(arguments, schema, name, 0);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[inline]
fn schema_is_permissive_object(schema: &serde_json::Value) -> bool {
    let Some(obj) = schema.as_object() else {
        return false;
    };

    if let Some(schema_type) = obj.get("type").and_then(serde_json::Value::as_str) {
        if schema_type != "object" {
            return false;
        }
    }

    let properties_empty = obj
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .is_none_or(serde_json::Map::is_empty);
    if !properties_empty {
        return false;
    }

    let required_empty = obj
        .get("required")
        .and_then(serde_json::Value::as_array)
        .is_none_or(std::vec::Vec::is_empty);
    if !required_empty {
        return false;
    }

    if let Some(additional) = obj.get("additionalProperties") {
        if !additional.as_bool().is_some_and(std::convert::identity) {
            return false;
        }
    }

    for key in obj.keys() {
        if !matches!(
            key.as_str(),
            "type" | "properties" | "required" | "additionalProperties" | "title" | "description"
        ) {
            return false;
        }
    }

    true
}

/// Validate multiple parsed tool calls, collecting all errors.
///
/// # Errors
///
/// Returns `Err(Vec<ValidationError>)` when any call fails validation.
pub fn validate_tool_calls(
    calls: &[ParsedToolCall],
    tools: &[CanonicalToolSpec],
) -> Result<(), Vec<ValidationError>> {
    let mut all_errors = Vec::new();
    for call in calls {
        if let Err(errs) = validate_tool_call(&call.name, &call.arguments, tools) {
            all_errors.extend(errs);
        }
    }
    if all_errors.is_empty() {
        Ok(())
    } else {
        Err(all_errors)
    }
}

/// Validate parser-native tool calls without intermediate cloning.
///
/// # Errors
///
/// Returns `Err(Vec<ValidationError>)` when any call fails validation.
pub fn validate_parser_tool_calls(
    calls: &[crate::fc::parser::ParsedToolCall],
    tools: &[CanonicalToolSpec],
) -> Result<(), Vec<ValidationError>> {
    let mut all_errors = Vec::new();
    for call in calls {
        if let Err(errs) = validate_tool_call(&call.name, &call.arguments, tools) {
            all_errors.extend(errs);
        }
    }
    if all_errors.is_empty() {
        Ok(())
    } else {
        Err(all_errors)
    }
}

/// Recursively validate a JSON value against a JSON Schema subset.
///
/// Ported from Python's `_validate_value_against_schema`.
fn validate_value(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    path: &str,
    depth: usize,
) -> Vec<ValidationError> {
    if depth > MAX_DEPTH {
        return vec![];
    }

    let Some(schema_obj) = schema.as_object() else {
        return vec![];
    };

    if let Some(combinator_errors) = validate_combinators(value, schema_obj, path, depth) {
        return combinator_errors;
    }

    let mut errors = Vec::new();

    if validate_const_enum(value, schema_obj, path, &mut errors) {
        return errors;
    }

    if let Some(schema_type) = schema_type(schema_obj) {
        if validate_type_constraints(schema_type, value, path, &mut errors) {
            return errors;
        }
    } else if has_implicit_object_type(schema_obj)
        && validate_type_constraints_object(value, path, &mut errors)
    {
        return errors;
    }

    validate_string_constraints(value, schema_obj, path, &mut errors);
    validate_numeric_constraints(value, schema_obj, path, &mut errors);
    validate_object_constraints(value, schema_obj, path, depth, &mut errors);
    validate_array_constraints(value, schema_obj, path, depth, &mut errors);

    errors
}

fn validate_combinators(
    value: &serde_json::Value,
    schema_obj: &serde_json::Map<String, serde_json::Value>,
    path: &str,
    depth: usize,
) -> Option<Vec<ValidationError>> {
    if let Some(all_of) = schema_obj.get("allOf").and_then(|v| v.as_array()) {
        let mut errors = Vec::new();
        for (idx, sub) in all_of.iter().enumerate() {
            let empty_schema = serde_json::Value::Object(serde_json::Map::new());
            let sub_schema = if sub.is_null() { &empty_schema } else { sub };
            errors.extend(validate_value(
                value,
                sub_schema,
                &format!("{path}.allOf[{idx}]"),
                depth + 1,
            ));
        }
        return Some(errors);
    }

    if let Some(any_of) = schema_obj.get("anyOf").and_then(|v| v.as_array()) {
        let mut option_errors = Vec::with_capacity(any_of.len());
        for sub in any_of {
            let empty_schema = serde_json::Value::Object(serde_json::Map::new());
            let sub_schema = if sub.is_null() { &empty_schema } else { sub };
            option_errors.push(validate_value(value, sub_schema, path, depth + 1));
        }
        let mut errors = Vec::new();
        if !option_errors.iter().any(std::vec::Vec::is_empty) {
            errors.push(ValidationError {
                path: path.to_string(),
                message: "value does not satisfy anyOf options".to_string(),
            });
        }
        return Some(errors);
    }

    if let Some(one_of) = schema_obj.get("oneOf").and_then(|v| v.as_array()) {
        let mut option_errors = Vec::with_capacity(one_of.len());
        for sub in one_of {
            let empty_schema = serde_json::Value::Object(serde_json::Map::new());
            let sub_schema = if sub.is_null() { &empty_schema } else { sub };
            option_errors.push(validate_value(value, sub_schema, path, depth + 1));
        }
        let ok_count = option_errors.iter().filter(|e| e.is_empty()).count();
        let mut errors = Vec::new();
        if ok_count != 1 {
            errors.push(ValidationError {
                path: path.to_string(),
                message: format!(
                    "value must satisfy exactly one oneOf option (matched {ok_count})"
                ),
            });
        }
        return Some(errors);
    }

    None
}

fn validate_const_enum(
    value: &serde_json::Value,
    schema_obj: &serde_json::Map<String, serde_json::Value>,
    path: &str,
    errors: &mut Vec<ValidationError>,
) -> bool {
    if let Some(const_val) = schema_obj.get("const") {
        if value != const_val {
            errors.push(ValidationError {
                path: path.to_string(),
                message: format!("expected const={const_val}, got {value}"),
            });
            return true;
        }
    }

    if let Some(enum_vals) = schema_obj.get("enum").and_then(|v| v.as_array()) {
        if !enum_vals.contains(value) {
            errors.push(ValidationError {
                path: path.to_string(),
                message: format!("expected one of {enum_vals:?}, got {value}"),
            });
            return true;
        }
    }

    false
}

#[inline]
fn schema_type(
    schema_obj: &serde_json::Map<String, serde_json::Value>,
) -> Option<&serde_json::Value> {
    schema_obj.get("type")
}

#[inline]
fn has_implicit_object_type(schema_obj: &serde_json::Map<String, serde_json::Value>) -> bool {
    schema_obj.contains_key("properties")
        || schema_obj.contains_key("required")
        || schema_obj.contains_key("additionalProperties")
}

#[inline]
fn validate_type_constraints_object(
    value: &serde_json::Value,
    path: &str,
    errors: &mut Vec<ValidationError>,
) -> bool {
    if value.is_object() {
        return false;
    }
    errors.push(ValidationError {
        path: path.to_string(),
        message: format!("expected type 'object', got '{}'", json_type_name(value)),
    });
    true
}

fn validate_type_constraints(
    schema_type: &serde_json::Value,
    value: &serde_json::Value,
    path: &str,
    errors: &mut Vec<ValidationError>,
) -> bool {
    match schema_type {
        serde_json::Value::String(t) => {
            if !type_ok(t, value) {
                errors.push(ValidationError {
                    path: path.to_string(),
                    message: format!("expected type '{}', got '{}'", t, json_type_name(value)),
                });
                return true;
            }
        }
        serde_json::Value::Array(types) => {
            let matches = types
                .iter()
                .any(|t| t.as_str().is_some_and(|ts| type_ok(ts, value)));
            if !matches {
                errors.push(ValidationError {
                    path: path.to_string(),
                    message: format!(
                        "expected type in {:?}, got '{}'",
                        types,
                        json_type_name(value)
                    ),
                });
                return true;
            }
        }
        _ => {}
    }
    false
}

fn validate_string_constraints(
    value: &serde_json::Value,
    schema_obj: &serde_json::Map<String, serde_json::Value>,
    path: &str,
    errors: &mut Vec<ValidationError>,
) {
    let Some(text) = value.as_str() else {
        return;
    };

    if let Some(min_len) = schema_obj
        .get("minLength")
        .and_then(serde_json::Value::as_u64)
    {
        if (text.len() as u64) < min_len {
            errors.push(ValidationError {
                path: path.to_string(),
                message: format!("string shorter than minLength={min_len}"),
            });
        }
    }

    if let Some(max_len) = schema_obj
        .get("maxLength")
        .and_then(serde_json::Value::as_u64)
    {
        if (text.len() as u64) > max_len {
            errors.push(ValidationError {
                path: path.to_string(),
                message: format!("string longer than maxLength={max_len}"),
            });
        }
    }

    if let Some(pattern) = schema_obj
        .get("pattern")
        .and_then(serde_json::Value::as_str)
    {
        if let Some(re) = cached_regex(pattern) {
            if re.is_match(text) {
                return;
            }
            errors.push(ValidationError {
                path: path.to_string(),
                message: format!("string does not match pattern {pattern:?}"),
            });
        }
    }
}

fn cached_regex(pattern: &str) -> Option<Regex> {
    if let Some(cached) = REGEX_CACHE.read().get(pattern) {
        return cached.clone();
    }

    let compiled = Regex::new(pattern).ok();
    let mut cache = REGEX_CACHE.write();
    if cache.len() >= 256 {
        cache.clear();
    }
    cache.insert(pattern.to_string(), compiled.clone());
    compiled
}

fn validate_numeric_constraints(
    value: &serde_json::Value,
    schema_obj: &serde_json::Map<String, serde_json::Value>,
    path: &str,
    errors: &mut Vec<ValidationError>,
) {
    if !value.is_number() {
        return;
    }

    if let Some(min) = schema_obj
        .get("minimum")
        .and_then(serde_json::Value::as_f64)
    {
        if let Some(n) = value.as_f64() {
            if n < min {
                errors.push(ValidationError {
                    path: path.to_string(),
                    message: format!("value {n} is less than minimum {min}"),
                });
            }
        }
    }

    if let Some(max) = schema_obj
        .get("maximum")
        .and_then(serde_json::Value::as_f64)
    {
        if let Some(n) = value.as_f64() {
            if n > max {
                errors.push(ValidationError {
                    path: path.to_string(),
                    message: format!("value {n} is greater than maximum {max}"),
                });
            }
        }
    }
}

fn validate_object_constraints(
    value: &serde_json::Value,
    schema_obj: &serde_json::Map<String, serde_json::Value>,
    path: &str,
    depth: usize,
    errors: &mut Vec<ValidationError>,
) {
    let Some(obj) = value.as_object() else {
        return;
    };

    let empty_map = serde_json::Map::new();
    let properties = schema_obj
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .unwrap_or(&empty_map);

    if let Some(required) = schema_obj
        .get("required")
        .and_then(serde_json::Value::as_array)
    {
        for key in required.iter().filter_map(serde_json::Value::as_str) {
            if !obj.contains_key(key) {
                errors.push(ValidationError {
                    path: path.to_string(),
                    message: format!("missing required property '{key}'"),
                });
            }
        }
    }

    let additional = schema_obj.get("additionalProperties");
    for (key, item) in obj {
        if let Some(prop_schema) = properties.get(key) {
            errors.extend(validate_value(
                item,
                prop_schema,
                &format!("{path}.{key}"),
                depth + 1,
            ));
            continue;
        }

        match additional {
            Some(serde_json::Value::Bool(false)) => {
                errors.push(ValidationError {
                    path: path.to_string(),
                    message: format!("unexpected property '{key}'"),
                });
            }
            Some(additional_schema) if additional_schema.is_object() => {
                errors.extend(validate_value(
                    item,
                    additional_schema,
                    &format!("{path}.{key}"),
                    depth + 1,
                ));
            }
            _ => {}
        }
    }
}

fn validate_array_constraints(
    value: &serde_json::Value,
    schema_obj: &serde_json::Map<String, serde_json::Value>,
    path: &str,
    depth: usize,
    errors: &mut Vec<ValidationError>,
) {
    let Some(arr) = value.as_array() else {
        return;
    };
    let Some(items_schema) = schema_obj.get("items") else {
        return;
    };
    if !items_schema.is_object() {
        return;
    }

    for (idx, item) in arr.iter().enumerate() {
        errors.extend(validate_value(
            item,
            items_schema,
            &format!("{path}[{idx}]"),
            depth + 1,
        ));
    }
}

/// Check if a JSON value matches a JSON Schema type string.
fn type_ok(schema_type: &str, value: &serde_json::Value) -> bool {
    match schema_type {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "null" => value.is_null(),
        _ => true,
    }
}

/// Return a human-readable type name for a JSON value.
fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "integer"
            } else {
                "number"
            }
        }
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::canonical::{CanonicalToolFunction, CanonicalToolSpec};
    use serde_json::json;

    fn make_tool(name: &str, params: serde_json::Value) -> CanonicalToolSpec {
        CanonicalToolSpec {
            function: CanonicalToolFunction {
                name: name.to_string(),
                description: None,
                parameters: params,
            },
        }
    }

    #[test]
    fn test_unknown_tool() {
        let tools = vec![make_tool("foo", json!({}))];
        let result = validate_tool_call("bar", &json!({}), &tools);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs[0].message.contains("unknown tool"));
    }

    #[test]
    fn test_valid_simple_call() {
        let tools = vec![make_tool(
            "get_weather",
            json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
        )];
        let result = validate_tool_call("get_weather", &json!({"city": "London"}), &tools);
        assert!(result.is_ok());
    }

    #[test]
    fn test_missing_required() {
        let tools = vec![make_tool(
            "get_weather",
            json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
        )];
        let result = validate_tool_call("get_weather", &json!({}), &tools);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs[0].message.contains("missing required property 'city'"));
    }

    #[test]
    fn test_wrong_type() {
        let tools = vec![make_tool(
            "test",
            json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer"}
                }
            }),
        )];
        let result = validate_tool_call("test", &json!({"count": "not a number"}), &tools);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs[0].message.contains("expected type 'integer'"));
    }

    #[test]
    fn test_additional_properties_false() {
        let tools = vec![make_tool(
            "test",
            json!({
                "type": "object",
                "properties": {
                    "a": {"type": "string"}
                },
                "additionalProperties": false
            }),
        )];
        let result = validate_tool_call("test", &json!({"a": "ok", "b": "extra"}), &tools);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs[0].message.contains("unexpected property"));
    }

    #[test]
    fn test_enum_validation() {
        let tools = vec![make_tool(
            "test",
            json!({
                "type": "object",
                "properties": {
                    "color": {"type": "string", "enum": ["red", "green", "blue"]}
                }
            }),
        )];
        assert!(validate_tool_call("test", &json!({"color": "red"}), &tools).is_ok());
        assert!(validate_tool_call("test", &json!({"color": "purple"}), &tools).is_err());
    }

    #[test]
    fn test_any_of() {
        let tools = vec![make_tool(
            "test",
            json!({
                "type": "object",
                "properties": {
                    "val": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "integer"}
                        ]
                    }
                }
            }),
        )];
        assert!(validate_tool_call("test", &json!({"val": "hello"}), &tools).is_ok());
        assert!(validate_tool_call("test", &json!({"val": 42}), &tools).is_ok());
        assert!(validate_tool_call("test", &json!({"val": true}), &tools).is_err());
    }

    #[test]
    fn test_one_of() {
        let tools = vec![make_tool(
            "test",
            json!({
                "type": "object",
                "properties": {
                    "val": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "integer"}
                        ]
                    }
                }
            }),
        )];
        assert!(validate_tool_call("test", &json!({"val": "hello"}), &tools).is_ok());
        assert!(validate_tool_call("test", &json!({"val": 42}), &tools).is_ok());
    }

    #[test]
    fn test_string_constraints() {
        let tools = vec![make_tool(
            "test",
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 2, "maxLength": 5}
                }
            }),
        )];
        assert!(validate_tool_call("test", &json!({"name": "abc"}), &tools).is_ok());
        assert!(validate_tool_call("test", &json!({"name": "a"}), &tools).is_err());
        assert!(validate_tool_call("test", &json!({"name": "abcdef"}), &tools).is_err());
    }

    #[test]
    fn test_numeric_constraints() {
        let tools = vec![make_tool(
            "test",
            json!({
                "type": "object",
                "properties": {
                    "age": {"type": "integer", "minimum": 0, "maximum": 150}
                }
            }),
        )];
        assert!(validate_tool_call("test", &json!({"age": 25}), &tools).is_ok());
        assert!(validate_tool_call("test", &json!({"age": -1}), &tools).is_err());
        assert!(validate_tool_call("test", &json!({"age": 200}), &tools).is_err());
    }

    #[test]
    fn test_pattern() {
        let tools = vec![make_tool(
            "test",
            json!({
                "type": "object",
                "properties": {
                    "email": {"type": "string", "pattern": "^[^@]+@[^@]+$"}
                }
            }),
        )];
        assert!(validate_tool_call("test", &json!({"email": "a@b.com"}), &tools).is_ok());
        assert!(validate_tool_call("test", &json!({"email": "nope"}), &tools).is_err());
    }

    #[test]
    fn test_array_items() {
        let tools = vec![make_tool(
            "test",
            json!({
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }),
        )];
        assert!(validate_tool_call("test", &json!({"tags": ["a", "b"]}), &tools).is_ok());
        assert!(validate_tool_call("test", &json!({"tags": ["a", 1]}), &tools).is_err());
    }

    #[test]
    fn test_batch_validation() {
        let tools = vec![make_tool(
            "foo",
            json!({"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}),
        )];
        let calls = vec![
            ParsedToolCall {
                name: "foo".to_string(),
                arguments: json!({"x": "ok"}),
            },
            ParsedToolCall {
                name: "foo".to_string(),
                arguments: json!({}),
            },
        ];
        let result = validate_tool_calls(&calls, &tools);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].message.contains("missing required property"));
    }

    #[test]
    fn test_const_validation() {
        let tools = vec![make_tool(
            "test",
            json!({
                "type": "object",
                "properties": {
                    "version": {"const": 2}
                }
            }),
        )];
        assert!(validate_tool_call("test", &json!({"version": 2}), &tools).is_ok());
        assert!(validate_tool_call("test", &json!({"version": 3}), &tools).is_err());
    }

    #[test]
    fn test_non_object_arguments() {
        let tools = vec![make_tool("test", json!({}))];
        let result = validate_tool_call("test", &json!("string"), &tools);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs[0].message.contains("arguments must be a JSON object"));
    }

    #[test]
    fn test_permissive_object_schema_short_circuit_accepts_any_object() {
        let tools = vec![make_tool(
            "test",
            json!({
                "type": "object",
                "properties": {},
            }),
        )];
        assert!(validate_tool_call("test", &json!({}), &tools).is_ok());
        assert!(validate_tool_call("test", &json!({"x": 1, "y": "z"}), &tools).is_ok());
    }

    #[test]
    fn test_permissive_short_circuit_not_applied_when_constraints_present() {
        let tools = vec![make_tool(
            "test",
            json!({
                "type": "object",
                "properties": {},
                "const": {"x": 1}
            }),
        )];
        assert!(validate_tool_call("test", &json!({"x": 1}), &tools).is_ok());
        assert!(validate_tool_call("test", &json!({"x": 2}), &tools).is_err());
    }
}
