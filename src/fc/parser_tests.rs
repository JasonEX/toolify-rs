use super::*;

const TRIGGER: &str = "<Function_AB12_Start/>";

// -- remove_think_blocks ------------------------------------------------

#[test]
fn remove_think_no_think_blocks() {
    let input = "Hello world";
    assert_eq!(remove_think_blocks(input), "Hello world");
}

#[test]
fn remove_think_single_block() {
    let input = "before<think>secret</think>after";
    assert_eq!(remove_think_blocks(input), "beforeafter");
}

#[test]
fn remove_think_nested_blocks() {
    let input = "a<think>outer<think>inner</think>still outer</think>b";
    assert_eq!(remove_think_blocks(input), "ab");
}

#[test]
fn remove_think_multiple_sequential() {
    let input = "<think>one</think>mid<think>two</think>end";
    assert_eq!(remove_think_blocks(input), "midend");
}

#[test]
fn remove_reasoning_single_block() {
    let input = "before<reasoning>hidden</reasoning>after";
    assert_eq!(remove_think_blocks(input), "beforeafter");
}

#[test]
fn remove_thinking_single_block() {
    let input = "before<thinking>hidden</thinking>after";
    assert_eq!(remove_think_blocks(input), "beforeafter");
}

#[test]
fn remove_analysis_single_block() {
    let input = "before<analysis>hidden</analysis>after";
    assert_eq!(remove_think_blocks(input), "beforeafter");
}

#[test]
fn remove_think_unclosed_block() {
    // Unclosed think block should stop removal and preserve the rest.
    let input = "before<think>never closed";
    assert_eq!(remove_think_blocks(input), "before<think>never closed");
}

// -- parse_function_calls: basic ----------------------------------------

#[test]
fn parse_single_invoke() {
    let text = format!(
        "Some text\n{TRIGGER}\n<function_calls>\
             <invoke name=\"my_tool\">\
             <parameter name=\"query\">hello world</parameter>\
             </invoke>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "my_tool");
    assert_eq!(result[0].arguments["query"], "hello world");
    assert!(result[0].arguments_json.is_none());
}

#[test]
fn parse_multiple_invokes() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <invoke name=\"tool_a\"><parameter name=\"x\">1</parameter></invoke>\
             <invoke name=\"tool_b\"><parameter name=\"y\">2</parameter></invoke>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].name, "tool_a");
    assert_eq!(result[1].name, "tool_b");
}

// -- CDATA unwrapping ---------------------------------------------------

#[test]
fn parse_cdata_parameter() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <invoke name=\"search\">\
             <parameter name=\"query\"><![CDATA[foo & bar]]></parameter>\
             </invoke>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result[0].arguments["query"], "foo & bar");
}

// -- JSON coercion ------------------------------------------------------

#[test]
fn parse_json_object_parameter() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <invoke name=\"run\">\
             <parameter name=\"config\">{{\"key\": \"val\"}}</parameter>\
             </invoke>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert!(result[0].arguments["config"].is_object());
    assert_eq!(result[0].arguments["config"]["key"], "val");
}

#[test]
fn parse_json_array_parameter() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <invoke name=\"run\">\
             <parameter name=\"items\">[1, 2, 3]</parameter>\
             </invoke>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert!(result[0].arguments["items"].is_array());
}

// -- Last trigger signal ------------------------------------------------

#[test]
fn uses_last_trigger_signal() {
    // Two trigger signals — parser should use the last one.
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <invoke name=\"old\"><parameter name=\"a\">1</parameter></invoke>\
             </function_calls>\n\
             Some intervening text\n\
             {TRIGGER}\n<function_calls>\
             <invoke name=\"new\"><parameter name=\"b\">2</parameter></invoke>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "new");
}

// -- Think block interaction --------------------------------------------

#[test]
fn trigger_inside_think_is_ignored() {
    let text = format!(
        "<think>Reasoning {TRIGGER}\n<function_calls>\
             <invoke name=\"bad\"><parameter name=\"a\">1</parameter></invoke>\
             </function_calls></think>\n\
             {TRIGGER}\n<function_calls>\
             <invoke name=\"good\"><parameter name=\"b\">2</parameter></invoke>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "good");
}

#[test]
fn trigger_inside_reasoning_is_ignored() {
    let text = format!(
        "<reasoning>Reasoning {TRIGGER}\n<function_calls>\
             <invoke name=\"bad\"><parameter name=\"a\">1</parameter></invoke>\
             </function_calls></reasoning>\n\
             {TRIGGER}\n<function_calls>\
             <invoke name=\"good\"><parameter name=\"b\">2</parameter></invoke>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "good");
}

#[test]
fn trigger_inside_thinking_is_ignored() {
    let text = format!(
        "<thinking>Reasoning {TRIGGER}\n<function_calls>\
             <invoke name=\"bad\"><parameter name=\"a\">1</parameter></invoke>\
             </function_calls></thinking>\n\
             {TRIGGER}\n<function_calls>\
             <invoke name=\"good\"><parameter name=\"b\">2</parameter></invoke>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "good");
}

#[test]
fn trigger_inside_analysis_is_ignored() {
    let text = format!(
        "<analysis>Reasoning {TRIGGER}\n<function_calls>\
             <invoke name=\"bad\"><parameter name=\"a\">1</parameter></invoke>\
             </function_calls></analysis>\n\
             {TRIGGER}\n<function_calls>\
             <invoke name=\"good\"><parameter name=\"b\">2</parameter></invoke>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "good");
}

// -- Error cases --------------------------------------------------------

#[test]
fn error_on_empty_input() {
    assert!(parse_function_calls("", TRIGGER).is_err());
}

#[test]
fn error_on_missing_trigger() {
    let text = "<function_calls><invoke name=\"x\"><parameter name=\"a\">1</parameter></invoke></function_calls>";
    assert!(parse_function_calls(text, TRIGGER).is_err());
}

#[test]
fn error_on_no_function_calls_block() {
    let text = format!("{TRIGGER}\nSome text without XML");
    assert!(parse_function_calls(&text, TRIGGER).is_err());
}

// -- Regex fallback -----------------------------------------------------

#[test]
fn regex_fallback_on_malformed_xml() {
    // Intentionally break the XML so strict parse fails — e.g. unclosed tag
    // but the invoke/parameter structure is still extractable via regex.
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <invoke name=\"tool\">\
             <parameter name=\"arg\">value</parameter>\
             </invoke>\
             <extra_unclosed>\
             </function_calls>"
    );
    // This might parse via strict XML or fall back to regex — either way
    // we should get the invoke.
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "tool");
    assert_eq!(result[0].arguments["arg"], "value");
}

// -- unwrap_cdata unit tests --------------------------------------------

#[test]
fn unwrap_cdata_no_cdata() {
    assert_eq!(unwrap_cdata("plain text"), "plain text");
}

#[test]
fn unwrap_cdata_single() {
    assert_eq!(unwrap_cdata("<![CDATA[hello]]>"), "hello");
}

#[test]
fn unwrap_cdata_multiple() {
    assert_eq!(unwrap_cdata("<![CDATA[a]]><![CDATA[b]]>"), "ab");
}

// -- coerce_json_value unit tests ---------------------------------------

#[test]
fn coerce_plain_string() {
    assert_eq!(
        coerce_json_value("hello"),
        serde_json::Value::String("hello".into())
    );
}

#[test]
fn coerce_json_object() {
    let v = coerce_json_value(r#"{"a": 1}"#);
    assert!(v.is_object());
}

#[test]
fn coerce_json_array() {
    let v = coerce_json_value("[1, 2]");
    assert!(v.is_array());
}

#[test]
fn coerce_json_number() {
    let v = coerce_json_value("123");
    assert_eq!(v, serde_json::json!(123));
}

#[test]
fn coerce_json_boolean() {
    let v = coerce_json_value("false");
    assert_eq!(v, serde_json::json!(false));
}

#[test]
fn coerce_json_null() {
    let v = coerce_json_value("null");
    assert_eq!(v, serde_json::Value::Null);
}

#[test]
fn coerce_invalid_json_returns_string() {
    let v = coerce_json_value("{broken");
    assert!(v.is_string());
}

// -- function_call format (prompt-instructed format) ---------------------

#[test]
fn parse_single_function_call_with_args_json() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <function_call>\
             <tool>my_tool</tool>\
             <args_json>{{\"query\": \"hello world\"}}</args_json>\
             </function_call>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "my_tool");
    assert_eq!(result[0].arguments["query"], "hello world");
    assert_eq!(
        result[0].arguments_json.as_deref(),
        Some("{\"query\": \"hello world\"}")
    );
}

#[test]
fn parse_multiple_function_calls() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <function_call>\
             <tool>tool_a</tool>\
             <args_json>{{\"x\": 1}}</args_json>\
             </function_call>\
             <function_call>\
             <tool>tool_b</tool>\
             <args_json>{{\"y\": 2}}</args_json>\
             </function_call>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].name, "tool_a");
    assert_eq!(result[0].arguments["x"], 1);
    assert_eq!(result[0].arguments_json.as_deref(), Some("{\"x\": 1}"));
    assert_eq!(result[1].name, "tool_b");
    assert_eq!(result[1].arguments["y"], 2);
    assert_eq!(result[1].arguments_json.as_deref(), Some("{\"y\": 2}"));
}

#[test]
fn parse_function_call_preserves_id_tag() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <function_call>\
             <id>call_abc-1</id>\
             <tool>tool_a</tool>\
             <args_json>{{\"x\": 1}}</args_json>\
             </function_call>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].id.as_deref(), Some("call_abc-1"));
}

#[test]
fn parse_function_call_cdata_args_json() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <function_call>\
             <tool>search</tool>\
             <args_json><![CDATA[{{\"query\": \"foo & bar\"}}]]></args_json>\
             </function_call>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "search");
    assert_eq!(result[0].arguments["query"], "foo & bar");
    assert_eq!(
        result[0].arguments_json.as_deref(),
        Some("{\"query\": \"foo & bar\"}")
    );
}

#[test]
fn parse_mixed_invoke_and_function_call() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <invoke name=\"old_tool\">\
             <parameter name=\"a\">1</parameter>\
             </invoke>\
             <function_call>\
             <tool>new_tool</tool>\
             <args_json>{{\"b\": 2}}</args_json>\
             </function_call>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].name, "old_tool");
    assert_eq!(result[0].arguments["a"], 1);
    assert_eq!(result[0].id, None);
    assert!(result[0].arguments_json.is_none());
    assert_eq!(result[1].name, "new_tool");
    assert_eq!(result[1].arguments["b"], 2);
    assert_eq!(result[1].id, None);
    assert_eq!(result[1].arguments_json.as_deref(), Some("{\"b\": 2}"));
}

#[test]
fn parse_invoke_preserves_id_attr() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <invoke name=\"old_tool\" id=\"call_old_1\">\
             <parameter name=\"a\">1</parameter>\
             </invoke>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].id.as_deref(), Some("call_old_1"));
}

#[test]
fn parse_function_call_no_args() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <function_call>\
             <tool>ping</tool>\
             <args_json>{{}}</args_json>\
             </function_call>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "ping");
    assert!(result[0].arguments.is_object());
    assert_eq!(result[0].arguments.as_object().unwrap().len(), 0);
}

#[test]
fn parse_function_call_nested_json_args() {
    let text = format!(
            "{TRIGGER}\n<function_calls>\
             <function_call>\
             <tool>complex</tool>\
             <args_json><![CDATA[{{\"config\": {{\"nested\": true}}, \"items\": [1, 2, 3]}}]]></args_json>\
             </function_call>\
             </function_calls>"
        );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "complex");
    assert!(result[0].arguments["config"].is_object());
    assert_eq!(result[0].arguments["config"]["nested"], true);
    assert!(result[0].arguments["items"].is_array());
}

#[test]
fn parse_permissive_single_quotes_and_arguments_tag() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <function_call name='get_weather'>\
               <arguments><![CDATA[{{\"city\":\"SF\"}}]]></arguments>\
             </function_call>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "get_weather");
    assert_eq!(result[0].arguments["city"], "SF");
}

#[test]
fn parse_permissive_without_function_calls_wrapper() {
    let text = format!(
        "{TRIGGER}\n<function_call>\
                <tool>search</tool>\
                <args_json>{{\"q\":\"rust\"}}</args_json>\
             </function_call>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "search");
    assert_eq!(result[0].arguments["q"], "rust");
}

#[test]
fn parse_permissive_case_insensitive_function_call_tags() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <FUNCTION_CALL NAME='Lookup'>\
               <NaMe>lookup_tool</NaMe>\
               <ARGUMENTS><![CDATA[{{\"q\":\"Rust\"}}]]></ARGUMENTS>\
             </FUNCTION_CALL>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "lookup_tool");
    assert_eq!(result[0].arguments["q"], "Rust");
}

#[test]
fn parse_permissive_case_insensitive_invoke_parameter_tags() {
    let text = format!(
        "{TRIGGER}\n<function_calls>\
             <INVOKE NAME='echo'>\
               <PARAMETER NAME='msg'><![CDATA[hi]]></PARAMETER>\
             </INVOKE>\
             </function_calls>"
    );
    let result = parse_function_calls(&text, TRIGGER).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "echo");
    assert_eq!(result[0].arguments["msg"], "hi");
}
