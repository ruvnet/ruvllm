//! Per-case failure taxonomy. Mirrors `measurement.py`'s grading so that its
//! correctness verdicts can be cross-checked, then names *why* a case failed.
//! Model output is parsed as data and never executed.
use regex::Regex;
use serde_json::{Map, Value};
use std::sync::OnceLock;

/// Why a case was or was not delivered correctly. Ordered by precedence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Category {
    Correct,
    /// Correct answer, but the generation did not finish cleanly.
    CorrectUndelivered,
    /// Hit the 512-token cap.
    Truncated,
    /// No tool call at all: answered in prose (instruction-following failure).
    NoToolCall,
    /// Tool envelope or call body could not be parsed.
    MalformedCall,
    /// More than one call where exactly one was required.
    ExtraCalls,
    WrongTool,
    ArgumentMissing,
    ArgumentExtra,
    ArgumentType,
    ArgumentValue,
    /// Math: no final `\boxed{integer}`.
    NoBoxedAnswer,
    WrongAnswer,
}

impl Category {
    pub fn name(self) -> &'static str {
        match self {
            Category::Correct => "correct",
            Category::CorrectUndelivered => "correct_undelivered",
            Category::Truncated => "truncated",
            Category::NoToolCall => "no_tool_call",
            Category::MalformedCall => "malformed_call",
            Category::ExtraCalls => "extra_calls",
            Category::WrongTool => "wrong_tool",
            Category::ArgumentMissing => "argument_missing",
            Category::ArgumentExtra => "argument_extra",
            Category::ArgumentType => "argument_type",
            Category::ArgumentValue => "argument_value",
            Category::NoBoxedAnswer => "no_boxed_answer",
            Category::WrongAnswer => "wrong_answer",
        }
    }

    /// The coarse group used for "what to improve next".
    pub fn group(self) -> &'static str {
        match self {
            Category::Correct => "correct",
            Category::CorrectUndelivered | Category::Truncated => "truncation",
            Category::NoToolCall | Category::ExtraCalls => "instruction_following",
            Category::MalformedCall => "format",
            Category::WrongTool => "tool_selection",
            Category::ArgumentMissing | Category::ArgumentExtra | Category::ArgumentType | Category::ArgumentValue => "argument_error",
            Category::NoBoxedAnswer | Category::WrongAnswer => "arithmetic",
        }
    }

    pub const ALL: [Category; 13] = [
        Category::Correct, Category::CorrectUndelivered, Category::Truncated, Category::NoToolCall,
        Category::MalformedCall, Category::ExtraCalls, Category::WrongTool, Category::ArgumentMissing,
        Category::ArgumentExtra, Category::ArgumentType, Category::ArgumentValue, Category::NoBoxedAnswer,
        Category::WrongAnswer,
    ];
}

pub struct Graded {
    pub category: Category,
    /// Same meaning as the scorer's `delivered_correct`.
    pub delivered_correct: bool,
    pub completed: bool,
    /// Argument key involved in an argument error, if any.
    pub detail: Option<String>,
}

fn re(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("static regex"))
}

/// Text after the last closed reasoning block, with chat end tokens removed.
fn visible(text: &str) -> String {
    static END: OnceLock<Regex> = OnceLock::new();
    let tail = text.rsplit_once("</think>").map_or(text, |(_, after)| after);
    re(&END, r"<\|(?:im_end|endoftext)\|>").replace_all(tail, "").trim().to_string()
}

enum ParseError {
    Malformed,
    UnknownTool(String),
    ArgType(String),
    ArgExtra(String),
}

fn schema_for<'a>(tools: &'a [Value], name: &str) -> Option<&'a Map<String, Value>> {
    tools.iter().find(|t| t["function"]["name"] == name)?.get("function")?["parameters"]["properties"].as_object()
}

fn parse_calls(text: &str, tools: &[Value]) -> Result<Vec<(String, Map<String, Value>)>, ParseError> {
    static BLOCK: OnceLock<Regex> = OnceLock::new();
    static FUNCTION: OnceLock<Regex> = OnceLock::new();
    static PARAM: OnceLock<Regex> = OnceLock::new();
    let mut blocks: Vec<String> = re(&BLOCK, r"(?s)<tool_call>(.*?)</tool_call>").captures_iter(text).map(|c| c[1].to_string()).collect();
    if blocks.is_empty() && text.trim().starts_with('{') {
        blocks.push(text.trim().to_string());
    } else if text.matches("<tool_call>").count() != blocks.len() || text.matches("</tool_call>").count() != blocks.len() {
        return Err(ParseError::Malformed);
    }
    let mut calls = Vec::new();
    for block in blocks {
        let block = block.trim();
        if block.starts_with('{') {
            let value: Value = serde_json::from_str(block).map_err(|_| ParseError::Malformed)?;
            let object = value.as_object().ok_or(ParseError::Malformed)?;
            if object.len() != 2 {
                return Err(ParseError::Malformed);
            }
            let name = object.get("name").and_then(Value::as_str).ok_or(ParseError::Malformed)?;
            let arguments = object.get("arguments").and_then(Value::as_object).cloned().ok_or(ParseError::Malformed)?;
            calls.push((name.to_string(), arguments));
            continue;
        }
        let whole = re(&FUNCTION, r"(?s)\A<function=([^<>]+)>(.*?)</function>\z").captures(block).ok_or(ParseError::Malformed)?;
        let (name, body) = (whole[1].to_string(), whole.get(2).map_or("", |m| m.as_str()));
        let schema = schema_for(tools, &name).ok_or_else(|| ParseError::UnknownTool(name.clone()))?;
        let (mut arguments, mut offset) = (Map::new(), 0);
        for parameter in re(&PARAM, r"(?s)<parameter=([^<>]+)>(.*?)</parameter>").captures_iter(body) {
            let span = parameter.get(0).expect("match");
            if !body[offset..span.start()].trim().is_empty() {
                return Err(ParseError::Malformed);
            }
            let key = parameter[1].to_string();
            let raw = parameter[2].strip_prefix('\n').unwrap_or(&parameter[2]);
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            let Some(spec) = schema.get(&key) else { return Err(ParseError::ArgExtra(key)) };
            if arguments.contains_key(&key) {
                return Err(ParseError::Malformed);
            }
            let value = match spec["type"].as_str() {
                Some("integer") => {
                    let ok = !raw.is_empty() && raw.strip_prefix('-').unwrap_or(raw).chars().all(|c| c.is_ascii_digit());
                    let number: i64 = if ok { raw.parse().map_err(|_| ParseError::ArgType(key.clone()))? } else { return Err(ParseError::ArgType(key)) };
                    Value::from(number)
                }
                Some("boolean") => match raw {
                    "true" => Value::Bool(true),
                    "false" => Value::Bool(false),
                    _ => return Err(ParseError::ArgType(key)),
                },
                _ => Value::String(raw.to_string()),
            };
            arguments.insert(key, value);
            offset = span.end();
        }
        if !body[offset..].trim().is_empty() {
            return Err(ParseError::Malformed);
        }
        calls.push((name, arguments));
    }
    Ok(calls)
}

fn tool_category(text: &str, case: &Value) -> (Category, Option<String>) {
    let tools = case["tools"].as_array().map_or(&[][..], Vec::as_slice);
    let expected_name = case["expected"]["name"].as_str().unwrap_or_default();
    let empty = Map::new();
    let expected_args = case["expected"]["arguments"].as_object().unwrap_or(&empty);
    let calls = match parse_calls(text, tools) {
        Ok(calls) => calls,
        Err(ParseError::Malformed) => return (Category::MalformedCall, None),
        Err(ParseError::UnknownTool(name)) => return (Category::WrongTool, Some(name)),
        Err(ParseError::ArgType(key)) => return (Category::ArgumentType, Some(key)),
        Err(ParseError::ArgExtra(key)) => return (Category::ArgumentExtra, Some(key)),
    };
    match calls.as_slice() {
        [] => (Category::NoToolCall, None),
        [(name, _)] if name != expected_name => (Category::WrongTool, Some(name.clone())),
        [(_, arguments)] => {
            if let Some(key) = expected_args.keys().find(|k| !arguments.contains_key(*k)) {
                return (Category::ArgumentMissing, Some(key.clone()));
            }
            if let Some(key) = arguments.keys().find(|k| !expected_args.contains_key(*k)) {
                return (Category::ArgumentExtra, Some(key.clone()));
            }
            match expected_args.iter().find(|(k, v)| arguments[*k] != **v) {
                Some((key, value)) if std::mem::discriminant(value) != std::mem::discriminant(&arguments[key]) => (Category::ArgumentType, Some(key.clone())),
                Some((key, _)) => (Category::ArgumentValue, Some(key.clone())),
                None => (Category::Correct, None),
            }
        }
        _ => (Category::ExtraCalls, None),
    }
}

fn math_category(text: &str, answer: &Value) -> Category {
    static BOXED: OnceLock<Regex> = OnceLock::new();
    let values: Vec<String> = re(&BOXED, r"\\boxed\{\s*([+-]?\d+)\s*\}").captures_iter(text).map(|c| c[1].to_string()).collect();
    match values.last() {
        None => Category::NoBoxedAnswer,
        Some(last) => match (last.parse::<i128>(), answer.as_i64()) {
            (Ok(got), Some(want)) if got == i128::from(want) => Category::Correct,
            _ => Category::WrongAnswer,
        },
    }
}

pub fn grade(case: &Value, row: &Value) -> Graded {
    let text = row["text"].as_str().unwrap_or_default();
    let closed = text.rfind("<think>").map_or(-1, |i| i as i64) <= text.rfind("</think>").map_or(-1, |i| i as i64);
    let finish = row["finish_reason"].as_str().unwrap_or_default();
    let completed = finish == "stop" && closed;
    let shown = visible(text);
    let (answer_category, detail) = if case["kind"] == "math" {
        (math_category(&shown, &case["answer"]), None)
    } else {
        tool_category(&shown, case)
    };
    let correct = answer_category == Category::Correct;
    let category = match (correct, completed, finish == "length") {
        (true, true, _) => Category::Correct,
        (true, false, _) => Category::CorrectUndelivered,
        (false, _, true) => Category::Truncated,
        _ => answer_category,
    };
    Graded { category, delivered_correct: correct && completed, completed, detail }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_case() -> Value {
        json!({"kind": "tool", "expected": {"name": "Read", "arguments": {"file_path": "/a", "limit": 5}},
               "tools": [{"function": {"name": "Read", "parameters": {"properties": {"file_path": {"type": "string"}, "limit": {"type": "integer"}}}}},
                         {"function": {"name": "Bash", "parameters": {"properties": {"command": {"type": "string"}}}}}]})
    }
    fn row(text: &str, finish: &str) -> Value { json!({"text": text, "finish_reason": finish}) }
    fn cat(text: &str) -> Category { grade(&tool_case(), &row(text, "stop")).category }

    #[test]
    fn xml_and_json_calls_grade_like_the_scorer() {
        let xml = "<tool_call>\n<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call><|im_end|>";
        assert_eq!(cat(xml), Category::Correct);
        assert_eq!(cat(r#"<tool_call>{"name": "Read", "arguments": {"file_path": "/a", "limit": 5}}</tool_call>"#), Category::Correct);
        assert!(grade(&tool_case(), &row(xml, "stop")).delivered_correct);
    }

    #[test]
    fn failures_are_named() {
        assert_eq!(cat("I would read the file."), Category::NoToolCall);
        assert_eq!(cat("<tool_call><function=Bash><parameter=command>ls</parameter></function></tool_call>"), Category::WrongTool);
        assert_eq!(cat("<tool_call><function=Read><parameter=file_path>/a</parameter></function></tool_call>"), Category::ArgumentMissing);
        assert_eq!(cat("<tool_call><function=Read><parameter=file_path>/b</parameter><parameter=limit>5</parameter></function></tool_call>"), Category::ArgumentValue);
        assert_eq!(cat("<tool_call><function=Read><parameter=file_path>/a</parameter><parameter=limit>five</parameter></function></tool_call>"), Category::ArgumentType);
        assert_eq!(cat(r#"<tool_call>{"name": "Read", "arguments": {"file_path": "/a", "limit": "5"}}</tool_call>"#), Category::ArgumentType);
        assert_eq!(cat("<tool_call><function=Read>"), Category::MalformedCall);
        let two = "<tool_call><function=Bash><parameter=command>ls</parameter></function></tool_call>";
        assert_eq!(cat(&format!("{two}{two}")), Category::ExtraCalls);
        assert_eq!(grade(&tool_case(), &row("<tool_call><function=Read>", "length")).category, Category::Truncated);
    }

    #[test]
    fn math_uses_the_last_boxed_integer_after_reasoning() {
        let case = json!({"kind": "math", "answer": 108});
        assert_eq!(grade(&case, &row("<think>\\boxed{1}</think> so \\boxed{ 108 }", "stop")).category, Category::Correct);
        assert_eq!(grade(&case, &row("\\boxed{108} then \\boxed{107}", "stop")).category, Category::WrongAnswer);
        assert_eq!(grade(&case, &row("108", "stop")).category, Category::NoBoxedAnswer);
        assert_eq!(grade(&case, &row("<think>\\boxed{108}", "length")).category, Category::CorrectUndelivered);
    }
}
