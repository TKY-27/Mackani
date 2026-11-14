use crate::{Error, Result, server::Limits};
use serde_json::Value;

/// Upper bound on the number of values a JSON text can hold, counted without
/// parsing it: every value except the root follows `[`, `,` or `:` outside a
/// string. Lets an expanding document be refused before it is built.
pub fn json_values(bytes: &[u8]) -> usize {
    let (mut n, mut in_string, mut escaped) = (1, false, false);
    for &b in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
        } else if b == b'"' {
            in_string = true;
        } else if matches!(b, b'[' | b',' | b':') {
            n += 1;
        }
    }
    n
}
/// Per-request shape limits, checked on the parsed body before it is cleaned
/// or copied. `values` is the body's [`json_values`] count; the values that
/// tool-call argument strings will expand to are added to it and returned.
pub fn check_size(body: &Value, limits: &Limits, mut values: usize) -> Result<usize> {
    let too_many =
        |what: String, n: usize| Error::request(format!("{what}: at most {n} are accepted"));
    if let Some(messages) = body["messages"].as_array() {
        if messages.len() > limits.max_messages {
            return Err(too_many("messages".into(), limits.max_messages));
        }
        for (i, m) in messages.iter().enumerate() {
            if m["content"].as_array().map_or(0, Vec::len) > limits.max_content_parts {
                return Err(too_many(
                    format!("messages[{i}]: content parts"),
                    limits.max_content_parts,
                ));
            }
            let calls = m["tool_calls"].as_array().map_or(&[][..], Vec::as_slice);
            if calls.len() > limits.max_tool_calls {
                return Err(too_many(
                    format!("messages[{i}]: tool calls"),
                    limits.max_tool_calls,
                ));
            }
            for c in calls {
                if let Some(s) = c["function"]["arguments"].as_str() {
                    values = values.saturating_add(json_values(s.as_bytes()));
                }
            }
        }
    }
    if body["tools"].as_array().map_or(0, Vec::len) > limits.max_tools {
        return Err(too_many("tools".into(), limits.max_tools));
    }
    let stops = match &body["stop"] {
        Value::Array(a) => a.as_slice(),
        s => std::slice::from_ref(s),
    };
    if stops
        .iter()
        .any(|s| s.as_str().is_some_and(|s| s.len() > limits.max_stop_bytes))
    {
        return Err(Error::request(format!(
            "stop strings must be at most {} bytes long",
            limits.max_stop_bytes
        )));
    }
    check_values(values, limits)?;
    Ok(values)
}
pub fn check_values(values: usize, limits: &Limits) -> Result<()> {
    if values > limits.max_json_values {
        return Err(Error::request(format!(
            "the request holds more than {} JSON values",
            limits.max_json_values
        )));
    }
    Ok(())
}
/// Bytes a template can render for one number, boolean or null: the longest
/// spelling of a JSON number, as Python's repr of `-2.2250738585072014e-308`.
pub const SCALAR_BYTES: usize = 24;
/// Bytes of text inside a value: its strings and object keys, and
/// [`SCALAR_BYTES`] for each other value that is not a list or an object.
pub fn text_bytes(v: &Value) -> usize {
    match v {
        Value::String(s) => s.len(),
        Value::Array(a) => a.iter().map(text_bytes).sum(),
        Value::Object(m) => m.iter().map(|(k, v)| k.len() + text_bytes(v)).sum(),
        _ => SCALAR_BYTES,
    }
}
