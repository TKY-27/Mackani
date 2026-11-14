//! Validation at the OpenAI request boundary.
mod format;
mod limits;
mod messages;

pub use format::{EXCERPT, excerpt};
pub use limits::{SCALAR_BYTES, check_size, check_values, json_values, text_bytes};
pub use messages::{clean_messages, clean_tools};

use crate::{Error, Result, chat::ChatParams};
use format::{excerpt_of, prefix, write_repr};
use serde_json::{Map, Value, json};

pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}
/// Characters of a request value an error message may quote.
#[derive(Debug, Clone)]
pub struct RequestDefaults {
    pub max_tokens: usize,
    pub max_tokens_cap: Option<usize>,
    pub thinking_budget: usize,
    pub sampling: crate::engine::Sampling,
}
impl Default for RequestDefaults {
    fn default() -> Self {
        Self {
            max_tokens: 32768,
            max_tokens_cap: None,
            sampling: crate::engine::Sampling {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                seed: 0,
            },
            thinking_budget: std::env::var("MACKANI_THINKING_BUDGET")
                .ok()
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0)
                .max(0) as usize,
        }
    }
}

// Python accepts numeric strings, bools and truncates float -> int. Reject
// non-finite / out-of-i64 values instead of wrapping or panicking.
fn numeric(v: &Value, key: &str, int: bool, lo: Option<f64>, hi: Option<f64>) -> Result<f64> {
    let n = match v {
        Value::Bool(b) => Some(u8::from(*b) as f64),
        Value::Number(n) => n.as_f64(),
        Value::String(s) => {
            if int {
                s.trim().parse::<i64>().ok().map(|n| n as f64)
            } else {
                s.trim().parse::<f64>().ok()
            }
        }
        _ => None,
    }
    .filter(|n| n.is_finite())
    .ok_or_else(|| Error::request(format!("{key} must be a number")))?;
    let n = if int { n.trunc() } else { n };
    if int && !(i64::MIN as f64..(i64::MAX as f64)).contains(&n) {
        return Err(Error::request(format!("{key} must be a number")));
    }
    if lo.is_some_and(|lo| n < lo) || hi.is_some_and(|hi| n > hi) {
        let bound = |v: Option<f64>| match v {
            None => "None".into(),
            Some(n) if int => format!("{}", n as i64),
            Some(0.000001) => "1e-06".into(),
            Some(n) if n.fract() == 0.0 => format!("{n:.1}"),
            Some(n) => n.to_string(),
        };
        return Err(Error::request(format!(
            "{key} must be between {} and {}",
            bound(lo),
            bound(hi)
        )));
    }
    Ok(n)
}
fn integer(v: &Value, key: &str, lo: Option<i64>) -> Result<i64> {
    let n = match v {
        Value::Number(n) if n.is_i64() => n.as_i64(),
        Value::Number(n) if n.is_u64() => n.as_u64().and_then(|n| i64::try_from(n).ok()),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => Some(numeric(v, key, true, None, None)? as i64),
    }
    .ok_or_else(|| Error::request(format!("{key} must be a number")))?;
    if lo.is_some_and(|lo| n < lo) {
        return Err(Error::request(format!(
            "{key} must be between {} and None",
            lo.unwrap_or_default()
        )));
    }
    Ok(n)
}
fn num(
    body: &Value,
    key: &str,
    default: f64,
    int: bool,
    lo: Option<f64>,
    hi: Option<f64>,
) -> Result<f64> {
    if body[key].is_null() {
        Ok(default)
    } else {
        numeric(&body[key], key, int, lo, hi)
    }
}
pub fn parse_chat_request(body: &Value, cfg: &RequestDefaults) -> Result<ChatParams> {
    if !body.is_object() {
        return Err(Error::request("the request body must be a JSON object"));
    }
    if truthy(&body["stream_options"]) && !body["stream_options"].is_object() {
        return Err(Error::request("stream_options must be an object"));
    }
    for key in ["functions", "function_call"] {
        if truthy(&body[key]) {
            return Err(Error::request(format!("{key} is not supported; use tools")));
        }
    }
    let mut tools = clean_tools(&body["tools"])?;
    match body["tool_choice"].as_str() {
        Some("none") => tools = None,
        Some("auto") => {}
        _ if body["tool_choice"].is_null() => {}
        _ => {
            return Err(Error::request(format!(
                "tool_choice {} is not supported; use \"auto\" or \"none\"",
                // As upstream: the first 80 characters of the JSON text.
                prefix(80, |out| mackani_text::write_json(
                    &body["tool_choice"],
                    true,
                    out
                ))
                .0
            )));
        }
    }
    if truthy(&body["logprobs"]) {
        return Err(Error::request("logprobs is not supported"));
    }
    if !body["n"].is_null()
        && body["n"] != json!(1)
        && body["n"] != json!(1.0)
        && body["n"] != json!(true)
    {
        return Err(Error::request("n must be 1 (one answer per request)"));
    }
    let rf = &body["response_format"];
    if rf.is_object() && !rf["type"].is_null() && rf["type"] != "text" {
        return Err(Error::request(format!(
            "response_format {} is not supported",
            excerpt_of(|out| write_repr(&rf["type"], out))
        )));
    }
    let messages = clean_messages(&body["messages"])?;
    let mt = body
        .get("max_completion_tokens")
        .unwrap_or(&body["max_tokens"]);
    let mut max_tokens = if mt.is_null() {
        cfg.max_tokens
    } else {
        integer(mt, "v", Some(1))? as usize
    };
    if let Some(cap) = cfg.max_tokens_cap.filter(|&n| n > 0) {
        max_tokens = max_tokens.min(cap);
    }
    let stop = if !truthy(&body["stop"]) {
        json!([])
    } else if body["stop"].is_string() {
        json!([body["stop"]])
    } else {
        body["stop"].clone()
    };
    let stop = stop
        .as_array()
        .filter(|a| a.len() <= 4 && a.iter().all(Value::is_string))
        .ok_or_else(|| Error::request("stop must be a string or a list of at most 4 strings"))?
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let mut kw = if !truthy(&body["chat_template_kwargs"]) {
        Map::new()
    } else {
        body["chat_template_kwargs"]
            .as_object()
            .cloned()
            .ok_or_else(|| Error::request("chat_template_kwargs must be an object"))?
    };
    // Upstream passes these on as keyword arguments too: `tools` reaches the
    // template whenever the request has no tools of its own (bypassing the
    // tool limit and tool_choice "none"), the others fail with a TypeError.
    if let Some(key) = kw.keys().find(|k| {
        mackani_text::RESERVED_VARIABLES.contains(&k.as_str()) || *k == "add_generation_prompt"
    }) {
        return Err(Error::request(format!(
            "chat_template_kwargs: {key} is set by the server and cannot be given"
        )));
    }
    if let Some(t @ ("enabled" | "disabled")) = body["thinking"]["type"].as_str() {
        kw.entry("enable_thinking").or_insert(json!(t == "enabled"));
    }
    if body["enable_thinking"].is_boolean() {
        kw.entry("enable_thinking")
            .or_insert(body["enable_thinking"].clone());
    }
    if !body["reasoning_effort"].is_null() {
        let effort = match body["reasoning_effort"].as_str() {
            Some("minimal" | "low") => "low",
            Some("medium") => "medium",
            Some("high" | "xhigh") => "xhigh",
            _ => {
                return Err(Error::request(
                    "reasoning_effort must be one of low, medium, high, xhigh",
                ));
            }
        };
        kw.entry("reasoning_effort").or_insert(json!(effort));
    }
    let nested = kw.remove("thinking_budget").unwrap_or(Value::Null);
    let tb = body.get("thinking_budget").unwrap_or(&nested);
    let thinking_budget = if tb.is_null() {
        cfg.thinking_budget
    } else {
        integer(tb, "thinking_budget", Some(0))? as usize
    };
    Ok(ChatParams {
        messages,
        max_tokens,
        temperature: num(
            body,
            "temperature",
            cfg.sampling.temperature,
            false,
            Some(0.0),
            Some(2.0),
        )?,
        top_p: num(
            body,
            "top_p",
            cfg.sampling.top_p,
            false,
            Some(1e-6),
            Some(1.0),
        )?,
        top_k: if body["top_k"].is_null() {
            cfg.sampling.top_k
        } else {
            integer(&body["top_k"], "top_k", None)?
        },
        min_p: num(
            body,
            "min_p",
            cfg.sampling.min_p,
            false,
            Some(0.0),
            Some(1.0),
        )?,
        seed: if body["seed"].is_null() {
            None
        } else {
            Some(integer(&body["seed"], "seed", None)?)
        },
        stop,
        ignore_eos: truthy(&body["ignore_eos"]),
        loop_guard: body["loop_guard"] != json!(false),
        thinking_budget,
        template_kwargs: Value::Object(kw),
        tools,
    })
}
