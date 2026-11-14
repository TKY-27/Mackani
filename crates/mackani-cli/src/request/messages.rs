use super::{format::py_string, truthy};
use crate::{Error, Result};
use serde_json::{Value, json};

fn content(v: &Value, i: usize) -> Result<Value> {
    if let Some(parts) = v.as_array() {
        let mut out = Vec::new();
        for p in parts {
            if !p.is_object()
                || (!p["type"].is_null() && p["type"] != "text")
                || p.get("text").is_none()
            {
                return Err(Error::request(format!(
                    "messages[{i}]: only text content is supported"
                )));
            }
            // Assigned, not interpolated: json! would copy the value.
            let mut part = json!({"type":"text"});
            part["text"] = Value::String(py_string(&p["text"]));
            out.push(part);
        }
        Ok(Value::Array(out))
    } else if v.is_null() {
        Ok(json!(""))
    } else if v.is_string() {
        Ok(v.clone())
    } else {
        Err(Error::request(format!(
            "messages[{i}]: content must be a string or a list of text parts"
        )))
    }
}
fn clean_calls(v: &Value, i: usize) -> Result<Value> {
    let calls = v
        .as_array()
        .ok_or_else(|| Error::request(format!("messages[{i}]: tool_calls must be a list")))?;
    let mut out = Vec::new();
    for (k, c) in calls.iter().enumerate() {
        let f = &c["function"];
        let name = f["name"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::request(format!(
                    "messages[{i}].tool_calls[{k}]: a function call with a name is expected"
                ))
            })?;
        // A string is parsed in place, not copied first.
        let mut args = match f["arguments"].as_str() {
            Some(s) if s.trim().is_empty() => json!({}),
            Some(s) => serde_json::from_str(s).map_err(|_| {
                Error::request(format!(
                    "messages[{i}].tool_calls[{k}]: arguments is not valid JSON"
                ))
            })?,
            None => f["arguments"].clone(),
        };
        if args.is_null() {
            args = json!({});
        }
        if !args.is_object() {
            return Err(Error::request(format!(
                "messages[{i}].tool_calls[{k}]: arguments must be a JSON object"
            )));
        }
        let mut function = json!({"name":name});
        function["arguments"] = args;
        let mut call = json!({"type":"function"});
        call["function"] = function;
        out.push(call);
    }
    Ok(Value::Array(out))
}
pub fn clean_messages(v: &Value) -> Result<Value> {
    let messages = v
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(|| Error::request("messages must be a non-empty list"))?;
    let mut out = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let role = match m["role"].as_str() {
            Some("developer") => "system",
            Some(s @ ("system" | "user" | "assistant" | "tool")) => s,
            _ => {
                return Err(Error::request(format!(
                    "messages[{i}]: role must be system, user, assistant or tool"
                )));
            }
        };
        let mut msg = json!({"role":role});
        msg["content"] = content(&m["content"], i)?;
        if role == "assistant" {
            if m["reasoning_content"].is_string() {
                msg["reasoning_content"] = m["reasoning_content"].clone();
            }
            if truthy(&m["tool_calls"]) {
                msg["tool_calls"] = clean_calls(&m["tool_calls"], i)?;
            }
        }
        out.push(msg);
    }
    Ok(Value::Array(out))
}
pub fn clean_tools(v: &Value) -> Result<Option<Value>> {
    if v.is_null() || v.as_array().is_some_and(Vec::is_empty) {
        return Ok(None);
    }
    let tools = v
        .as_array()
        .ok_or_else(|| Error::request("tools must be a list"))?;
    let mut out = Vec::new();
    for (i, t) in tools.iter().enumerate() {
        let f = &t["function"];
        if !t.is_object() || t.get("type").is_some_and(|x| x != "function") || !f.is_object() {
            return Err(Error::request(format!(
                "tools[{i}]: only tools of type \"function\" are supported"
            )));
        }
        let name = f["name"]
            .as_str()
            .filter(|s| !s.is_empty() && !s.contains(['<', '>', '\n']))
            .ok_or_else(|| {
                Error::request(format!(
                    "tools[{i}]: the function needs a name (without <, > or line breaks)"
                ))
            })?;
        if !f["parameters"].is_null() && !f["parameters"].is_object() {
            return Err(Error::request(format!(
                "tools[{i}]: parameters must be a JSON schema object"
            )));
        }
        let mut function = json!({"name":name});
        if f["description"].is_string() {
            function["description"] = f["description"].clone();
        }
        function["parameters"] = if truthy(&f["parameters"]) {
            f["parameters"].clone()
        } else {
            json!({"type":"object","properties":{}})
        };
        let mut tool = json!({"type":"function"});
        tool["function"] = function;
        out.push(tool);
    }
    Ok(Some(Value::Array(out)))
}
