use super::args::Args;
use crate::{
    Error, Result,
    chat::{ChatParams, ChatRunner, Event},
    request::parse_chat_request,
};
use serde_json::json;
use std::{
    io::{BufRead, Write},
    sync::atomic::{AtomicBool, Ordering},
};

pub const CHAT_HELP: &str = "commands:  /exit  quit      /reset  forget the conversation      /think on|off  thinking on or off (default on)\n           /effort low|medium|xhigh  how long the model thinks      /help  this text      Ctrl+C while it writes: stop the answer";
pub(super) fn chat_params(args: &Args) -> Result<ChatParams> {
    let mut body = json!({"messages":[{"role":"user","content":""}],"enable_thinking":!args.flag("no-think"),"loop_guard":!args.flag("no-loop-guard")});
    for key in [
        "max-tokens",
        "temperature",
        "top-p",
        "top-k",
        "min-p",
        "seed",
    ] {
        if let Some(v) = args.value(key) {
            body[key.replace('-', "_")] = json!(v);
        }
    }
    if let Some(e) = args.value("effort") {
        if !matches!(e, "low" | "medium" | "xhigh") {
            return Err(Error::request("--effort must be low, medium or xhigh"));
        }
        body["reasoning_effort"] = json!(e);
    }
    let mut p = parse_chat_request(&body, &args.defaults()?)?;
    p.messages = if let Some(system) = args.value("system") {
        json!([{"role":"system","content":system}])
    } else {
        json!([])
    };
    Ok(p)
}
/// Shows C0 controls other than newline and tab, DEL and C1 controls as
/// `\xNN`, and bidirectional overrides, isolates and marks as `\uNNNN`, so
/// text written to a terminal cannot drive it or reorder what it shows.
/// Joiners and variation selectors stay: emoji need them.
pub fn visible(text: &str) -> std::borrow::Cow<'_, str> {
    let control = |c: char| c.is_control() && c != '\n' && c != '\t';
    let bidi = |c: char| {
        matches!(
            c,
            '\u{61c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}'
        )
    };
    if !text.contains(|c| control(c) || bidi(c)) {
        return text.into();
    }
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        if control(c) {
            out.push_str(&format!("\\x{:02x}", c as u32));
        } else if bidi(c) {
            out.push_str(&format!("\\u{:04x}", c as u32));
        } else {
            out.push(c);
        }
    }
    out.into()
}
/// Plain line-oriented chat, usable with any buffered input/output (including
/// pipes). Model text and errors are passed through [`visible`] unless `raw`.
#[allow(clippy::too_many_arguments)]
pub fn chat_loop(
    runner: &mut ChatRunner,
    mut p: ChatParams,
    hide: bool,
    raw: bool,
    mut input: impl BufRead,
    mut output: impl Write,
    writing: &AtomicBool,
    cancel: &AtomicBool,
) -> Result<()> {
    let base = p.messages.as_array().cloned().unwrap_or_default();
    let shown = |text: &str| -> String {
        if raw {
            text.into()
        } else {
            visible(text).into_owned()
        }
    };
    let mut messages = base.clone();
    writeln!(
        output,
        "mackani chat - type /help for commands, /exit to quit."
    )
    .map_err(Error::engine)?;
    loop {
        write!(output, "\n> ")
            .and_then(|_| output.flush())
            .map_err(Error::engine)?;
        let mut line = String::new();
        if input.read_line(&mut line).map_err(Error::engine)? == 0 {
            writeln!(output).map_err(Error::engine)?;
            return Ok(());
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(command) = line.strip_prefix('/') {
            let (cmd, rest) = command.split_once(' ').unwrap_or((command, ""));
            let rest = rest.trim().to_ascii_lowercase();
            match cmd {
                "exit" | "quit" | "q" => return Ok(()),
                "reset" => {
                    messages = base.clone();
                    writeln!(output, "(conversation cleared)").map_err(Error::engine)?;
                }
                "think" if matches!(rest.as_str(), "on" | "off") => {
                    p.template_kwargs["enable_thinking"] = json!(rest == "on");
                    writeln!(output, "(thinking {rest})").map_err(Error::engine)?;
                }
                "effort" if matches!(rest.as_str(), "low" | "medium" | "xhigh") => {
                    p.template_kwargs["reasoning_effort"] = json!(rest);
                    writeln!(output, "(reasoning effort {rest})").map_err(Error::engine)?;
                }
                _ => writeln!(output, "{CHAT_HELP}").map_err(Error::engine)?,
            }
            continue;
        }
        messages.push(json!({"role":"user","content":line}));
        p.messages = json!(messages);
        let mut reason = String::new();
        let mut content = String::new();
        let mut last = "";
        cancel.store(false, Ordering::Release);
        writing.store(true, Ordering::Release);
        let prepared = runner.prepare_checked(&p);
        let result = prepared.and_then(|prepared| {
            runner.run_prepared(
                &p,
                prepared,
                &|| cancel.load(Ordering::Acquire),
                &mut |event| {
                    let (kind, text) = match event {
                        Event::Reasoning(t) => {
                            reason.push_str(&t);
                            if hide {
                                return Ok(());
                            }
                            ("reasoning", t)
                        }
                        Event::Content(t) => {
                            content.push_str(&t);
                            ("content", t)
                        }
                        _ => return Ok(()),
                    };
                    if last != kind {
                        if kind == "reasoning" {
                            writeln!(output, "[thinking]").map_err(Error::engine)?;
                        } else if last == "reasoning" {
                            writeln!(output, "\n[/thinking]\n").map_err(Error::engine)?;
                        }
                        last = kind;
                    }
                    write!(output, "{}", shown(&text))
                        .and_then(|_| output.flush())
                        .map_err(Error::engine)
                },
            )
        });
        writing.store(false, Ordering::Release);
        if cancel.load(Ordering::Acquire) {
            writeln!(output, "\n[stopped]").map_err(Error::engine)?;
            messages.pop();
            continue;
        }
        let u = match result {
            Ok(u) => u,
            Err(e) => {
                writeln!(
                    output,
                    "{}: {}",
                    if e.status == 400 {
                        "cannot do that"
                    } else {
                        "\nengine error"
                    },
                    shown(&e.message)
                )
                .map_err(Error::engine)?;
                messages.pop();
                if e.status != 400 {
                    runner.engine.reset()?;
                }
                continue;
            }
        };
        if !reason.is_empty() && content.is_empty() && u.finish != "stop" {
            writeln!(output,"\n(the limit of {} tokens was reached while the model was still thinking, so no answer was written.  /effort medium or /effort low shortens the thinking, /think off skips it, --max-tokens raises the limit)",p.max_tokens).map_err(Error::engine)?;
        }
        let tps = if u.gen_ms > 0.0 {
            u.completion_tokens as f64 / (u.gen_ms / 1000.0)
        } else {
            0.0
        };
        writeln!(
            output,
            "\n\n[{} tokens, {tps:.0} tokens/s, {}{}]",
            u.completion_tokens,
            if u.finish == "stop" {
                "complete"
            } else {
                "hit the length limit"
            },
            if u.loop_guard.is_some() {
                ", loop guard cut the thinking"
            } else {
                ""
            }
        )
        .map_err(Error::engine)?;
        let mut msg = json!({"role":"assistant","content":content.trim()});
        if !reason.is_empty() {
            msg["reasoning_content"] = json!(reason.trim());
        }
        messages.push(msg);
    }
}
