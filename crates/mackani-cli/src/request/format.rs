use serde_json::Value;
use std::fmt::{self, Write as _};

pub const EXCERPT: usize = 64;
/// `text` cut to [`EXCERPT`] characters, with an ellipsis if it was longer.
pub fn excerpt(text: &str) -> String {
    excerpt_of(|out| out.write_str(text))
}
/// Keeps the first `left` characters written into it, then refuses more:
/// its error stops the writer, so quoting a huge value costs no more than
/// the excerpt, with no copy of the value.
pub(super) struct Prefix {
    text: String,
    left: usize,
    cut: bool,
}
impl fmt::Write for Prefix {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for c in s.chars() {
            if self.left == 0 {
                self.cut = true;
                return Err(fmt::Error);
            }
            self.text.push(c);
            self.left -= 1;
        }
        Ok(())
    }
}
/// The first `n` characters `write` produces, and whether it had more.
pub(super) fn prefix(n: usize, write: impl FnOnce(&mut Prefix) -> fmt::Result) -> (String, bool) {
    let mut out = Prefix {
        text: String::new(),
        left: n,
        cut: false,
    };
    // The writers used here fail only when `out` stops them, which `cut` records.
    let _ = write(&mut out);
    (out.text, out.cut)
}
/// [`EXCERPT`] characters of what `write` produces, with an ellipsis if it had more.
pub(super) fn excerpt_of(write: impl FnOnce(&mut Prefix) -> fmt::Result) -> String {
    let (mut out, cut) = prefix(EXCERPT, write);
    if cut {
        out.push('…');
    }
    out
}
pub(super) fn py_string(v: &Value) -> String {
    if let Value::String(s) = v {
        s.clone()
    } else {
        py_repr(v)
    }
}
/// Counts the bytes written, to size a buffer before writing into it.
struct Len(usize);
impl fmt::Write for Len {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0 += s.len();
        Ok(())
    }
}
fn py_repr(v: &Value) -> String {
    // Sized exactly by a first pass: a buffer grown by doubling briefly holds
    // its old and new copy, up to twice the result. Neither writer can fail.
    let mut len = Len(0);
    let _ = write_repr(v, &mut len);
    let mut out = String::with_capacity(len.0);
    let _ = write_repr(v, &mut out);
    out
}
/// Python's repr() of `v`, written into one buffer: nested values are not
/// copied once per level, so the result is the only large allocation.
pub(super) fn write_repr(v: &Value, out: &mut impl fmt::Write) -> fmt::Result {
    match v {
        Value::String(s) => write_str_repr(s, out),
        Value::Null => out.write_str("None"),
        Value::Bool(true) => out.write_str("True"),
        Value::Bool(false) => out.write_str("False"),
        Value::Array(a) => {
            out.write_char('[')?;
            for (i, v) in a.iter().enumerate() {
                if i > 0 {
                    out.write_str(", ")?;
                }
                write_repr(v, out)?;
            }
            out.write_char(']')
        }
        Value::Object(m) => {
            out.write_char('{')?;
            for (i, (k, v)) in m.iter().enumerate() {
                if i > 0 {
                    out.write_str(", ")?;
                }
                write_str_repr(k, out)?;
                out.write_str(": ")?;
                write_repr(v, out)?;
            }
            out.write_char('}')
        }
        _ => out.write_str(&mackani_text::json_dumps(v, false)),
    }
}
fn write_str_repr(s: &str, out: &mut impl fmt::Write) -> fmt::Result {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.write_char(quote)?;
    for c in s.chars() {
        match c {
            '\\' => out.write_str("\\\\")?,
            '\n' => out.write_str("\\n")?,
            '\r' => out.write_str("\\r")?,
            '\t' => out.write_str("\\t")?,
            c if c == quote => {
                out.write_char('\\')?;
                out.write_char(c)?;
            }
            c if c.is_control() && (c as u32) <= 255 => write!(out, "\\x{:02x}", c as u32)?,
            c => out.write_char(c)?,
        }
    }
    out.write_char(quote)
}
