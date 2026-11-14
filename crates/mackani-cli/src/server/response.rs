use super::{transport::WorkingGuard, worker::Cancel};
use crate::{
    Error, Result,
    chat::{Event, Usage},
};
use bytes::Bytes;
use http_body_util::{BodyExt, Full, combinators::BoxBody};
use hyper::{
    Response, StatusCode,
    body::{Body, Frame},
};
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    future::Future,
    pin::Pin,
    sync::atomic::{AtomicU64, Ordering},
    task::{Context, Poll, ready},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::mpsc,
    time::{Instant, Sleep},
};

// Every response, including errors, carries the upstream protection headers.
pub(super) type HttpBody = BoxBody<Bytes, Infallible>;
pub(super) fn response(
    status: u16,
    content_type: &'static str,
    body: HttpBody,
) -> Response<HttpBody> {
    let mut r = Response::new(body);
    *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let h = r.headers_mut();
    for (k, v) in [
        ("content-type", content_type),
        ("x-frame-options", "DENY"),
        ("content-security-policy", "frame-ancestors 'none'"),
        ("cross-origin-resource-policy", "same-origin"),
        ("x-content-type-options", "nosniff"),
    ] {
        h.insert(k, hyper::header::HeaderValue::from_static(v));
    }
    r
}
pub(super) fn json_response(status: u16, v: Value) -> Response<HttpBody> {
    let bytes = v.to_string();
    let len = bytes.len();
    let mut r = response(
        status,
        "application/json; charset=utf-8",
        Full::new(Bytes::from(bytes)).boxed(),
    );
    if let Ok(v) = hyper::header::HeaderValue::from_str(&len.to_string()) {
        r.headers_mut().insert("content-length", v);
    }
    r
}
fn error_value(e: &Error) -> Value {
    json!({"error":{"message":e.message,"type":e.kind,"code":e.code.map_or_else(||json!(e.status),|c|json!(c))}})
}
pub(super) fn error_response(e: Error) -> Response<HttpBody> {
    json_response(e.status, error_value(&e))
}
pub(super) fn http_error(status: u16, message: impl Into<String>, kind: &'static str) -> Error {
    Error {
        message: message.into(),
        status,
        kind,
        code: None,
    }
}
pub(super) fn close(mut r: Response<HttpBody>) -> Response<HttpBody> {
    r.headers_mut().insert(
        "connection",
        hyper::header::HeaderValue::from_static("close"),
    );
    r
}

fn unique(prefix: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    format!(
        "{prefix}{time:016x}{:08x}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}
pub fn usage_json(u: &Usage) -> Value {
    let mut v = json!({"prompt_tokens":u.prompt_tokens,"completion_tokens":u.completion_tokens,"total_tokens":u.prompt_tokens+u.completion_tokens,"prompt_tokens_details":{"cached_tokens":u.cached_tokens},"completion_tokens_details":{"reasoning_tokens":u.reasoning_tokens}});
    if let Some(info) = &u.loop_guard {
        v["loop_guard"] = info.clone();
    }
    v
}
pub(super) fn tool_call_json(call: Value, index: Option<usize>) -> Value {
    let mut v = json!({"id":unique("call_"),"type":"function","function":{"name":call["name"],"arguments":mackani_text::json_dumps(&call["arguments"],false)}});
    if let Some(i) = index {
        let mut indexed = json!({"index":i});
        if let Some(m) = indexed.as_object_mut() {
            if let Some(v) = v.as_object() {
                m.extend(v.clone());
            }
        }
        v = indexed;
    }
    v
}
pub(super) struct Envelope {
    id: String,
    created: u64,
    model: String,
}
impl Envelope {
    pub(super) fn new(model: &str) -> Self {
        Self {
            id: unique("chatcmpl-"),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: model.into(),
        }
    }
    pub(super) fn chunk(&self, delta: Value, finish: Option<&str>, usage: Option<Value>) -> Value {
        let choices = if usage.is_some() {
            json!([])
        } else {
            json!([{"index":0,"delta":delta,"finish_reason":finish}])
        };
        let mut v = json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":choices});
        if let Some(u) = usage {
            v["usage"] = u;
        }
        v
    }
    pub(super) fn completed(
        &self,
        reason: String,
        content: String,
        calls: Vec<Value>,
        u: Usage,
    ) -> Value {
        let mut msg = json!({"role":"assistant","content":content});
        if !reason.is_empty() {
            msg["reasoning_content"] = json!(reason);
        }
        if !calls.is_empty() {
            msg["tool_calls"] = json!(calls);
            if msg["content"] == "" {
                msg["content"] = Value::Null;
            }
        }
        json!({"id":self.id,"object":"chat.completion","created":self.created,"model":self.model,"choices":[{"index":0,"message":msg,"finish_reason":u.finish}],"usage":usage_json(&u)})
    }
}
pub(super) fn sse(v: Value) -> Bytes {
    Bytes::from(format!("data: {v}\n\n"))
}

const KEEPALIVE: &[u8] = b": keep-alive\n\n";
pub(super) struct EventBody {
    pub(super) events: mpsc::Receiver<Result<Event>>,
    pub(super) _cancel: Cancel,
    // Held until the stream ends or the connection drops it.
    pub(super) _working: Option<WorkingGuard>,
    pub(super) envelope: Envelope,
    pub(super) include_usage: bool,
    pub(super) calls: usize,
    pub(super) pending: std::collections::VecDeque<Bytes>,
    pub(super) ended: bool,
    /// Fires after `interval` without output. Hyper polls the body only when
    /// it can buffer more, so a client that does not read gets no more
    /// comments than the bytes Hyper holds for it anyway.
    pub(super) keepalive: Pin<Box<Sleep>>,
    pub(super) interval: Duration,
}
impl Body for EventBody {
    type Data = Bytes;
    type Error = Infallible;
    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<std::result::Result<Frame<Bytes>, Infallible>>> {
        let this = self.get_mut();
        let bytes = match this.poll_output(cx) {
            Poll::Ready(bytes) => bytes,
            Poll::Pending => {
                ready!(this.keepalive.as_mut().poll(cx));
                Some(Bytes::from_static(KEEPALIVE))
            }
        };
        this.keepalive
            .as_mut()
            .reset(Instant::now() + this.interval);
        Poll::Ready(bytes.map(|b| Ok(Frame::data(b))))
    }
}
impl EventBody {
    fn poll_output(&mut self, cx: &mut Context<'_>) -> Poll<Option<Bytes>> {
        let this = self;
        if let Some(bytes) = this.pending.pop_front() {
            return Poll::Ready(Some(bytes));
        }
        if this.ended {
            return Poll::Ready(None);
        }
        match this.events.poll_recv(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(event) => {
                match event {
                    Some(Ok(Event::Reasoning(t))) => this.pending.push_back(sse(this
                        .envelope
                        .chunk(json!({"reasoning_content":t}), None, None))),
                    Some(Ok(Event::Content(t))) => this
                        .pending
                        .push_back(sse(this.envelope.chunk(json!({"content":t}), None, None))),
                    Some(Ok(Event::ToolCall(c))) => {
                        let c = tool_call_json(c, Some(this.calls));
                        this.calls += 1;
                        this.pending.push_back(sse(this.envelope.chunk(
                            json!({"tool_calls":[c]}),
                            None,
                            None,
                        )));
                    }
                    Some(Ok(Event::Done(u))) => {
                        this.pending.push_back(sse(this.envelope.chunk(
                            json!({}),
                            Some(u.finish),
                            None,
                        )));
                        if this.include_usage {
                            this.pending.push_back(sse(this.envelope.chunk(
                                json!({}),
                                None,
                                Some(usage_json(&u)),
                            )));
                        }
                        this.pending
                            .push_back(Bytes::from_static(b"data: [DONE]\n\n"));
                        this.ended = true;
                    }
                    Some(Err(e)) => {
                        this.pending.push_back(sse(error_value(&e)));
                        this.pending
                            .push_back(Bytes::from_static(b"data: [DONE]\n\n"));
                        this.ended = true;
                    }
                    None => {
                        this.pending.push_back(sse(error_value(&Error::engine(
                            "engine worker ended without a completion",
                        ))));
                        this.pending
                            .push_back(Bytes::from_static(b"data: [DONE]\n\n"));
                        this.ended = true;
                    }
                }
                Poll::Ready(this.pending.pop_front())
            }
        }
    }
}
