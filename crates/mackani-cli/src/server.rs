//! HTTP/1 routing for the local chat API.
use crate::{Error, chat::Event, request::excerpt};
use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::{Method, Request, Response, body::Body};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::oneshot;

mod admission;
mod config;
mod response;
mod transport;
mod worker;

pub use config::{Config, DECODE_FACTOR, Limits, MAX_BODY, VALUE_COST};
pub use response::usage_json;
pub use transport::Server;
pub use worker::Api;

use admission::{
    BodyError, Charge, authorized, busy, busy_error, decode, header, read_body, refused,
    waited_too_long,
};
use response::{
    Envelope, EventBody, HttpBody, close, error_response, http_error, json_response, response, sse,
    tool_call_json,
};
use transport::Working;
use worker::{Cancel, Job};

async fn handle<B>(mut r: Request<B>, api: &Api) -> Response<HttpBody>
where
    B: Body<Data = Bytes> + Unpin,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    let shared = &api.shared;
    let cfg = &shared.cfg;
    let post = r.method() == Method::POST;
    if let Some(why) = refused(&r, cfg, post) {
        return close(error_response(http_error(
            403,
            format!("forbidden: {why}"),
            "forbidden",
        )));
    }
    let path = r.uri().path();
    if r.method() == Method::GET && path == "/health" {
        let loaded = shared.loaded.load(Ordering::Acquire);
        let mut body = json!({"status":if loaded{"ok"}else{"loading"},"version":env!("CARGO_PKG_VERSION"),"model":cfg.model_id});
        if loaded {
            body["max_context_tokens"] = json!(shared.capacity.load(Ordering::Acquire));
            body["thinking_budget"] = json!(cfg.defaults.thinking_budget);
        }
        return json_response(if loaded { 200 } else { 503 }, body);
    }
    if let Some(key) = cfg.api_key.as_ref().filter(|k| !k.is_empty()) {
        if !authorized(&r, key) {
            return close(error_response(http_error(
                401,
                "invalid or missing API key",
                "authentication_error",
            )));
        }
    }
    if r.method() == Method::GET {
        let model = json!({"id":cfg.model_id,"object":"model","owned_by":"local"});
        return if matches!(path, "/v1/models" | "/v1/models/") {
            json_response(200, json!({"object":"list","data":[model]}))
        } else if path == format!("/v1/models/{}", cfg.model_id) {
            json_response(200, model)
        } else {
            error_response(http_error(404, "not found", "not_found"))
        };
    }
    if !post {
        return close(error_response(http_error(
            501,
            format!("Unsupported method ('{}')", excerpt(r.method().as_str())),
            "invalid_request_error",
        )));
    }
    if path != "/v1/chat/completions" {
        return close(error_response(http_error(404, "not found", "not_found")));
    }
    if !shared.loaded.load(Ordering::Acquire) {
        return close(error_response(http_error(
            503,
            "the model is still loading",
            "unavailable",
        )));
    }
    // Upstream requires Content-Length and doesn't implement chunked requests.
    let len = header(&r, "content-length").and_then(|s| s.parse::<usize>().ok());
    if len.is_some_and(|n| n > cfg.limits.max_body) || r.headers().contains_key("transfer-encoding")
    {
        return close(error_response(http_error(
            413,
            "request body too large or length missing",
            "invalid_request_error",
        )));
    }
    let Some(mut charge) = Charge::new(&shared.budget) else {
        return close(busy(busy_error()));
    };
    let bytes = match tokio::time::timeout(
        cfg.limits.body_timeout,
        read_body(
            r.body_mut(),
            len.unwrap_or(0),
            cfg.limits.max_body,
            &mut charge,
        ),
    )
    .await
    {
        Err(_) => {
            return close(error_response(http_error(
                408,
                "request body timed out",
                "invalid_request_error",
            )));
        }
        Ok(Err(BodyError::Busy)) => return close(busy(busy_error())),
        Ok(Err(BodyError::TooLarge)) => {
            return close(error_response(http_error(
                413,
                "request body too large or length missing",
                "invalid_request_error",
            )));
        }
        Ok(Ok(b)) => b,
    };
    // The client has sent everything: until the response is complete the
    // idle deadline does not apply to reads (see `IdleIo`).
    let working = r.extensions().get::<Working>().map(Working::start);
    let decoded = match decode(bytes, cfg, &mut charge) {
        Ok(d) => d,
        Err(e) if e.status == 503 => return busy(e),
        Err(e) => return error_response(e),
    };
    let envelope = Envelope::new(&cfg.model_id);
    let cancel = Cancel(Arc::new(AtomicBool::new(false)));
    let (ready, rx) = oneshot::channel();
    let Some(jobs) = api.jobs.as_ref() else {
        return error_response(Error::engine("engine worker is unavailable"));
    };
    let job = Job {
        params: decoded.params,
        ready,
        cancel: cancel.0.clone(),
        deadline: std::time::Instant::now() + cfg.limits.queue_timeout,
        _charge: charge,
    };
    // On timeout or disconnect `cancel` drops, and the engine thread skips
    // the job when it reaches it, or stops preparing it.
    let waited = tokio::time::timeout(cfg.limits.queue_timeout, async {
        jobs.send(job).await.ok()?;
        rx.await.ok()
    })
    .await;
    let events = match waited {
        Err(_) => return busy(waited_too_long()),
        Ok(Some(Ok(rx))) => rx,
        Ok(Some(Err(e))) if e.status == 503 => return busy(e),
        Ok(Some(Err(e))) => return error_response(e),
        Ok(None) => return error_response(Error::engine("engine worker is unavailable")),
    };
    if decoded.stream {
        let stream = EventBody {
            events,
            _cancel: cancel,
            _working: working,
            envelope,
            include_usage: decoded.include_usage,
            calls: 0,
            pending: std::collections::VecDeque::new(),
            ended: false,
            keepalive: Box::pin(tokio::time::sleep(cfg.limits.keepalive_interval)),
            interval: cfg.limits.keepalive_interval,
        };
        let mut stream = stream;
        stream.pending.push_back(sse(stream.envelope.chunk(
            json!({"role":"assistant"}),
            None,
            None,
        )));
        let mut r = response(200, "text/event-stream; charset=utf-8", stream.boxed());
        r.headers_mut().insert(
            "cache-control",
            hyper::header::HeaderValue::from_static("no-cache"),
        );
        return close(r);
    }
    let mut events = events;
    let mut reason = String::new();
    let mut content = String::new();
    let mut calls = vec![];
    while let Some(event) = events.recv().await {
        match event {
            Err(e) => return error_response(e),
            Ok(Event::Reasoning(t)) => reason.push_str(&t),
            Ok(Event::Content(t)) => content.push_str(&t),
            Ok(Event::ToolCall(c)) => calls.push(tool_call_json(c, None)),
            Ok(Event::Done(u)) => {
                return json_response(200, envelope.completed(reason, content, calls, u));
            }
        }
    }
    drop(cancel);
    error_response(Error::engine("engine worker ended without a completion"))
}
