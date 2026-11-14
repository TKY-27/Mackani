use super::{
    Config, DECODE_FACTOR, VALUE_COST,
    config::local,
    response::{HttpBody, error_response, http_error},
};
use crate::{
    Error, Result,
    request::{check_size, check_values, json_values, parse_chat_request, text_bytes, truthy},
};
use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::{Request, Response, body::Body};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// A request's share of `Limits::max_pending_bytes`.
pub(super) struct Charge {
    budget: Arc<Semaphore>,
    permit: tokio::sync::OwnedSemaphorePermit,
}
impl Charge {
    pub(super) fn new(budget: &Arc<Semaphore>) -> Option<Self> {
        Some(Self {
            budget: budget.clone(),
            permit: budget.clone().try_acquire_many_owned(0).ok()?,
        })
    }
    /// Grows or shrinks the charge to `n` bytes; false if the budget lacks room.
    pub(super) fn resize(&mut self, n: usize) -> bool {
        let have = self.permit.num_permits();
        if n <= have {
            drop(self.permit.split(have - n));
            return true;
        }
        let more = u32::try_from(n - have)
            .ok()
            .and_then(|k| self.budget.clone().try_acquire_many_owned(k).ok());
        more.map(|p| self.permit.merge(p)).is_some()
    }
}
pub(super) fn header<'a, B>(r: &'a Request<B>, key: &str) -> Option<&'a str> {
    r.headers().get(key).and_then(|v| v.to_str().ok())
}
pub(super) fn refused<B>(r: &Request<B>, cfg: &Config, post: bool) -> Option<&'static str> {
    if !local(&cfg.host) && cfg.api_key.as_ref().is_some_and(|s| !s.is_empty()) {
        return None;
    }
    let names = if local(&cfg.host) {
        vec!["127.0.0.1", "localhost", "[::1]"]
    } else {
        vec![cfg.host.as_str()]
    };
    let own: Vec<String> = names.iter().map(|h| format!("{h}:{}", cfg.port)).collect();
    // Duplicate security headers are ambiguous across intermediaries: refuse.
    if r.headers().get_all("host").iter().count() != 1
        || !own
            .iter()
            .any(|h| header(r, "host").is_some_and(|v| v.eq_ignore_ascii_case(h)))
    {
        return Some("host");
    }
    // An absolute-form target names its own authority: hold it to the same rule.
    let uri = r.uri();
    if let Some(authority) = uri.authority() {
        if uri.scheme_str() != Some("http")
            || !own
                .iter()
                .any(|h| authority.as_str().eq_ignore_ascii_case(h))
        {
            return Some("host");
        }
    }
    if post {
        if r.headers().get_all("origin").iter().count() > 1
            || (r.headers().contains_key("origin")
                && !own.iter().any(|h| {
                    header(r, "origin")
                        .is_some_and(|v| v.eq_ignore_ascii_case(&format!("http://{h}")))
                }))
        {
            return Some("origin");
        }
        if r.headers().get_all("sec-fetch-site").iter().count() > 1
            || (r.headers().contains_key("sec-fetch-site") && header(r, "sec-fetch-site").is_none())
            || !matches!(
                header(r, "sec-fetch-site")
                    .unwrap_or("none")
                    .to_ascii_lowercase()
                    .as_str(),
                "same-origin" | "none"
            )
        {
            return Some("cross-site");
        }
    }
    None
}
pub(super) fn authorized<B>(r: &Request<B>, key: &str) -> bool {
    if r.headers().get_all("authorization").iter().count() != 1 {
        return false;
    }
    let Some(value) = header(r, "authorization").and_then(|s| s.strip_prefix("Bearer ")) else {
        return false;
    };
    // Fixed-size digest comparison with no prefix-dependent branches. black_box
    // prevents the optimizer replacing this reduction with an early-exit memcmp.
    let a = Sha256::digest(value.as_bytes());
    let b = Sha256::digest(key.as_bytes());
    let mut diff = 0_u8;
    for i in 0..32 {
        diff |= std::hint::black_box(a[i] ^ b[i]);
    }
    std::hint::black_box(diff) == 0
}
pub(super) fn busy_error() -> Error {
    http_error(
        503,
        "the server is busy with other requests; retry later",
        "unavailable",
    )
}
pub(super) fn waited_too_long() -> Error {
    http_error(
        503,
        "the request waited too long for the engine; retry later",
        "unavailable",
    )
}
pub(super) fn busy(e: Error) -> Response<HttpBody> {
    let mut r = error_response(e);
    r.headers_mut().insert(
        "retry-after",
        hyper::header::HeaderValue::from_static(RETRY_AFTER),
    );
    r
}
const RETRY_AFTER: &str = "5";
pub(super) enum BodyError {
    TooLarge,
    Busy,
}
/// Reads the body, charging every received byte to the global budget first.
pub(super) async fn read_body<B>(
    body: &mut B,
    declared: usize,
    limit: usize,
    charge: &mut Charge,
) -> std::result::Result<Vec<u8>, BodyError>
where
    B: Body<Data = Bytes> + Unpin,
{
    // Sized once so the resident buffer matches the bytes charged (pages are
    // only touched as they are written).
    let mut out = Vec::with_capacity(declared.min(limit));
    while let Some(frame) = body.frame().await {
        let Ok(data) = frame.map_err(|_| BodyError::TooLarge)?.into_data() else {
            continue; // trailers
        };
        let len = out.len() + data.len();
        if len > limit {
            return Err(BodyError::TooLarge);
        }
        if !charge.resize(len) {
            return Err(BodyError::Busy);
        }
        out.extend_from_slice(&data);
    }
    Ok(out)
}
pub(super) struct Decoded {
    pub(super) params: crate::chat::ChatParams,
    pub(super) stream: bool,
    pub(super) include_usage: bool,
}
/// Validates and cleans a body. Before parsing, the charge grows to the most
/// the request can hold while it is decoded, so a request the budget cannot
/// hold is never built; afterwards it shrinks to the cleaned request, which
/// is all the request holds while it waits and generates. Synchronous: the
/// parsed JSON never lives across an await, and only one body is decoded at
/// a time on a current-thread runtime.
pub(super) fn decode(bytes: Vec<u8>, cfg: &Config, charge: &mut Charge) -> Result<Decoded> {
    let limits = &cfg.limits;
    let size = bytes.len();
    let estimate = |values: usize| {
        size.saturating_mul(DECODE_FACTOR)
            .saturating_add(values.saturating_mul(VALUE_COST))
    };
    let values = json_values(&bytes);
    check_values(values, limits)?;
    if !charge.resize(estimate(values)) {
        return Err(busy_error());
    }
    let body: Value = serde_json::from_slice(if bytes.is_empty() { b"{}" } else { &bytes })
        .map_err(|_| Error::request("the request body is not valid JSON"))?;
    drop(bytes);
    // Adds the values that tool-call argument strings expand to.
    let values = check_size(&body, limits, values)?;
    if !charge.resize(estimate(values)) {
        return Err(busy_error());
    }
    let params = parse_chat_request(&body, &cfg.defaults)?;
    // Everything the template can render from the request.
    let text = text_bytes(&params.messages)
        + params.tools.as_ref().map_or(0, text_bytes)
        + text_bytes(&params.template_kwargs);
    if text > limits.max_text_bytes {
        return Err(Error::request(format!(
            "the request holds more than {} bytes of text",
            limits.max_text_bytes
        )));
    }
    let decoded = Decoded {
        stream: truthy(&body["stream"]),
        include_usage: truthy(&body["stream_options"]["include_usage"]),
        params,
    };
    drop(body);
    let held = text + decoded.params.stop.iter().map(String::len).sum::<usize>();
    // Normally below the decode estimate; growth past the budget is refused.
    if !charge.resize(held.saturating_add(values.saturating_mul(VALUE_COST))) {
        return Err(busy_error());
    }
    Ok(decoded)
}
