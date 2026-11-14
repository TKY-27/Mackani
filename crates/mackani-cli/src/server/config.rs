use crate::{Error, Result, request::RequestDefaults};
use std::time::Duration;
use tokio::sync::Semaphore;

pub const MAX_BODY: usize = 8 * 1024 * 1024;
/// Heap bytes charged per counted JSON value of a request, on top of its text
/// bytes: an upper estimate for the structure of the parsed body and the
/// cleaned copy that exist together while it is decoded. Measured worst case:
/// about 590 bytes per value, for tool schemas made of one-member objects.
pub const VALUE_COST: usize = 640;
/// Text bytes a request can hold per body byte while it is decoded: the raw
/// body and the parsed JSON (unescaping only shrinks) while parsing; then the
/// parsed JSON and the cleaned copy, which Python-style `str()` of a
/// non-string text part makes up to 4 times the body (a raw DEL byte becomes
/// `\x7f`), while cleaning. The raw body is dropped once it is parsed.
pub const DECODE_FACTOR: usize = 5;
#[derive(Debug, Clone)]
pub struct Limits {
    pub max_body: usize,
    /// Global budget for request bytes being received, decoded, queued or
    /// generating; a request that would exceed it gets 503 and Retry-After.
    pub max_pending_bytes: usize,
    pub max_json_values: usize,
    pub max_messages: usize,
    pub max_content_parts: usize,
    pub max_tool_calls: usize,
    pub max_tools: usize,
    pub max_text_bytes: usize,
    /// Longest rendered prompt the engine thread tokenizes; a longer one is
    /// refused (400) between rendering and tokenization.
    pub max_prompt_bytes: usize,
    pub max_stop_bytes: usize,
    /// Longest wait for the engine, from queueing until generation starts;
    /// exceeded: 503 with Retry-After. Generation itself is bounded by
    /// `max_tokens` and ends when the client disconnects.
    pub queue_timeout: Duration,
    pub max_connections: usize,
    pub max_headers: usize,
    pub header_bytes: usize,
    pub header_timeout: Duration,
    pub body_timeout: Duration,
    /// Longest wait for the client: for request bytes (head, body, the next
    /// request on a kept-alive connection) and for a write to make progress.
    /// It does not run while a request is queued or the engine works on it.
    pub idle_timeout: Duration,
    pub stream_timeout: Duration,
    /// A stream with no output for this long sends an SSE comment line, so
    /// read timeouts of clients and intermediaries do not fire while the
    /// engine is silent (a long prefill). 10 ms to 1 hour.
    pub keepalive_interval: Duration,
    /// Ending a connection after a response half-closes first, then discards
    /// at most `linger_bytes` from the peer for at most `linger_timeout`.
    pub linger_bytes: usize,
    pub linger_timeout: Duration,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_body: MAX_BODY,
            max_pending_bytes: 512 * 1024 * 1024,
            max_json_values: 1 << 19,
            max_messages: 65536,
            max_content_parts: 1024,
            max_tool_calls: 1024,
            max_tools: 1024,
            // Tokenizing costs time and memory in proportion to the text, and a full
            // 262,144-token context of ordinary text is under 1 MiB.
            max_text_bytes: 2 * 1024 * 1024,
            max_prompt_bytes: crate::chat::MAX_PROMPT_BYTES,
            max_stop_bytes: 1024,
            queue_timeout: Duration::from_secs(600),
            max_connections: 64,
            max_headers: 100,
            header_bytes: 32768,
            header_timeout: Duration::from_secs(10),
            body_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300),
            stream_timeout: Duration::from_secs(30),
            keepalive_interval: Duration::from_secs(15),
            linger_bytes: 256 * 1024,
            linger_timeout: Duration::from_secs(2),
        }
    }
}
#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub model_id: String,
    pub defaults: RequestDefaults,
    pub limits: Limits,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8765,
            api_key: None,
            model_id: "qwen3.8-27b".into(),
            defaults: RequestDefaults::default(),
            limits: Limits::default(),
        }
    }
}
pub(super) fn local(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        if !local(&self.host) && self.api_key.as_ref().is_none_or(String::is_empty) {
            return Err(Error::request(format!(
                "refusing to listen on {} without --api-key: anyone who can reach this computer could use the engine",
                self.host
            )));
        }
        let l = &self.limits;
        // A single largest request must fit the budget when nothing else runs.
        let largest = l
            .max_body
            .saturating_mul(DECODE_FACTOR)
            .saturating_add(l.max_json_values.saturating_mul(VALUE_COST));
        if l.max_pending_bytes < largest || l.max_pending_bytes > Semaphore::MAX_PERMITS {
            return Err(Error::request("invalid server limits"));
        }
        if self.limits.max_connections == 0
            || self.limits.max_connections > 1024
            || self.limits.max_headers == 0
            || self.limits.header_bytes < 8192
            || !(Duration::from_millis(10)..=Duration::from_secs(3600))
                .contains(&self.limits.keepalive_interval)
            || self.defaults.max_tokens == 0
            || self.defaults.max_tokens_cap == Some(0)
        {
            return Err(Error::request("invalid server limits"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_transport_limits_match_the_public_security_contract() {
        let limits = Limits::default();
        assert_eq!(MAX_BODY, 8 * 1024 * 1024);
        assert_eq!(limits.max_body, MAX_BODY);
        assert_eq!(DECODE_FACTOR, 5);
        assert_eq!(VALUE_COST, 640);
        assert_eq!(limits.max_pending_bytes, 512 * 1024 * 1024);
        assert_eq!(limits.max_json_values, 524_288);
        assert_eq!(limits.max_messages, 65_536);
        assert_eq!(limits.max_content_parts, 1024);
        assert_eq!(limits.max_tool_calls, 1024);
        assert_eq!(limits.max_tools, 1024);
        assert_eq!(limits.max_text_bytes, 2 * 1024 * 1024);
        assert_eq!(limits.max_prompt_bytes, 2 * 1024 * 1024);
        assert_eq!(limits.max_stop_bytes, 1024);
        assert_eq!(limits.queue_timeout, Duration::from_secs(600));
        assert_eq!(limits.max_connections, 64);
        assert_eq!(limits.max_headers, 100);
        assert_eq!(limits.header_bytes, 32_768);
        assert_eq!(limits.header_timeout, Duration::from_secs(10));
        assert_eq!(limits.body_timeout, Duration::from_secs(30));
        assert_eq!(limits.idle_timeout, Duration::from_secs(300));
        assert_eq!(limits.stream_timeout, Duration::from_secs(30));
        assert_eq!(limits.keepalive_interval, Duration::from_secs(15));
        assert_eq!(limits.linger_bytes, 256 * 1024);
        assert_eq!(limits.linger_timeout, Duration::from_secs(2));
        let config = Config::default();
        assert_eq!(config.defaults.max_tokens, 32_768);
        assert_eq!(config.defaults.max_tokens_cap, None);
        config.validate().unwrap();
    }
}
