use super::help;
use crate::{
    Error, Result,
    chat::ChatRunner,
    engine::Engine,
    real_engine::{DraftMode, RealEngineConfig, load_with_tokenizer},
    request::RequestDefaults,
    stub::{self, StubEngine},
};
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
pub(super) struct Args {
    pub(super) command: String,
    pub(super) values: HashMap<String, String>,
    pub(super) flags: HashSet<String>,
    pub(super) comparisons: HashMap<String, Vec<String>>,
}
impl Args {
    pub(super) fn value(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
    pub(super) fn flag(&self, key: &str) -> bool {
        self.flags.contains(key)
    }
    pub(super) fn parse<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T> {
        self.value(key).map_or(Ok(default), |s| {
            s.parse()
                .map_err(|_| Error::request(format!("--{key} has an invalid value")))
        })
    }
    pub(super) fn capacity(&self) -> Result<usize> {
        let n = self.parse("ctx", 32768_usize)?;
        if n < crate::memory::SMALLEST_CONTEXT {
            return Err(Error::request(format!(
                "--ctx must be greater than {}",
                crate::memory::SMALLEST_CONTEXT - 1
            )));
        }
        Ok(n)
    }
    pub(super) fn is_stub(&self) -> bool {
        self.value("engine") == Some("stub")
    }
    pub(super) fn model_dir(&self) -> Result<std::path::PathBuf> {
        self.value("model-dir")
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("MACKANI_MODEL_DIR").map(Into::into))
            .ok_or_else(|| {
                Error::request(
                    "--model-dir is required for the real engine (or set MACKANI_MODEL_DIR)",
                )
            })
    }
    pub(super) fn real_config(&self) -> Result<RealEngineConfig> {
        let defaults = RealEngineConfig::default();
        let draft = match self.value("draft").unwrap_or(defaults.draft.name()) {
            "none" => DraftMode::None,
            "ngram" => DraftMode::Ngram,
            "neural" => DraftMode::Neural,
            "both" => DraftMode::Both,
            _ => {
                return Err(Error::request(
                    "--draft must be none, ngram, neural or both",
                ));
            }
        };
        for (key, expected) in [("kernel", "portable"), ("cpu-stages", "reference")] {
            if let Some(value) = self.value(key) {
                if value != expected {
                    return Err(Error::request(format!(
                        "--{key}={value}: expected {expected}"
                    )));
                }
            }
        }
        Ok(RealEngineConfig {
            model_dir: self.model_dir()?,
            kv: mackani_ref::quality::parse_kv_mode(std::ffi::OsStr::new(
                self.value("kv").unwrap_or("kv5"),
            ))
            .map_err(Error::engine)?,
            backend: self.switch("backend", parse)?,
            kernel: self.switch("kernel", parse)?,
            cpu_stages: self
                .switch("cpu-stages", parse)?
                .unwrap_or(defaults.cpu_stages),
            metal: mackani_engine::MetalOptions {
                cpu_share: self
                    .switch("cpu-share", parse)?
                    .unwrap_or(defaults.metal.cpu_share),
            },
            draft,
            k: self.parse("k", defaults.k)?,
            depth: self.value("depth").map_or(Ok(defaults.depth), str::parse)?,
            context: self.capacity()?,
            threads: self
                .value("threads")
                .map(|_| self.parse("threads", 1))
                .transpose()?,
            prefill_width: self.parse("prefill-width", 16)?,
        })
    }
    fn switch<T>(
        &self,
        key: &str,
        parse: impl Fn(&str) -> std::result::Result<T, String>,
    ) -> Result<Option<T>> {
        self.value(key)
            .map(|value| parse(value).map_err(|e| Error::request(format!("--{key}={value}: {e}"))))
            .transpose()
    }
    pub(super) fn defaults(&self) -> Result<RequestDefaults> {
        let mut defaults = RequestDefaults::default();
        if !self.is_stub() {
            defaults.sampling = self.real_config()?.sampling_defaults()?;
        }
        Ok(defaults)
    }
    pub(super) fn runner(&self) -> Result<ChatRunner> {
        let (engine, fmt): (Box<dyn Engine>, _) = if self.is_stub() {
            let fmt = if let Some(path) = self.value("model-dir") {
                mackani_text::ChatFormatter::load(path)?
            } else {
                stub::formatter()?
            };
            (
                Box::new(StubEngine::new(self.capacity()?).using_formatter(fmt.clone())),
                fmt,
            )
        } else {
            // Reject unsupported or oversized configurations before allocating model data.
            let (engine, fmt) = load_with_tokenizer(self.real_config()?, None)?;
            (Box::new(engine), fmt)
        };
        ChatRunner::new(engine, fmt, self.parse("seed0", 1)?)
    }
}
fn parse<T: std::str::FromStr<Err = mackani_engine::Error>>(
    value: &str,
) -> std::result::Result<T, String> {
    value
        .parse()
        .map_err(|e: mackani_engine::Error| e.to_string())
}
pub(super) fn parse_args(argv: Vec<String>) -> Result<Args> {
    let mut it = argv.into_iter();
    let command = it.next().ok_or_else(|| Error::request(help::text(None)))?;
    if !matches!(
        command.as_str(),
        "serve" | "chat" | "info" | "bench" | "selfcheck"
    ) {
        return Err(Error::request(format!(
            "unknown command: {command}\n{}",
            help::text(None)
        )));
    }
    let common = [
        "engine",
        "model-dir",
        "ctx",
        "seed0",
        "kv",
        "backend",
        "draft",
        "k",
        "depth",
        "threads",
        "prefill-width",
        "kernel",
        "cpu-stages",
        "cpu-share",
    ];
    let serve = [
        "host",
        "port",
        "api-key",
        "default-max-tokens",
        "max-tokens-cap",
    ];
    let chat = [
        "system",
        "max-tokens",
        "temperature",
        "top-p",
        "top-k",
        "min-p",
        "seed",
        "effort",
    ];
    let mut values = HashMap::new();
    let mut flags = HashSet::new();
    let mut comparisons: HashMap<String, Vec<String>> = HashMap::new();
    while let Some(arg) = it.next() {
        let Some(arg) = arg.strip_prefix("--") else {
            return Err(Error::request(format!("unexpected argument: {arg}")));
        };
        let (key, inline) = arg
            .split_once('=')
            .map_or((arg, None), |(k, v)| (k, Some(v.to_owned())));
        let is_flag = (command == "serve" && matches!(key, "no-warmup" | "exit-when-stdin-closes"))
            || (command == "chat"
                && matches!(key, "no-think" | "hide-thinking" | "no-loop-guard" | "raw"));
        if is_flag && inline.is_none() {
            flags.insert(key.into());
            continue;
        }
        if !common.contains(&key)
            && !(command == "serve" && serve.contains(&key))
            && !(command == "chat" && chat.contains(&key))
            && !(command == "bench" && matches!(key, "n" | "json"))
            && !(command == "selfcheck" && key == "reference")
        {
            return Err(Error::request(format!("unknown option: --{key}")));
        }
        let value = inline
            .or_else(|| it.next())
            .ok_or_else(|| Error::request(format!("--{key} requires a value")))?;
        if value.starts_with("--") {
            return Err(Error::request(format!("--{key} requires a value")));
        }
        if command == "bench" && matches!(key, "draft" | "k" | "depth") {
            comparisons
                .entry(key.into())
                .or_default()
                .push(value.clone());
        }
        values.insert(key.into(), value);
    }
    Ok(Args {
        command,
        values,
        flags,
        comparisons,
    })
}
