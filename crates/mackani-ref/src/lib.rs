//! Scalar implementation of [chapters 01–06](../../../docs/spec/01_MODEL_ARCHITECTURE.md).
#![forbid(unsafe_code)]

pub mod attention;
pub mod compressed_linear;
pub mod draft;
mod draft_depth;
pub mod draft_head;
mod draft_proposer;
pub mod gated_deltanet;
pub mod kv_codec;
pub mod kv_profile;
pub mod model;
pub mod output_head_sampler;
pub mod portable_math;
pub mod quality;
pub mod residual_norm_mlp;
pub mod speculation;

// The shared test support names this crate as `mackani_ref`, as integration tests do.
#[cfg(test)]
extern crate self as mackani_ref;

#[cfg(test)]
#[allow(dead_code)]
#[path = "../tests/support/mod.rs"]
mod test_support;

use std::fmt;

pub use attention::KvMode;
pub use model::{Model, ModelDimensions, Snapshot};
pub use output_head_sampler::{Sampler, SamplingParameters, SamplingVersion};

/// A fallible operation using this crate's error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Failures encountered while validating or executing model data.
#[derive(Debug)]
pub enum Error {
    Format(mackani_format::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidModel(String),
    InvalidInput(String),
    NonFinite(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::InvalidModel(message) => write!(formatter, "invalid model: {message}"),
            Self::InvalidInput(message) => write!(formatter, "invalid input: {message}"),
            Self::NonFinite(stage) => write!(formatter, "non-finite value in {stage}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<mackani_format::Error> for Error {
    fn from(error: mackani_format::Error) -> Self {
        Self::Format(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

pub(crate) fn require_finite(values: &[f32], stage: &'static str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFinite(stage));
    }
    Ok(())
}
