//! Exact CPU stages with an explicit scalar diagnostic fallback.
/// Whether the exact NEON fast paths around the projections run. Both
/// choices give identical bits; `Reference` only runs the scalar stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuStages {
    /// The fast paths when run-time NEON detection succeeds, else the reference.
    Auto,
    /// Always use the scalar reference stages.
    Reference,
}

impl CpuStages {
    /// The stable option name used in configuration and diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Reference => "reference",
        }
    }
    /// Whether the fast paths run on this CPU.
    pub fn fast(self) -> bool {
        self == Self::Auto && neon_available()
    }
}

impl std::str::FromStr for CpuStages {
    type Err = crate::Error;
    fn from_str(value: &str) -> crate::Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "reference" => Ok(Self::Reference),
            _ => Err(crate::Error::InvalidInput(
                "CPU stages must be auto or reference".into(),
            )),
        }
    }
}

fn neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub(crate) mod attention;
pub(crate) mod recurrence;
pub(crate) mod rest;

pub(crate) mod kv_codec;

pub(crate) mod math;
