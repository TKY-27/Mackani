//! Projection device selection and persistent execution resources.
use super::*;

/// Where compressed projections run (their exact i64 sums are identical);
/// everything else runs on the CPU pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
}
impl Backend {
    /// The stable option name used in configuration and diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
    /// Prefer Metal when a device exists, otherwise use the CPU.
    pub fn detected() -> Self {
        #[cfg(target_os = "macos")]
        if MetalEngine::device_exists() {
            return Self::Metal;
        }
        Self::Cpu
    }
}
impl std::str::FromStr for Backend {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            _ => Err(Error::InvalidInput(format!(
                "unknown backend {value}; expected cpu or metal"
            ))),
        }
    }
}

/// The CPU kernel's share of each Metal projection's rows, computed while the
/// GPU computes the rest. The sums are exact whatever the share.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CpuShare {
    /// The calibrated table, but only on the machine it was measured on (an
    /// Apple M5 with 4+6 CPU cores and 10 GPU cores, with 10 CPU workers); off on
    /// every other machine and where those facts cannot be read.
    Auto,
    Off,
    /// One fraction in (0, 0.9] for every row count, e.g. from `bench_metal --shares`.
    Fixed(f64),
}
impl CpuShare {
    /// Largest accepted fixed share.
    pub const MAX: f64 = 0.9;
    /// The stable option name used in configuration and diagnostics.
    pub fn name(self) -> String {
        match self {
            Self::Auto => "auto".into(),
            Self::Off => "off".into(),
            Self::Fixed(share) => share.to_string(),
        }
    }
    /// A fraction in 0..=0.9; 0 is [`Self::Off`].
    pub fn fixed(share: f64) -> Result<Self> {
        if share == 0.0 {
            Ok(Self::Off)
        } else if (0.0..=Self::MAX).contains(&share) {
            Ok(Self::Fixed(share))
        } else {
            Err(Error::InvalidInput(
                "the CPU share must be within 0..=0.9".into(),
            ))
        }
    }
}
impl std::str::FromStr for CpuShare {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            _ => value
                .parse::<f64>()
                .ok()
                .and_then(|share| Self::fixed(share).ok())
                .ok_or_else(|| {
                    Error::InvalidInput("the CPU share must be auto, off or 0..=0.9".into())
                }),
        }
    }
}

/// The CPU share of exact Metal projection work.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetalOptions {
    pub cpu_share: CpuShare,
}
impl Default for MetalOptions {
    fn default() -> Self {
        Self {
            cpu_share: CpuShare::Auto,
        }
    }
}

/// Projection backend and diagnostic fallbacks for one model or draft.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExecutionOptions {
    /// Where compressed projections run.
    pub backend: Backend,
    /// Use vectorized CPU stages or the scalar diagnostic fallback.
    pub cpu_stages: crate::CpuStages,
    /// How projection rows are shared between Metal and the CPU.
    pub metal: MetalOptions,
}
impl Default for ExecutionOptions {
    fn default() -> Self {
        Self {
            backend: Backend::detected(),
            cpu_stages: crate::CpuStages::Auto,
            metal: MetalOptions::default(),
        }
    }
}

/// The CPU pool (all row/head work and the CPU kernel) plus the selected projector.
pub(crate) struct Compute {
    engine: Engine,
    /// Whether this model dispatches vectorized CPU stages.
    pub(crate) cpu_stages: bool,
    #[cfg(target_os = "macos")]
    metal: Option<MetalEngine>,
}
impl Compute {
    pub(crate) fn new(engine: Engine, backend: Backend) -> Result<Self> {
        #[cfg(not(target_os = "macos"))]
        if backend == Backend::Metal {
            return Err(Error::InvalidInput(
                "the Metal backend requires macOS".into(),
            ));
        }
        Ok(Self {
            #[cfg(target_os = "macos")]
            metal: match backend {
                Backend::Cpu => None,
                Backend::Metal => Some(
                    MetalEngine::new()?
                        .ok_or_else(|| Error::InvalidInput("no Metal device".into()))?,
                ),
            },
            cpu_stages: crate::CpuStages::Auto.fast(),
            engine,
        })
    }
    /// Apply explicit Metal options; nothing on the CPU backend.
    pub(crate) fn set_metal_options(&mut self, options: MetalOptions) {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &mut self.metal {
            metal.set_options(options);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = options;
    }
    pub(crate) fn metal_options(&self) -> Option<MetalOptions> {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &self.metal {
            return Some(metal.options());
        }
        None
    }
    pub(crate) fn threads(&self) -> usize {
        self.engine.threads()
    }
    pub(crate) fn backend(&self) -> Backend {
        #[cfg(target_os = "macos")]
        if self.metal.is_some() {
            return Backend::Metal;
        }
        Backend::Cpu
    }
    /// The projection device: the Metal device name, or the CPU kernel and threads.
    pub(crate) fn device(&self) -> String {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &self.metal {
            return metal.device_name();
        }
        format!("{} x{}", self.engine.kernel().name(), self.engine.threads())
    }
    pub(crate) fn kernel(&self) -> crate::Kernel {
        self.engine.kernel()
    }
    pub(crate) fn map_range<T: Send, F: Fn(usize) -> Result<T> + Sync>(
        &mut self,
        range: std::ops::Range<usize>,
        function: F,
    ) -> Result<Vec<T>> {
        self.engine.map_range(range, function)
    }
    /// Each matrix's M×N sums for one shared input: one Metal command buffer
    /// (with the CPU kernel computing its calibrated share of the rows meanwhile,
    ///), or one CPU kernel call per matrix.
    pub(crate) fn weighted_sums(
        &mut self,
        matrices: &[&Matrix],
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<Vec<i64>>> {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &mut self.metal {
            return metal.weighted_sums_split(&mut self.engine, matrices, activation, scales);
        }
        matrices
            .iter()
            .map(|matrix| self.engine.weighted_sums(matrix, activation, scales))
            .collect()
    }
}
