use std::sync::Once;

use mackani_engine::{Backend, CpuShare, CpuStages, Kernel, Machine, MetalOptions};
use serde_json::{Value, json};

use crate::{Error, Result, real_engine::RealEngineConfig};

/// The machine the Metal CPU share table was measured on.
const CALIBRATED_MACHINE: &str = "Apple M5 with 4+6 CPU cores and 10 GPU cores";

/// The effective switches of one run (`mackani info` shows them).
#[derive(Debug, Clone)]
pub struct RunChoice {
    pub backend: Backend,
    /// The Metal device name, or the CPU kernel and worker count.
    pub device: String,
    /// Why the default selection fell back to the CPU backend.
    pub fallback: Option<String>,
    pub kernel: Kernel,
    pub threads: usize,
    pub cpu_stages: CpuStages,
    pub metal: MetalOptions,
    /// Whether `CpuShare::Auto` runs the calibrated table on this machine.
    pub share_table: bool,
    pub max_buffer_length: Option<usize>,
    pub recommended_working_set: Option<u64>,
    pub machine: Machine,
    /// Where the adaptive depth's cost table comes from.
    pub draft_cost_table: String,
}

/// What a working Metal engine reports, or why there is none.
struct Probe {
    name: String,
    share_table: bool,
    max_buffer_length: usize,
    working_set: u64,
}

fn probe_metal(threads: usize) -> std::result::Result<Probe, String> {
    #[cfg(target_os = "macos")]
    {
        match mackani_engine::MetalEngine::new() {
            Ok(Some(metal)) => Ok(Probe {
                name: metal.device_name(),
                share_table: metal.share_table_applies(threads),
                max_buffer_length: metal.max_buffer_length(),
                working_set: metal.recommended_working_set(),
            }),
            Ok(None) => Err("no Metal device".into()),
            Err(error) => Err(error.to_string()),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = threads;
        Err("Metal requires macOS".into())
    }
}

/// The longest reason the fallback notice quotes, in characters.
const NOTICE_REASON_CHARS: usize = 300;

/// The one stderr line of a default selection without Metal; the cost is the
/// CPU kernel's relative to Metal, measured on the M5. A multi-line
/// reason (a shader compiler log) is joined with spaces and shortened.
fn fallback_notice(why: &str) -> String {
    let joined = why.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut why: String = joined.chars().take(NOTICE_REASON_CHARS).collect();
    if why.len() < joined.len() {
        why.push_str(" ...");
    }
    format!(
        "mackani: Metal is unavailable ({why}); using the CPU backend, whose compressed projections take about 6 times as long (646 instead of 111 ms per pass at one row on an Apple M5). --backend metal makes this an error."
    )
}

impl RunChoice {
    /// Explicit choices, else the detected ones. Without `--backend`, Metal is
    /// used when its device, shader and required features work, else the CPU
    /// backend with one stderr line saying why; an explicit Metal is an error.
    pub fn select(config: &RealEngineConfig) -> Result<Self> {
        Self::select_with(config, probe_metal)
    }

    fn select_with(
        config: &RealEngineConfig,
        probe: impl FnOnce(usize) -> std::result::Result<Probe, String>,
    ) -> Result<Self> {
        let kernel = config.kernel.unwrap_or_else(Kernel::fastest);
        if !kernel.available() {
            return Err(Error::engine(format!(
                "--kernel {} is not available on this CPU (available: {})",
                kernel.name(),
                Kernel::available_variants()
                    .iter()
                    .map(|k| k.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let threads = config
            .threads
            .unwrap_or_else(mackani_engine::Engine::default_threads);
        let cpu_device = format!("{} x{threads}", kernel.name());
        let machine = Machine::detect();
        let chip = machine.chip.as_deref().unwrap_or("unknown chip");
        let draft_cost_table = if matches!(
            config.depth_policy(),
            mackani_ref::speculation::DepthPolicy::Fixed
        ) {
            "unused (no adaptive neural draft)".to_owned()
        } else if chip == "Apple M5" {
            "measured on an Apple M5".to_owned()
        } else {
            format!(
                "measured on an Apple M5; on this {chip} it can only change speed, never output"
            )
        };
        let mut choice = Self {
            backend: Backend::Cpu,
            device: cpu_device,
            fallback: None,
            kernel,
            threads,
            cpu_stages: config.cpu_stages,
            metal: config.metal,
            share_table: false,
            max_buffer_length: None,
            recommended_working_set: None,
            machine,
            draft_cost_table,
        };
        if config.backend == Some(Backend::Cpu) {
            return Ok(choice);
        }
        match probe(threads) {
            Ok(probe) => {
                choice.backend = Backend::Metal;
                choice.device = probe.name;
                choice.share_table = probe.share_table;
                choice.max_buffer_length = Some(probe.max_buffer_length);
                choice.recommended_working_set = Some(probe.working_set);
            }
            Err(why) if config.backend == Some(Backend::Metal) => {
                return Err(Error::engine(format!(
                    "--backend metal was requested, but Metal is unavailable: {why}"
                )));
            }
            Err(why) => {
                static NOTICE: Once = Once::new();
                NOTICE.call_once(|| eprintln!("{}", fallback_notice(&why)));
                choice.fallback = Some(why);
            }
        }
        Ok(choice)
    }

    /// Replaces the requested choices with what a loaded model reports it
    /// uses, so `info` shows the settings actually applied.
    pub(crate) fn read_back(&mut self, model: &mackani_engine::Model) {
        self.backend = model.backend();
        self.device = model.device();
        self.kernel = model.kernel();
        self.threads = model.threads();
        // `Auto` without NEON runs the reference stages too, and says so.
        let auto_without_neon = self.cpu_stages == CpuStages::Auto && !CpuStages::Auto.fast();
        self.cpu_stages = if model.cpu_stages() || auto_without_neon {
            CpuStages::Auto
        } else {
            CpuStages::Reference
        };
        if let Some(metal) = model.metal_options() {
            self.metal = metal;
        }
    }

    fn cpu_stages_text(&self) -> String {
        match (self.cpu_stages, self.cpu_stages.fast()) {
            (CpuStages::Reference, _) => "reference".into(),
            (CpuStages::Auto, true) => "auto (NEON fast paths)".into(),
            (CpuStages::Auto, false) => "auto (no NEON detected: reference stages)".into(),
        }
    }

    fn cpu_share_text(&self) -> String {
        match self.metal.cpu_share {
            CpuShare::Off => "off".into(),
            CpuShare::Fixed(share) => format!("fixed {share}"),
            CpuShare::Auto if self.share_table => {
                format!("auto: calibrated table ({CALIBRATED_MACHINE}, 10 CPU workers)")
            }
            CpuShare::Auto => {
                let levels: Vec<_> = self
                    .machine
                    .core_levels
                    .iter()
                    .map(usize::to_string)
                    .collect();
                let cpu = if levels.is_empty() {
                    "unknown".into()
                } else {
                    levels.join("+")
                };
                let gpu = self
                    .machine
                    .gpu_cores
                    .map_or("unknown".into(), |n| n.to_string());
                format!(
                    "auto: off, the table was measured on an {CALIBRATED_MACHINE}, run with 10 CPU workers, and this is {} ({cpu} CPU cores, {gpu} GPU cores) with {}; measure with bench_metal --shares and pass --cpu-share",
                    self.device, self.threads
                )
            }
        }
    }

    /// Fields of `mackani info`.
    pub fn to_json(&self) -> Value {
        let metal = self.backend == Backend::Metal;
        let unused = || json!("unused (CPU backend)");
        json!({
            "kernel": self.kernel.name(),
            "cpu_stages": self.cpu_stages_text(),
            "metal_mac": if metal { json!("float") } else { unused() },
            "cpu_share": if metal { json!(self.cpu_share_text()) } else { unused() },
            "backend_fallback": self.fallback,
            "draft_cost_table": self.draft_cost_table,
            "machine": {
                "chip": self.machine.chip,
                "macos": self.machine.macos_version,
                "architecture": std::env::consts::ARCH,
                "cores_per_performance_level": self.machine.core_levels,
                "gpu_cores": self.machine.gpu_cores,
                "page_size": self.machine.page_size,
                "metal_max_buffer_length": self.max_buffer_length,
                "metal_recommended_working_set": self.recommended_working_set,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shader that does not compile or a missing feature (MetalEngine::new
    /// fails) or no device: the default selection runs on the CPU and says why;
    /// an explicit Metal backend is an error; a working probe selects Metal.
    #[test]
    fn default_backend_falls_back_to_the_cpu_and_explicit_metal_fails() {
        let failing = |_| Err("Metal shader compilation: expected failure".to_owned());
        let config = RealEngineConfig::default();
        let choice = RunChoice::select_with(&config, failing).unwrap();
        assert_eq!(choice.backend, Backend::Cpu);
        assert_eq!(
            choice.device,
            format!("{} x{}", choice.kernel.name(), choice.threads)
        );
        assert_eq!(
            choice.fallback.as_deref(),
            Some("Metal shader compilation: expected failure")
        );
        let notice = fallback_notice("no Metal device");
        assert!(
            notice.starts_with(
                "mackani: Metal is unavailable (no Metal device); using the CPU backend"
            )
        );
        assert!(notice.contains("6 times as long") && !notice.contains('\n'));
        let metal = RealEngineConfig {
            backend: Some(Backend::Metal),
            ..RealEngineConfig::default()
        };
        let error = RunChoice::select_with(&metal, failing)
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "--backend metal was requested, but Metal is unavailable: Metal shader compilation: expected failure"
        );
        let working = |threads| {
            Ok(Probe {
                name: "Apple M9".into(),
                share_table: threads == 10,
                max_buffer_length: 1 << 34,
                working_set: 1 << 33,
            })
        };
        let choice = RunChoice::select_with(&config, working).unwrap();
        assert_eq!(
            (choice.backend, choice.device.as_str()),
            (Backend::Metal, "Apple M9")
        );
        assert!(choice.fallback.is_none());
        let cpu = RealEngineConfig {
            backend: Some(Backend::Cpu),
            ..RealEngineConfig::default()
        };
        let choice = RunChoice::select_with(&cpu, |_| panic!("CPU must not probe Metal")).unwrap();
        assert_eq!(choice.backend, Backend::Cpu);
        assert!(choice.fallback.is_none());
    }

    /// A shader compiler log with line breaks stays one line, and a long one
    /// is cut at `NOTICE_REASON_CHARS` characters.
    #[test]
    fn fallback_notice_is_one_bounded_line() {
        let notice = fallback_notice(
            "Metal shader compilation: program_source:12:3: error: x\r\n  int y;\n\n  ^\nprogram_source:14:1: error: z",
        );
        assert_eq!(
            notice,
            "mackani: Metal is unavailable (Metal shader compilation: program_source:12:3: error: x int y; ^ program_source:14:1: error: z); using the CPU backend, whose compressed projections take about 6 times as long (646 instead of 111 ms per pass at one row on an Apple M5). --backend metal makes this an error."
        );
        let long = "é\n".repeat(1000);
        let notice = fallback_notice(&long);
        assert!(!notice.contains('\n') && !notice.contains('\r'), "{notice}");
        let quoted = notice
            .strip_prefix("mackani: Metal is unavailable (")
            .and_then(|rest| rest.split_once(" ...);"))
            .unwrap()
            .0;
        assert_eq!(quoted.chars().count(), NOTICE_REASON_CHARS);
    }

    /// Off-machine CPU share: `auto` says it is off and how to calibrate.
    #[test]
    fn cpu_share_text_names_the_calibrated_machine_and_how_to_calibrate() {
        let config = RealEngineConfig {
            threads: Some(8),
            ..RealEngineConfig::default()
        };
        let probe = |threads| {
            Ok(Probe {
                name: "Apple M4 Pro".into(),
                share_table: threads == 10,
                max_buffer_length: 1 << 34,
                working_set: 1 << 33,
            })
        };
        let mut choice = RunChoice::select_with(&config, probe).unwrap();
        choice.machine.core_levels = vec![10, 4];
        choice.machine.gpu_cores = Some(20);
        assert_eq!(
            choice.to_json()["cpu_share"],
            "auto: off, the table was measured on an Apple M5 with 4+6 CPU cores and 10 GPU cores, run with 10 CPU workers, and this is Apple M4 Pro (10+4 CPU cores, 20 GPU cores) with 8; measure with bench_metal --shares and pass --cpu-share"
        );
        choice.machine.core_levels.clear();
        choice.machine.gpu_cores = None;
        assert!(
            choice.to_json()["cpu_share"]
                .as_str()
                .unwrap()
                .contains("this is Apple M4 Pro (unknown CPU cores, unknown GPU cores) with 8"),
        );
        choice.share_table = true;
        assert_eq!(
            choice.to_json()["cpu_share"],
            "auto: calibrated table (Apple M5 with 4+6 CPU cores and 10 GPU cores, 10 CPU workers)"
        );
    }
}
