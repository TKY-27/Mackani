//! Start-up checks and the effective run configuration on this machine.
//! Every check only reads facts, never changes a system setting, and fails
//! closed: a fact that cannot be read is an error, not a pass.
use std::fs;

mod run;

pub use run::RunChoice;

use mackani_engine::Machine;
use serde_json::Value;

use crate::{Error, Result, real_engine::RealEngineConfig};

/// The oldest supported macOS major version.
pub const MIN_MACOS: u32 = 26;

/// Apple silicon (macOS on aarch64) and macOS 26 or later; `macos` is
/// `kern.osproductversion` (`None` when it cannot be read).
pub fn check_platform(arch: &str, os: &str, macos: Option<&str>) -> Result<()> {
    if os != "macos" || arch != "aarch64" {
        return Err(Error::engine(format!(
            "mackani requires a Mac with Apple silicon (macOS on aarch64); this program runs on {os} {arch}"
        )));
    }
    let Some(version) = macos else {
        return Err(Error::engine(format!(
            "mackani requires macOS {MIN_MACOS} or later and cannot read this Mac's version (sysctl kern.osproductversion)"
        )));
    };
    // Every component must be a decimal number ("26.invalid" and "26..1" are not).
    let components: Option<Vec<u32>> = version
        .split('.')
        .map(|part| {
            part.bytes()
                .all(|b| b.is_ascii_digit())
                .then(|| part.parse().ok())
                .flatten()
        })
        .collect();
    let major = components
        .and_then(|components| components.first().copied())
        .ok_or_else(|| {
            Error::engine(format!(
                "mackani requires macOS {MIN_MACOS} or later and cannot read the version {version:?}"
            ))
        })?;
    // macOS 26 can report 16.x to programs built with an older SDK (as macOS 11
    // reported 10.16); no macOS 16 exists, so 16 is read as 26.
    let major = if major == 16 { 26 } else { major };
    if major < MIN_MACOS {
        return Err(Error::engine(format!(
            "mackani requires macOS {MIN_MACOS} or later; this Mac runs macOS {version}"
        )));
    }
    Ok(())
}

/// The checks for this process: its architecture and the running macOS.
pub fn check_this_platform() -> Result<()> {
    check_platform(
        std::env::consts::ARCH,
        std::env::consts::OS,
        Machine::detect().macos_version.as_deref(),
    )
}

/// Every file the configuration reads, before anything is opened or mapped;
/// one error lists all missing ones and which option needs each.
pub fn check_folder(config: &RealEngineConfig) -> Result<()> {
    let folder = &config.model_dir;
    let manifest_path = folder.join("yasakani.json");
    let manifest: Value = match fs::read(&manifest_path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|e| Error::engine(format!("{}: {e}", manifest_path.display())))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
        Err(e) => return Err(Error::engine(format!("{}: {e}", manifest_path.display()))),
    };
    // The same defaults as mackani_ref's manifest resolution; it validates the names.
    let file = |key: &str, default: &'static str| {
        manifest["files"][key]
            .as_str()
            .unwrap_or(default)
            .to_owned()
    };
    let mut needed = vec![
        ("yasakani.json".to_owned(), "the model manifest".to_owned()),
        (
            file("weights", "qwen38_27b_e3x.yskn"),
            "files.weights, the main model".to_owned(),
        ),
    ];
    if config.kv != mackani_engine::KvMode::Int8 {
        needed.push((
            file("kv_profile", "kvq_u5.bin"),
            format!(
                "files.kv_profile, for --kv {}; --kv int8 runs without it",
                config.kv_name()
            ),
        ));
    }
    if config.draft.neural() {
        let why = format!(
            "for --draft {}; --draft ngram or none runs without it",
            config.draft.name()
        );
        needed.push((
            file("draft", "bdraft2_k8.yskn"),
            format!("files.draft, the neural draft, {why}"),
        ));
        needed.push((
            file("draft_head", "bdraft2_k8.shead"),
            format!("files.draft_head, the draft head, {why}"),
        ));
    }
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        needed.push((name.to_owned(), "the tokenizer".to_owned()));
    }
    let missing: Vec<String> = needed
        .into_iter()
        .filter(|(name, _)| !folder.join(name).is_file())
        .map(|(name, why)| format!("{name} ({why})"))
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(Error::engine(format!(
            "model folder {} is incomplete: missing {}",
            folder.display(),
            missing.join(", ")
        )))
    }
}
