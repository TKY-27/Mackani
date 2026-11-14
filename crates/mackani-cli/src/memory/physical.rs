use std::{fs, process::Command};

use super::mul;
use crate::{Error, Result};

/// Physical memory only; no settings are changed. Failure is explicit, never a
/// silent bypass of the start-up fit check.
pub fn physical_memory() -> Result<u64> {
    if cfg!(target_os = "macos") {
        let primary = Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output();
        if let Ok(result) = &primary {
            if result.status.success() {
                if let Ok(bytes) = String::from_utf8_lossy(&result.stdout)
                    .trim()
                    .parse::<u64>()
                {
                    if bytes > 0 {
                        return Ok(bytes);
                    }
                }
            }
        }
        // Some macOS sandboxes deny sysctl's format lookup. Foundation exposes
        // the same physical-memory fact without changing any system setting.
        // This script is fixed: no model path or request input is interpolated.
        let fallback = Command::new("/usr/bin/osascript")
            .args([
                "-l",
                "JavaScript",
                "-e",
                "ObjC.import('Foundation'); $.NSProcessInfo.processInfo.physicalMemory.toString()",
            ])
            .output();
        if let Ok(result) = &fallback {
            if result.status.success() {
                if let Ok(bytes) = String::from_utf8_lossy(&result.stdout)
                    .trim()
                    .parse::<u64>()
                {
                    if bytes > 0 {
                        return Ok(bytes);
                    }
                }
            }
        }
        let explain = |result: std::io::Result<std::process::Output>| match result {
            Ok(output) => String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            Err(error) => error.to_string(),
        };
        return Err(Error::engine(format!(
            "cannot read physical memory: sysctl hw.memsize: {}; Foundation NSProcessInfo.physicalMemory: {}",
            explain(primary),
            explain(fallback)
        )));
    }

    let meminfo = fs::read_to_string("/proc/meminfo").map_err(Error::engine)?;
    let kb = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|n| n.parse::<u64>().ok())
        .ok_or_else(|| Error::engine("cannot read physical memory"))?;
    mul(kb, 1024)
}
