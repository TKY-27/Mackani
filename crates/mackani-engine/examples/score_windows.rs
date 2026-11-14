use mackani_engine::{Backend, Engine, Error, ExecutionOptions, Kernel, KvMode, Model, Result};
use mackani_ref::quality::{TokenWindows, parse_kv_mode, score_with_kv};
use std::{path::PathBuf, time::Duration};

const USAGE: &str = "MODEL_FOLDER WINDOWS.json [--kv int8|int8-profile|kv5] [--backend cpu|metal] [--rows-per-pass N] [--windows A..B] [--json OUT] [--kernel portable|neon|dotprod] [--threads N] [--cpu-stages auto|reference] [--cpu-share auto|off|F]";

fn main() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let folder = arguments
        .next()
        .ok_or_else(|| Error::InvalidInput(format!("expected {USAGE}")))?;
    if folder == "--help" || folder == "-h" {
        println!("Usage: score_windows {USAGE}");
        println!("The default backend is Metal when available, otherwise CPU.");
        return Ok(());
    }
    let path = arguments
        .next()
        .ok_or_else(|| Error::InvalidInput("missing WINDOWS.json".into()))?;
    let windows = TokenWindows::load(path)?;
    let mut mode = KvMode::Int8;
    let mut execution = ExecutionOptions::default();
    let mut kernel = Kernel::fastest();
    let mut threads = Engine::default_threads();
    let mut rows = 16;
    let mut backend = None;
    let mut range = 0..windows.windows.len();
    let mut json = None;
    while let Some(option) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| Error::InvalidInput("option requires a value".into()))?;
        match option.to_str() {
            Some("--kernel" | "--threads" | "--cpu-stages" | "--cpu-share") => {
                let text = value
                    .to_str()
                    .ok_or_else(|| Error::InvalidInput("invalid execution option".into()))?;
                match option.to_str().unwrap() {
                    "--kernel" => kernel = text.parse()?,
                    "--threads" => {
                        threads =
                            text.parse().ok().filter(|&n| n > 0).ok_or_else(|| {
                                Error::InvalidInput("threads must be positive".into())
                            })?
                    }
                    "--cpu-stages" => execution.cpu_stages = text.parse()?,
                    "--cpu-share" => execution.metal.cpu_share = text.parse()?,
                    _ => unreachable!(),
                }
            }
            Some("--kv") => mode = parse_kv_mode(&value)?,
            Some("--backend") => {
                backend = Some(
                    value
                        .to_str()
                        .ok_or_else(|| Error::InvalidInput("invalid backend".into()))?
                        .parse()?,
                )
            }
            Some("--rows-per-pass") => {
                rows = value
                    .to_str()
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|&n| n > 0)
                    .ok_or_else(|| Error::InvalidInput("invalid rows per pass".into()))?
            }
            Some("--windows") => {
                let (first, last) =
                    value
                        .to_str()
                        .and_then(|s| s.split_once(".."))
                        .ok_or_else(|| {
                            Error::InvalidInput("expected half-open A..B window range".into())
                        })?;
                range = first
                    .parse()
                    .map_err(|_| Error::InvalidInput("invalid first window".into()))?
                    ..last
                        .parse()
                        .map_err(|_| Error::InvalidInput("invalid last window".into()))?;
            }
            Some("--json") => json = Some(PathBuf::from(value)),
            _ => return Err(Error::InvalidInput("unknown score option".into())),
        }
    }
    let backend = match backend {
        Some(backend) => backend,
        None => Backend::detected(),
    };
    let mut totals = [Duration::ZERO; 6];
    let mut passes = 0;
    let mut configuration = None;
    let result = score_with_kv(
        mode,
        |mode| {
            let engine = Engine::new(kernel, threads)?;
            let kernel = engine.kernel().name();
            let threads = engine.threads();
            let (weights, _) = mackani_ref::model::model_location(std::path::Path::new(&folder))?;
            let size_bytes = std::fs::metadata(&weights)?.len();
            // SAFETY: this read-only scoring command requires stable input weights until exit.
            let model = unsafe {
                Model::with_execution_options(
                    &folder,
                    engine,
                    ExecutionOptions {
                        backend,
                        ..execution
                    },
                    rows,
                    mode,
                )?
            };
            configuration = Some(serde_json::json!({
                "kv_mode": mode.name(),
                "backend": model.backend().name(),
                "device": model.device(),
                "kernel": kernel,
                "threads": threads,
                "rows_per_pass": rows,
                "window_range": {"start": range.start, "end": range.end},
                "model_file": {"name": weights.file_name().unwrap().to_string_lossy(), "size_bytes": size_bytes},
                "crate_version": env!("CARGO_PKG_VERSION"),
            }));
            eprintln!(
                "backend={} device={} rows_per_pass={rows} kv={mode:?}",
                model.backend().name(),
                model.device()
            );
            Ok(model)
        },
        &windows,
        range.clone(),
        rows,
        |model| {
            for timing in model.pass_timings() {
                totals[0] += timing.total;
                totals[1] += timing.kernel;
                totals[2] += timing.attention;
                totals[3] += timing.recurrence;
                totals[4] += timing.rest();
                totals[5] += timing.kv_encode;
                passes += 1;
                if timing.position < 200 || timing.position >= 1984 {
                    eprintln!(
                        "pass position={} rows={} total={:.6} kernel={:.6} attention={:.6} kv_encode={:.6} recurrence={:.6} rest={:.6}",
                        timing.position,
                        timing.rows,
                        timing.total.as_secs_f64(),
                        timing.kernel.as_secs_f64(),
                        timing.attention.as_secs_f64(),
                        timing.kv_encode.as_secs_f64(),
                        timing.recurrence.as_secs_f64(),
                        timing.rest().as_secs_f64()
                    );
                }
            }
        },
    )?;
    result.print();
    eprintln!(
        "profile passes={passes} total={:.6} kernel={:.6} attention={:.6} kv_encode={:.6} recurrence={:.6} rest={:.6} score_seconds={:.6}",
        totals[0].as_secs_f64(),
        totals[1].as_secs_f64(),
        totals[2].as_secs_f64(),
        totals[5].as_secs_f64(),
        totals[3].as_secs_f64(),
        totals[4].as_secs_f64(),
        result.elapsed_seconds
    );
    if let Some(path) = json {
        let mut output = configuration.expect("scoring loaded a model");
        output["score"] = serde_json::to_value(&result)?;
        std::fs::write(path, serde_json::to_vec_pretty(&output)?)?;
    }
    Ok(())
}
