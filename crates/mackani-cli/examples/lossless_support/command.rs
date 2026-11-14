use super::{check, configure, load_prompts, profile_configs, profile_samplers};
use mackani::{
    Error, Result, engine::Engine, real_engine::RealEngineConfig, request::RequestDefaults,
};
use std::path::PathBuf;

pub fn run(
    args: impl IntoIterator<Item = String>,
    mut load: impl FnMut(RealEngineConfig) -> Result<Box<dyn Engine>>,
) -> Result<bool> {
    let mut args = args.into_iter();
    let (mut model, mut prompts, mut report) = (None, None, None);
    let mut max_tokens = 256;
    let mut context = RealEngineConfig::default().context;
    let mut backend = None;
    let mut threads = None;
    let mut profile = "quick".to_string();
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "lossless --model-dir MODEL_FOLDER [--profile quick|full] [--prompts FILE] [--max-tokens N] [--ctx N] [--backend cpu|metal] [--threads N] [--report FILE]\nProfiles use all five draft configurations and four samplers; quick is the default.",
            );
            return Ok(true);
        }
        if ![
            "--model-dir",
            "--prompts",
            "--max-tokens",
            "--ctx",
            "--report",
            "--profile",
            "--backend",
            "--threads",
        ]
        .contains(&arg.as_str())
        {
            return Err(Error::request(format!("unknown option {arg}")));
        }
        let value = args
            .next()
            .ok_or_else(|| Error::request(format!("{arg} requires a value")))?;
        match arg.as_str() {
            "--model-dir" => model = Some(PathBuf::from(value)),
            "--prompts" => prompts = Some(PathBuf::from(value)),
            "--report" => report = Some(PathBuf::from(value)),
            "--backend" => backend = Some(value.parse().map_err(Error::engine)?),
            "--profile" => {
                profile_samplers(&value)?;
                profile = value;
            }
            "--max-tokens" | "--ctx" | "--threads" => {
                let n = value
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| Error::request(format!("{arg} must be positive")))?;
                if arg == "--ctx" {
                    context = n;
                } else if arg == "--threads" {
                    threads = Some(n);
                } else {
                    max_tokens = n;
                }
            }
            _ => unreachable!(),
        }
    }
    let names = profile_configs(&profile)?;
    let model_dir = model.ok_or_else(|| Error::request("--model-dir is required"))?;
    let prompts = load_prompts(prompts.as_deref())?;
    let base = RealEngineConfig {
        model_dir,
        context,
        backend,
        threads,
        ..RealEngineConfig::default()
    };
    let defaults = RequestDefaults {
        sampling: base.sampling_defaults()?,
        max_tokens,
        ..RequestDefaults::default()
    };
    let formatter = mackani_text::ChatFormatter::load(&base.model_dir)?;
    let result = check(
        &prompts,
        &names,
        &defaults,
        max_tokens,
        &profile,
        |name| load(configure(&base, name)?),
        &formatter,
    )?;
    let text = serde_json::to_string_pretty(&result).map_err(Error::engine)?;
    if let Some(path) = report {
        std::fs::write(&path, text)
            .map_err(|e| Error::engine(format!("{}: {e}", path.display())))?;
    } else {
        println!("{text}");
    }
    Ok(result["passed"] == true)
}
