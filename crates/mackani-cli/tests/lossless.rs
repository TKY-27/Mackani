//! Socket-free proof of the development harness against the actual product engine.
#[allow(dead_code)]
#[path = "../examples/lossless_support/mod.rs"]
mod lossless_support;
#[allow(dead_code)]
mod product_support;

use lossless_support::{CONFIG_NAMES, check, configure, prompts};
use product_support::Folder;
use serde_json::{Value, json};
use std::{
    cell::Cell,
    collections::HashMap,
    process::{Command, Output},
    rc::Rc,
    sync::OnceLock,
};

use mackani::{
    Result,
    engine::{Engine, Generation, Sampling},
    real_engine::RealEngine,
    request::RequestDefaults,
};

#[derive(Clone, Copy, PartialEq)]
enum Fault {
    None,
    Configuration,
    Repeat,
    NoAcceptance,
    LaterError,
    Empty,
    ConstantSeed,
    Reuse,
}
struct TestEngine {
    inner: RealEngine,
    fault: Fault,
    counts: HashMap<(Vec<u32>, i64), usize>,
    live: Rc<Cell<usize>>,
    reset_requested: bool,
}
impl Drop for TestEngine {
    fn drop(&mut self) {
        self.live.set(self.live.get() - 1);
    }
}
impl Engine for TestEngine {
    fn context_capacity(&self) -> usize {
        self.inner.context_capacity()
    }
    fn reset(&mut self) -> Result<()> {
        self.reset_requested = true;
        self.inner.reset()
    }
    fn generate(
        &mut self,
        prompt: &[u32],
        n: usize,
        sampling: &Sampling,
        stops: Option<&[u32]>,
        emit: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Generation> {
        if self.fault == Fault::LaterError && prompt.first() == Some(&13) && n > 1 {
            return Err(mackani::Error::engine("injected later generation error"));
        }
        let mut requested = sampling.clone();
        if self.fault == Fault::ConstantSeed {
            requested.seed = 42;
        }
        let sampling = &requested;
        let mut result = if self.fault == Fault::Empty {
            self.inner
                .generate(prompt, n, sampling, stops, &mut |_| Ok(true))?
        } else {
            self.inner.generate(prompt, n, sampling, stops, emit)?
        };
        let original = prompt.first() == Some(&1) && n > 1;
        let count = self
            .counts
            .entry((prompt.to_vec(), sampling.seed))
            .or_default();
        if original {
            *count += 1;
        }
        if original
            && (matches!(self.fault, Fault::Configuration | Fault::LaterError)
                || (self.fault == Fault::Repeat && *count == 2))
        {
            result.ids[2] = (result.ids[2] + 1) % 32;
        }
        if n == 8
            && self.fault == Fault::Reuse
            && (result.snap_mode == 1 || (prompt.contains(&4) && !self.reset_requested))
        {
            result.ids[0] ^= 1;
        }
        if self.fault == Fault::Empty {
            result.ids.clear();
        }
        if self.fault == Fault::NoAcceptance {
            result.guesses_accepted = 0;
        }
        self.reset_requested = false;
        Ok(result)
    }
}

fn exercise(fault: Fault, all: bool) -> Value {
    exercise_template(
        fault,
        all,
        "{% if enable_thinking %}t13 {% endif %}{% if tools %}t14 {% endif %}{% for m in messages %}{% if m.role == 'assistant' %}t10 {% endif %}{{ m.content }} {% endfor %}t10",
    )
}

fn exercise_template(fault: Fault, all: bool, template: &str) -> Value {
    let started = std::time::Instant::now();
    let folder = Folder::new();
    // Generation headers are present on historical assistant turns too. The
    // system setting changes the start, as it does in the real chat template.
    folder.write("chat_template.jinja", template);
    let cases = prompts(
        "test prompts.json",
        include_str!("lossless_support/prompts.json"),
    )
    .unwrap();
    let selected: &[&str] = if fault == Fault::ConstantSeed {
        &["none", "both-k7-adaptive"]
    } else if all {
        &CONFIG_NAMES
    } else {
        &["none", "ngram-k3-fixed"]
    };
    let names = selected
        .iter()
        .map(|name| (*name).into())
        .collect::<Vec<_>>();
    let live = Rc::new(Cell::new(0));
    let defaults = RequestDefaults {
        max_tokens: 12,
        sampling: folder
            .config(mackani::real_engine::DraftMode::None)
            .sampling_defaults()
            .unwrap(),
        ..RequestDefaults::default()
    };
    let result = check(
        &cases,
        &names,
        &defaults,
        12,
        "full",
        |name| {
            assert_eq!(
                live.get(),
                0,
                "previous weights must drop before the next load"
            );
            let config = configure(&folder.config(mackani::real_engine::DraftMode::None), name)?;
            let mut options = product_support::model::draft::options();
            options.capture_layers = vec![0];
            let inner = RealEngine::load_with_draft_options(config, options)?;
            live.set(live.get() + 1);
            Ok(Box::new(TestEngine {
                inner,
                fault: if name == "none" && fault != Fault::ConstantSeed {
                    Fault::None
                } else {
                    fault
                },
                counts: HashMap::new(),
                live: live.clone(),
                reset_requested: false,
            }))
        },
        &mackani_text::ChatFormatter::load(&folder.0).unwrap(),
    )
    .unwrap();
    assert_eq!(live.get(), 0);
    eprintln!(
        "{}: {} engine loads, {:.3}s",
        std::thread::current().name().unwrap_or("fault child"),
        names.len(),
        started.elapsed().as_secs_f64()
    );
    result
}

#[test]
fn actual_synthetic_engine_matches_ids_but_rejects_unexercised_draft_cells() {
    let report = exercise(Fault::None, true);
    assert_eq!(report["passed"], false);
    let failures = report["failures"].as_array().unwrap();
    assert_eq!(failures.len(), 9, "{failures:?}");
    assert!(
        failures
            .iter()
            .all(|f| f["kind"] == "speculation not exercised"
                || f["kind"] == "draft source not exercised"),
        "{failures:?}"
    );
    let runs = report["runs"].as_array().unwrap();
    assert_eq!(
        runs.iter().filter(|r| r["probe"] != true).count(),
        5 * 4 * 3 * 2
    );
    for run in runs
        .iter()
        .filter(|r| r["configuration"] == "none" && r["probe"] == true)
    {
        eprintln!(
            "reuse {}: modes={} cached={}",
            run["run"], run["snapshot_modes"], run["cached_tokens"]
        );
    }
    assert!(
        runs.iter()
            .any(|r| r["prompt"] == "tool" && r["prompt_tokens"] == 6)
    );
    for name in CONFIG_NAMES {
        let own: Vec<_> = runs.iter().filter(|r| r["configuration"] == name).collect();
        assert!(
            own.iter()
                .filter(|r| r["expected_error"].is_null())
                .all(|r| r["passes"].as_u64().unwrap() > 0)
        );
        assert!(own.iter().any(|r| r["run"] == "after_failed"
            && r["snapshot_modes"] == json!([2])
            && r["cached_tokens"].as_u64().unwrap() > 0));
        assert!(
            own.iter()
                .any(|r| r["reset_before"] == true && r["snapshot_modes"] == json!([0]))
        );
        assert!(
            own.iter()
                .any(|r| r["run"] == "conversation_extension" && r["snapshot_modes"] == json!([1]))
        );
        for run in [
            "after_system_sibling",
            "after_unrelated",
            "after_reset",
            "after_cancelled",
            "after_failed",
        ] {
            assert!(
                own.iter().any(|r| r["run"] == run && r["token_count"] == 1),
                "{report}"
            );
        }
        for row in own.iter().filter(|r| r["expected_error"].is_null()) {
            assert_eq!(
                row["guesses_offered"].as_u64().unwrap(),
                row["neural_guesses_offered"].as_u64().unwrap()
                    + row["ngram_guesses_offered"].as_u64().unwrap()
            );
        }
        if name != "none" {
            assert!(
                own.iter()
                    .filter(|r| r["probe"] != true)
                    .map(|r| r["guesses_accepted"].as_u64().unwrap())
                    .sum::<u64>()
                    > 0
            );
        }
    }
}

#[test]
fn rewritten_continuation_is_compared_and_fixed_pair_still_exercises_live_reuse() {
    let report = exercise_template(
        Fault::None,
        false,
        "{% if enable_thinking %}t13 {% endif %}{% if tools %}t14 {% endif %}{% for m in messages %}{% if m.role == 'assistant' %}t10 t11 {% endif %}{{ m.content }} {% endfor %}t10",
    );
    let runs = report["runs"].as_array().unwrap();
    for name in ["none", "ngram-k3-fixed"] {
        let real = runs
            .iter()
            .find(|r| r["configuration"] == name && r["run"] == "conversation_extension")
            .unwrap();
        assert_eq!(real["continuation_strict_prefix"], false);
        assert_ne!(real["assistant_answer"], "Cedar.");
        assert_eq!(real["token_count"], 8);
        let fixed = runs
            .iter()
            .find(|r| r["configuration"] == name && r["run"] == "fixed_extension")
            .unwrap();
        assert_eq!(fixed["snapshot_modes"], json!([1]));
        assert!(fixed["cached_tokens"].as_u64().unwrap() > 0);
        let reset = runs
            .iter()
            .find(|r| r["configuration"] == name && r["run"] == "fixed_extension_reset")
            .unwrap();
        assert_eq!(fixed["sha256_ids_le_u32"], reset["sha256_ids_le_u32"]);
    }
    assert!(
        report["failures"]
            .as_array()
            .unwrap()
            .iter()
            .all(|f| f["kind"] == "speculation not exercised"
                || f["kind"] == "draft source not exercised"),
        "{}",
        report["failures"]
    );
}

#[test]
fn wrong_ids_only_during_live_or_sibling_reuse_fail_comparison() {
    let report = exercise(Fault::Reuse, false);
    for run in ["conversation_extension", "system_sibling"] {
        assert_eq!(report["passed"], false);
        assert!(
            report["failures"]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| f["kind"] == "reuse_mismatch" && f["run"] == run && f["position"] == 0),
            "{report}"
        );
    }
}

#[test]
fn a_constant_seed_fails_the_same_settings_probe() {
    let report = exercise(Fault::ConstantSeed, true);
    let failures = report["failures"].as_array().unwrap();
    for name in ["none", "both-k7-adaptive"] {
        assert!(
            failures
                .iter()
                .any(|f| f["kind"] == "seed not exercised" && f["configuration"] == name),
            "{report}"
        );
    }
    let runs = report["runs"].as_array().unwrap();
    let own: Vec<_> = runs
        .iter()
        .filter(|r| r["configuration"] == "none" && r["sampler"] == "seed-only")
        .collect();
    assert_eq!(own.len(), 8);
    assert_eq!(
        own.iter()
            .filter(|r| r["requested_sampling_by_stage"][0]["seed"] == 42)
            .count(),
        4
    );
    assert_eq!(
        own.iter()
            .filter(|r| r["requested_sampling_by_stage"][0]["seed"] == 2027)
            .count(),
        4
    );
    let mut settings: Vec<_> = own
        .iter()
        .map(|r| r["requested_sampling_by_stage"][0].clone())
        .collect();
    for setting in &mut settings {
        setting.as_object_mut().unwrap().remove("seed");
    }
    assert!(settings.iter().all(|s| s == &settings[0]));
}

#[test]
fn an_empty_actual_engine_generation_fails_instead_of_comparing_equal() {
    let report = exercise(Fault::Empty, false);
    assert_eq!(report["passed"], false);
    assert!(report["failures"].as_array().unwrap().iter().any(|f| {
        f["kind"] == "error"
            && f["message"]
                .as_str()
                .unwrap()
                .contains("generation not exercised")
    }));
}

#[test]
fn later_error_preserves_the_mismatch_and_completed_rows() {
    let report = exercise(Fault::LaterError, false);
    assert_eq!(report["passed"], false);
    let failures = report["failures"].as_array().unwrap();
    assert!(
        failures
            .iter()
            .any(|f| f["kind"] == "configuration_mismatch")
    );
    assert!(failures.iter().any(|f| {
        f["kind"] == "error"
            && f["message"]
                .as_str()
                .unwrap()
                .contains("injected later generation error")
    }));
    assert!(report["runs"].as_array().unwrap().len() > 1);
}

#[test]
fn token_and_repeat_faults_and_inactive_drafting_report_diagnostics() {
    for (fault, kind) in [
        (Fault::Configuration, "configuration_mismatch"),
        (Fault::Repeat, "repeat_mismatch"),
        (Fault::NoAcceptance, "speculation not exercised"),
    ] {
        let report = exercise(fault, false);
        assert_eq!(report["passed"], false);
        let failures = report["failures"].as_array().unwrap();
        assert!(failures.iter().any(|f| f["kind"] == kind), "{report}");
        if fault != Fault::NoAcceptance {
            assert!(failures.iter().any(|f| f["prompt"] == "refrain"
                && f["position"] == 2
                && f["expected_id"] != f["actual_id"]));
        } else {
            assert!(failures.iter().all(|f| f["kind"] == kind), "{report}");
        }
    }
}

#[test]
fn malformed_prompt_files_name_the_file_and_entry_instead_of_panicking() {
    assert!(lossless_support::profile_samplers("invalid").is_err());
    for text in [
        "{",
        "null",
        "[{},{}]",
        r#"[{"id":"x","request":{"messages":[{"role":"user","content":"a"}]}},{"id":"x","request":{"messages":[{"role":"user","content":"b"}]}}]"#,
        "[{\"id\":\"x\",\"request\":{}},{}]",
        "[{\"id\":\"x\",\"request\":{\"messages\":[{\"role\":\"user\",\"content\":\"\"}],\"seed\":1}},{}]",
    ] {
        let error = prompts("broken.json", text).err().unwrap().to_string();
        assert!(
            error.contains("broken.json") && error.contains("entry"),
            "{error}"
        );
    }
}

#[test]
fn bundled_prompt_file_has_unique_ids_documented_budgets_and_profile_arithmetic() {
    let defaults = prompts("defaults.json", lossless_support::DEFAULT_PROMPTS).unwrap();
    assert_eq!(defaults.len(), 24);
    assert_eq!(
        defaults
            .iter()
            .map(|p| &p.id)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        24
    );
    assert_eq!(
        defaults
            .iter()
            .map(|p| p.request["max_tokens"].as_u64().unwrap())
            .sum::<u64>(),
        1792
    );
    let mut budgets: Vec<_> = defaults
        .iter()
        .map(|p| p.request["max_tokens"].as_u64().unwrap())
        .collect();
    budgets.sort();
    assert_eq!(budgets, [vec![32; 16], vec![64; 4], vec![256; 4]].concat());
    let entries: Value = serde_json::from_str(lossless_support::DEFAULT_PROMPTS).unwrap();
    assert_eq!(
        entries
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["quick_max_tokens"].as_u64().unwrap())
            .sum::<u64>(),
        200
    );
    for (profile, configs, samplers, main, unrelated, reuse_min, reuse_max, total_min, total_max) in [
        ("full", 5, 4, 71_680, 460, 205, 290, 72_473, 72_558),
        ("quick", 5, 4, 8_000, 460, 205, 290, 8_793, 8_878),
    ] {
        assert_eq!(
            lossless_support::profile_configs(profile).unwrap().len(),
            configs
        );
        assert_eq!(
            lossless_support::profile_samplers(profile).unwrap().len(),
            samplers
        );
        let sweep: usize = if profile == "quick" {
            entries
                .as_array()
                .unwrap()
                .iter()
                .map(|e| e["quick_max_tokens"].as_u64().unwrap() as usize)
                .sum()
        } else {
            defaults
                .iter()
                .map(|p| p.request["max_tokens"].as_u64().unwrap() as usize)
                .sum()
        };
        assert_eq!(configs * samplers * 2 * sweep, main);
        assert_eq!(configs * samplers * (defaults.len() - 1), unrelated);
        assert_eq!(configs * 41, reuse_min);
        assert_eq!(configs * (41 + 17), reuse_max);
        assert_eq!(main + unrelated + reuse_min + 128, total_min);
        assert_eq!(main + unrelated + reuse_max + 128, total_max);
    }
}

/// Constant-zero main logits and draft scores form a uniform teacher. The
/// default-option proof and the small all-configuration proof share this oracle.
fn uniform_teacher_fixture(folder: &Folder) {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut main = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(folder.0.join("weights.yskn"))
        .unwrap();
    let mut prefix = [0; 32];
    main.read_exact(&mut prefix).unwrap();
    let header_len = u64::from_le_bytes(prefix[8..16].try_into().unwrap()) as usize;
    let mut bytes = vec![0; header_len];
    main.read_exact(&mut bytes).unwrap();
    let header: Value = serde_json::from_slice(&bytes).unwrap();
    let head = header["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "lm_head")
        .unwrap();
    let offset = head["scale_off"].as_u64().unwrap();
    let rows = head["physical_rows"].as_u64().unwrap() as usize;
    main.seek(SeekFrom::Start(offset)).unwrap();
    main.write_all(&vec![0; rows * 2]).unwrap();
    drop(main);
    let mut draft = std::fs::read(folder.0.join("draft.yskn")).unwrap();
    let header_len = u64::from_le_bytes(draft[8..16].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&draft[32..32 + header_len]).unwrap();
    for v in header["vectors"].as_array().unwrap() {
        if v["name"].as_str().unwrap().starts_with("bdraft2.sel.") {
            let off = v["off"].as_u64().unwrap() as usize;
            let len = v["n"].as_u64().unwrap() as usize * 2;
            draft[off..off + len].fill(0);
        }
    }
    folder.write("draft.yskn", draft);
    let head_path = folder.0.join("draft.shead");
    let len = std::fs::metadata(&head_path).unwrap().len();
    let mut prefix = [0; 24];
    std::fs::File::open(&head_path)
        .unwrap()
        .read_exact(&mut prefix)
        .unwrap();
    folder.write("draft.shead", prefix);
    std::fs::OpenOptions::new()
        .write(true)
        .open(head_path)
        .unwrap()
        .set_len(len)
        .unwrap();
}

fn child_output(command: &mut Command, expected_code: i32) -> Output {
    let child = command
        .output()
        .unwrap_or_else(|e| panic!("{command:?}: {e}"));
    assert_eq!(
        child.status.code(),
        Some(expected_code),
        "{command:?}: status={}\nstdout={}\nstderr={}",
        child.status,
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr),
    );
    child
}

fn example_binary() -> &'static std::path::Path {
    static BINARY: OnceLock<std::path::PathBuf> = OnceLock::new();
    BINARY.get_or_init(|| {
        let exe = std::env::current_exe().unwrap();
        let root = exe
            .ancestors()
            .find(|p| {
                matches!(
                    p.file_name().and_then(|n| n.to_str()),
                    Some("release" | "debug")
                )
            })
            .expect("Cargo profile directory");
        let mut command = Command::new(env!("CARGO"));
        command
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args([
                "build",
                "--offline",
                "-p",
                "mackani",
                "--example",
                "lossless",
                "--target-dir",
            ])
            .arg(root.parent().unwrap())
            .env("CARGO_NET_OFFLINE", "true");
        if root.file_name().unwrap() == "release" {
            command.arg("--release");
        }
        child_output(&mut command, 0);
        root.join("examples/lossless")
    })
}

#[test]
fn command_passes_every_configuration_and_saves_mismatch_reports_with_a_uniform_teacher() {
    use mackani_ref::{Sampler, SamplingParameters, SamplingVersion};
    // Default capture taps, mask token and shortlist need a full-size fixture
    // once; their compatibility does not depend on configuration/sampler loops.
    {
        let folder = Folder::new();
        folder.neural_bench_fixture();
        uniform_teacher_fixture(&folder);
        let config = folder.config(mackani::real_engine::DraftMode::Neural);
        let mut engine = RealEngine::load(config).unwrap();
        let sampling = Sampling {
            temperature: 0.0,
            ..RequestDefaults::default().sampling
        };
        let generated = engine
            .generate(&[1], 2, &sampling, None, &mut |_| Ok(true))
            .unwrap();
        assert_eq!(generated.ids, [0, 0]);
        assert!(generated.neural_guesses_offered > 0);
        assert!(generated.guesses_accepted > 0);
    }
    let folder = Folder::new();
    uniform_teacher_fixture(&folder);
    folder.write("chat_template.jinja", "{% for m in messages %}{% if m.role == 'assistant' %}t10 {% endif %}{{ m.content }} {% endfor %}t10");
    // At four output tokens, one prompt must offer a long n-gram immediately;
    // the other has a distinct suffix so both must use the neural proposer.
    // The prefix includes the generation header and the pending sampled token.
    const PROMPT_TOKENS: usize = 64;
    let mut teacher = Vec::new();
    for (seed, temperature, top_p, top_k, min_p) in [
        (0, 0.0, 1.0, 1, 0.0),
        (42, 0.7, 0.9, 20, 0.0),
        (2027, 0.75, 0.9, 11, 0.05),
        (91, 1.3, 1.0, 20, 0.0),
    ] {
        let sampler = if temperature == 0.0 {
            Sampler::Greedy
        } else {
            Sampler::Sampled(SamplingParameters {
                seed,
                temperature,
                top_p,
                top_k,
                min_p,
                version: SamplingVersion::Gumbel,
            })
        };
        teacher.extend([0, 0, 0, 0, 0, 10]);
        for position in PROMPT_TOKENS..PROMPT_TOKENS + 4 {
            teacher.push(sampler.sample(&[0.0; 32], position as u64).unwrap());
        }
    }
    let cases: Vec<_> = [1, 2].into_iter().map(|id| {
        let mut ids = vec![id];
        ids.extend_from_slice(&teacher);
        ids.resize(PROMPT_TOKENS - 1, 0);
        if id == 2 {
            ids[PROMPT_TOKENS - 6..].copy_from_slice(&[1, 2, 3, 4, 5]);
        }
        let words: Vec<_> = ids.iter().map(|id| if *id < 10 { id.to_string() } else { format!("t{id}") }).collect();
        json!({"id":format!("teacher-{id}"),"quick_max_tokens":4,"request":{"messages":[{"role":"user","content":words.join(" ")}],"enable_thinking":false,"max_tokens":4,"loop_guard":false}})
    }).collect();
    let formatter = mackani_text::ChatFormatter::load(&folder.0).unwrap();
    for case in &cases {
        let (rendered, _) = formatter
            .prompt(
                &case["request"]["messages"],
                None,
                &json!({"enable_thinking":false}),
            )
            .unwrap();
        assert_eq!(rendered.len(), PROMPT_TOKENS);
        assert_eq!(&rendered[1..1 + teacher.len()], &teacher);
    }
    folder.json("prompts.json", json!(cases));
    let path = folder.0.join("teacher-report.json");
    let args = vec![
        "--profile".into(),
        "full".into(),
        "--ctx".into(),
        "512".into(),
        "--backend".into(),
        "cpu".into(),
        "--threads".into(),
        "1".into(),
        "--max-tokens".into(),
        "4".into(),
        "--model-dir".into(),
        folder.0.to_str().unwrap().into(),
        "--prompts".into(),
        folder.0.join("prompts.json").to_str().unwrap().into(),
        "--report".into(),
        path.to_str().unwrap().into(),
    ];
    let load = |config| {
        let mut options = product_support::model::draft::options();
        options.capture_layers = vec![0];
        RealEngine::load_with_draft_options(config, options)
    };
    assert!(lossless_support::run(args.clone(), |config| Ok(Box::new(load(config)?))).unwrap());
    let report: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let live = Rc::new(Cell::new(0));
    assert!(
        !lossless_support::run(args, |config| {
            let inner = load(config)?;
            live.set(live.get() + 1);
            Ok(Box::new(TestEngine {
                inner,
                fault: Fault::Repeat,
                counts: HashMap::new(),
                live: live.clone(),
                reset_requested: false,
            }))
        })
        .unwrap()
    );
    assert_eq!(live.get(), 0);
    let mismatched: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(mismatched["passed"], false);
    assert!(
        mismatched["failures"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["kind"] == "repeat_mismatch"
                && f["position"] == 2
                && f["expected_id"] != f["actual_id"])
    );
    assert!(
        report["runs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["token_count"].as_u64().unwrap_or(0) <= 4)
    );
    assert_eq!(report["passed"], true);
    let runs = report["runs"].as_array().unwrap();
    for name in CONFIG_NAMES {
        for sampler in lossless_support::SAMPLERS {
            let own: Vec<_> = runs
                .iter()
                .filter(|r| {
                    r["configuration"] == name && r["sampler"] == sampler && r["probe"] != true
                })
                .collect();
            assert_eq!(own.len(), 4);
            let sum = |key: &str| own.iter().map(|r| r[key].as_u64().unwrap()).sum::<u64>();
            if name != "none" {
                assert!(sum("guesses_offered") > 0);
                if sampler == "greedy" || sampler == "seed-42" {
                    assert!(sum("guesses_accepted") > 0);
                }
                if name == "both-k7-adaptive" {
                    assert!(sum("neural_guesses_offered") > 0 && sum("ngram_guesses_offered") > 0);
                }
            }
            eprintln!(
                "teacher {name}/{sampler}: offered={} accepted={}",
                sum("guesses_offered"),
                sum("guesses_accepted")
            );
        }
    }
}

#[test]
fn real_example_binary_validates_options_and_saves_failed_report_before_exiting() {
    let binary = example_binary();
    let help = child_output(Command::new(binary).arg("--help"), 0);
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains("quick|full") && help.contains("quick is the default"));
    for (option, value, message) in [
        ("--backend", "invalid", "expected cpu or metal"),
        ("--threads", "0", "must be positive"),
        ("--profile", "release", "quick or full"),
        ("--draft-capture-layers", "0", "unknown option"),
        ("--draft-mask-token", "31", "unknown option"),
        ("--draft-shortlist", "32", "unknown option"),
        ("--configs", "none", "unknown option"),
    ] {
        let child = child_output(Command::new(binary).args([option, value]), 1);
        let stderr = String::from_utf8_lossy(&child.stderr);
        assert!(stderr.contains(message), "{stderr}");
    }
    let folder = Folder::new();
    folder.write(
        "prompts.json",
        include_str!("lossless_support/prompts.json"),
    );
    let report_path = folder.0.join("failed-report.json");
    child_output(
        Command::new(binary)
            .args([
                "--profile",
                "quick",
                "--ctx",
                "512",
                "--backend",
                "cpu",
                "--threads",
                "1",
                "--max-tokens",
                "8",
            ])
            .arg("--model-dir")
            .arg(&folder.0)
            .arg("--prompts")
            .arg(folder.0.join("prompts.json"))
            .arg("--report")
            .arg(&report_path),
        1,
    );
    let report: Value = serde_json::from_slice(&std::fs::read(report_path).unwrap()).unwrap();
    assert_eq!(report["profile"], "quick");
    assert_eq!(report["passed"], false);
    assert!(
        report["runs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["configuration"] == "none")
    );
    assert!(
        report["failures"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["kind"] == "error")
    );
}
