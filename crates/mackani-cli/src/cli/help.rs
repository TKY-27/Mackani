const COMMON: &str = "Options:
  --model-dir FOLDER    model folder, or MACKANI_MODEL_DIR
  --engine real|stub    default real; stub is a development aid with no inference
  --ctx N               context capacity, default 32768 (minimum 66)
  --kv int8|int8-profile|kv5  key/value cache, default kv5
  --draft none|ngram|neural|both  speculative decoding, default both
  --k N                 most guesses per pass, default 7 (neural/both: 1..7)
  --depth adaptive|fixed  neural guess depth policy, default adaptive
  --backend metal|cpu   default Metal when available, otherwise CPU
  --threads N           CPU workers, default one per physical core
  --cpu-share auto|off|F  Metal CPU share, F in 0..=0.9, default auto
                        auto uses the measured Apple M5 table, otherwise off
  --prefill-width N     cancellable prefill pass width, default 16
  --seed0 N             first automatic request seed, default 1
  --help, -h            show help
Diagnostics:
  --kernel portable     use the portable CPU kernel instead of automatic selection
  --cpu-stages reference  use reference CPU stages instead of automatic fast paths
";
const SERVE: &str = "Serve options:
  --host HOST           bind address, default 127.0.0.1
  --port PORT           bind port, default 8765; 0 selects an available port
  --api-key KEY         or MACKANI_API_KEY; required for non-loopback binds
  --default-max-tokens N  default request output limit, default 32768
  --max-tokens-cap N    optional server cap on request output limits
  --no-warmup           skip engine warmup
  --exit-when-stdin-closes  stop successfully when standard input closes
";
const CHAT: &str = "Chat options:
  --system TEXT         initial system message
  --max-tokens N        output limit, default 32768
  --temperature N --top-p N --top-k N --min-p N  override model sampling defaults
  --seed N              request seed (otherwise advances from --seed0)
  --no-think            disable thinking (enabled by default)
  --effort low|medium|xhigh  reasoning effort
  --hide-thinking       display only the answer
  --no-loop-guard       disable repeated-token protection
  --raw                 display model text and chat errors without escaping controls
Terminal commands:
  /exit, /quit, /q       quit
  /reset                forget the conversation
  /think on|off         change thinking mode
  /effort low|medium|xhigh  change reasoning effort
  /help                 show terminal commands
Ctrl+C stops an answer or exits at the prompt. Input and output support pipes.
";
const BENCH: &str = "Bench options:
  --n N                 tokens per task, default 512 (greedy, ignores EOS)
  --json FILE           save metrics and configuration
Repeat --draft, --k and --depth to compare their Cartesian product.
";
const SELFCHECK: &str = "Selfcheck options:
  --reference FILE      reference token hashes, default MODEL_DIR/selfcheck.json
";
const ENVIRONMENT: &str = "Environment:
  MACKANI_MODEL_DIR      model folder when --model-dir is omitted
  MACKANI_API_KEY        server key when --api-key is omitted
  MACKANI_THINKING_BUDGET  default chat/server thinking budget; 0 is unlimited
Model files must remain unchanged while the program runs.
";

pub(super) fn text(command: Option<&str>) -> String {
    let mut help = match command {
        Some(command) => format!("mackani {command} - {}\nUsage: mackani {command} [OPTIONS]\n", description(command)),
        None => "mackani - local chat and OpenAI-compatible server\nUsage: mackani COMMAND [OPTIONS]\nCommands:\n  serve      run the HTTP API\n  chat       chat in the terminal\n  info       inspect model, backend and memory estimates without loading weights\n  bench      benchmark fixed tasks and decoding configurations\n  selfcheck  compare greedy token output with reference hashes\n  --version  show version\n".into(),
    };
    help.push_str(COMMON);
    match command {
        Some("serve") => help.push_str(SERVE),
        Some("chat") => help.push_str(CHAT),
        Some("bench") => help.push_str(BENCH),
        Some("selfcheck") => help.push_str(SELFCHECK),
        Some(_) => {}
        None => {
            for section in [SERVE, CHAT, BENCH, SELFCHECK] {
                help.push_str(section);
            }
        }
    }
    help.push_str(ENVIRONMENT);
    help
}

fn description(command: &str) -> &'static str {
    match command {
        "serve" => "run the HTTP API",
        "chat" => "chat in the terminal",
        "info" => "inspect model, backend and memory estimates without loading weights",
        "bench" => "benchmark fixed tasks and decoding configurations",
        "selfcheck" => "compare greedy token output with reference hashes",
        _ => unreachable!(),
    }
}
