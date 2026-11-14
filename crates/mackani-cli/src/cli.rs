mod args;
mod help;
#[path = "product_commands.rs"]
mod product_commands;
mod terminal;

use crate::{
    Error, Result,
    server::{Config, Server},
};
use args::parse_args;
use product_commands::{bench, selfcheck};
use std::{
    io::{self, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use terminal::chat_params;
pub use terminal::{CHAT_HELP, chat_loop, visible};

fn stdout(text: &str) -> Result<()> {
    match io::stdout().lock().write_all(text.as_bytes()) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(error) => Err(Error::engine(error)),
    }
}

pub const BENCH_TASKS: [(&str, &str, bool); 4] = [
    (
        "prose",
        "Explain how a hash table works, including collisions and resizing, with a short example.",
        false,
    ),
    (
        "code",
        "Write a Python function that merges overlapping intervals and add a few tests for it.",
        false,
    ),
    (
        "structured",
        "Convert this CSV to a JSON array, with numeric ages: name,age\nAda,36\nLin,28",
        false,
    ),
    (
        "thinking",
        "Prove that the square root of 2 is irrational.",
        true,
    ),
];
pub async fn main(argv: Vec<String>) -> Result<()> {
    if argv.iter().any(|s| s == "--help" || s == "-h") {
        let command = argv.first().map(String::as_str).filter(|command| {
            matches!(*command, "serve" | "chat" | "info" | "bench" | "selfcheck")
        });
        return stdout(&help::text(command));
    }
    if argv == ["--version"] {
        return stdout(&format!("mackani {}\n", env!("CARGO_PKG_VERSION")));
    }
    let args = parse_args(argv)?;
    if !matches!(args.value("engine"), None | Some("real" | "stub")) {
        return Err(Error::request("--engine must be real or stub"));
    }
    match args.command.as_str() {
        "info" => {
            if args.is_stub() {
                let runner = args.runner()?;
                stdout(&format!(
                    "mackani {}\nengine   stub (deterministic demo, no inference)\nplatform {} {}\ncontext  {} tokens\ntokenizer  ok\n",
                    env!("CARGO_PKG_VERSION"),
                    std::env::consts::OS,
                    std::env::consts::ARCH,
                    runner.engine.context_capacity()
                ))?;
            } else {
                let info = args.real_config()?.inspect()?;
                stdout(&format!(
                    "{}\nModel files must remain unchanged while the program runs.\n",
                    serde_json::to_string_pretty(&info.to_json()).map_err(Error::engine)?
                ))?;
            }
            Ok(())
        }
        "bench" => bench(&args),
        "selfcheck" => selfcheck(&args),
        "serve" => {
            let mut cfg = Config {
                host: args.value("host").unwrap_or("127.0.0.1").into(),
                port: args.parse("port", 8765)?,
                api_key: args
                    .value("api-key")
                    .map(str::to_owned)
                    .or_else(|| std::env::var("MACKANI_API_KEY").ok()),
                ..Config::default()
            };
            cfg.defaults = args.defaults()?;
            cfg.defaults.max_tokens = args.parse("default-max-tokens", 32768)?;
            cfg.defaults.max_tokens_cap = args
                .value("max-tokens-cap")
                .map(|s| {
                    s.parse::<usize>()
                        .map_err(|_| Error::request("--max-tokens-cap has an invalid value"))
                })
                .transpose()?;
            cfg.validate()?;
            args.capacity()?;
            let engine_name = if args.is_stub() {
                "stub (deterministic demo)"
            } else {
                "real; model files must remain unchanged while running"
            };
            let keyed = cfg.api_key.as_ref().is_some_and(|s| !s.is_empty());
            if !args.is_stub() {
                eprintln!("{}", crate::real_engine::MODEL_FILE_STABILITY);
            }
            let lifeline = args.flag("exit-when-stdin-closes");
            let server = Server::bind(cfg, move || {
                let mut r = args.runner()?;
                if !args.flag("no-warmup") {
                    r.warmup()?;
                }
                Ok(r)
            })
            .await?;
            let addr = server.local_addr()?;
            eprintln!(
                "listening on http://{addr} (answers /health while the model loads)\nOpenAI-compatible API at http://{addr}/v1{}\nengine: {engine_name}; stop with Ctrl+C",
                if keyed { ", API key required" } else { "" }
            );
            server
                .run(async {
                    tokio::select! {
                        _ = termination() => {}
                        _ = stdin_closed(), if lifeline => {}
                    }
                    eprintln!("stopping ...");
                })
                .await
        }
        "chat" => {
            let p = chat_params(&args)?;
            if !args.is_stub() {
                eprintln!("Model files must remain unchanged while the program runs.");
            }
            let hide = args.flag("hide-thinking");
            let raw = args.flag("raw");
            let writing = Arc::new(AtomicBool::new(false));
            let cancel = Arc::new(AtomicBool::new(false));
            let worker_writing = writing.clone();
            let worker_cancel = cancel.clone();
            let (done, rx) = tokio::sync::oneshot::channel();
            std::thread::Builder::new()
                .name("mackani-terminal".into())
                .spawn(move || {
                    let result = (|| {
                        let mut runner = args.runner()?;
                        runner.warmup()?;
                        chat_loop(
                            &mut runner,
                            p,
                            hide,
                            raw,
                            io::stdin().lock(),
                            // Not locked across the loop: the interrupt handler
                            // writes too, while this thread waits for input.
                            io::stdout(),
                            &worker_writing,
                            &worker_cancel,
                        )
                    })();
                    let _ = done.send(result);
                })
                .map_err(Error::engine)?;
            tokio::pin!(rx);
            loop {
                tokio::select! {
                    result = &mut rx => {
                        return result.map_err(|_| Error::engine("terminal worker ended"))?;
                    }
                    signal = termination() => {
                        signal.map_err(Error::engine)?;
                        if writing.load(Ordering::Acquire) {
                            cancel.store(true, Ordering::Release);
                        } else {
                            stdout("\n")?;
                            return Ok(());
                        }
                    }
                }
            }
        }
        _ => Err(Error::request("unknown command")),
    }
}
/// Resolves when standard input reaches end of file, so a parent process
/// holding the other end of the pipe cannot leave the server behind.
async fn stdin_closed() {
    let (done, closed) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let _ = io::copy(&mut io::stdin().lock(), &mut io::sink());
        let _ = done.send(());
    });
    let _ = closed.await;
}
async fn termination() -> io::Result<()> {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result,
            _ = term.recv() => Ok(()),
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await
    }
}
