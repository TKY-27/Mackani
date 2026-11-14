use std::{
    io::{BufRead, BufReader, Read, Write},
    process::{Command, Output, Stdio},
    time::Duration,
};
const PATIENCE: Duration = Duration::from_secs(30);
fn command() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_mackani"));
    for key in [
        "MACKANI_API_KEY",
        "MACKANI_MODEL_DIR",
        "MACKANI_THINKING_BUDGET",
    ] {
        c.env_remove(key);
    }
    c
}
fn run(args: &[&str], input: &str) -> Output {
    let mut child = command()
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}
fn text(r: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&r.stdout),
        String::from_utf8_lossy(&r.stderr)
    )
}
#[test]
fn help_and_version() {
    let r = run(&["--version"], "");
    assert!(r.status.success());
    assert!(text(&r).contains("mackani"));
    let r = run(&["serve", "--engine", "stub", "--port", "0", "--help"], "");
    assert!(r.status.success());
    assert!(text(&r).contains("--port"));
    assert!(!run(&[], "").status.success());
}
#[test]
fn info() {
    let r = run(&["info", "--engine", "stub"], "");
    assert!(r.status.success(), "{}", text(&r));
    assert!(text(&r).contains("tokenizer  ok"));
    assert!(text(&r).contains("deterministic demo"));
    let r = run(&["info"], "");
    assert!(!r.status.success());
    assert!(text(&r).contains("--model-dir is required for the real engine"));
    let r = run(
        &[
            "info",
            "--engine",
            "stub",
            "--model-dir",
            "/nonexistent/mackani-model",
        ],
        "",
    );
    assert!(!r.status.success());
    assert!(text(&r).contains("tokenizer.json"));
}
#[test]
fn chat() {
    let r = run(
        &["chat", "--engine", "stub", "--ctx", "4096"],
        "/think off\nhello\n/reset\n/bogus\n/exit\n",
    );
    assert!(r.status.success(), "{}", text(&r));
    for needle in [
        "(thinking off)",
        "tokens/s",
        "complete",
        "(conversation cleared)",
        "commands:",
    ] {
        assert!(text(&r).contains(needle), "{}", text(&r));
    }
}
#[test]
fn chat_ends_on_end_of_input() {
    let r = run(
        &["chat", "--engine", "stub", "--no-think", "--ctx", "4096"],
        "hi\n",
    );
    assert!(r.status.success(), "{}", text(&r));
    assert!(text(&r).contains("complete"));
}
#[test]
fn bad_inputs() {
    let r = run(&["chat"], "");
    assert!(!r.status.success());
    assert!(text(&r).contains("--model-dir is required for the real engine"));
    let r = run(
        &[
            "serve", "--engine", "stub", "--host", "0.0.0.0", "--port", "0",
        ],
        "",
    );
    assert!(!r.status.success());
    assert!(text(&r).contains("without --api-key"));
    let r = run(
        &[
            "chat",
            "--engine",
            "stub",
            "--model-dir",
            "/nonexistent/mackani-model",
        ],
        "",
    );
    assert!(!r.status.success());
    assert!(text(&r).contains("tokenizer.json"));
    for args in [
        vec!["chat", "--engine", "stub", "--effort", "bad"],
        vec!["chat", "--engine", "stub", "--max-tokens", "0"],
        vec!["serve", "--engine", "stub", "--port", "65536"],
        vec!["chat", "--engine", "stub", "--ctx", "64"],
        vec!["chat", "--engine", "stub", "--unknown"],
        vec!["chat", "--engine"],
        vec!["serve", "--engine", "other"],
    ] {
        assert!(!run(&args, "").status.success(), "{args:?}");
    }
}
#[test]
fn real_commands_require_a_model_folder() {
    for cmd in ["bench", "selfcheck"] {
        let r = run(&[cmd], "");
        assert!(!r.status.success());
        assert!(text(&r).contains("--model-dir is required for the real engine"));
    }
}
#[test]
fn chat_thinking_visibility_effort_and_aliases() {
    let r = run(
        &["chat", "--engine", "stub", "--effort", "low"],
        "/effort medium\nhi\n/q\n",
    );
    assert!(r.status.success(), "{}", text(&r));
    for s in [
        "[thinking]",
        "[/thinking]",
        "(reasoning effort medium)",
        "Stub response",
    ] {
        assert!(text(&r).contains(s), "{}", text(&r));
    }
    let r = run(&["chat", "--engine", "stub", "--raw"], "hi\n/q\n");
    assert!(r.status.success(), "{}", text(&r));
    assert!(text(&r).contains("Stub response"));
    let r = run(
        &["chat", "--engine", "stub", "--hide-thinking"],
        "hi\n/quit\n",
    );
    assert!(r.status.success());
    assert!(!text(&r).contains("[thinking]"));
    assert!(text(&r).contains("Stub response"));
    let r = run(
        &["chat", "--engine", "stub", "--max-tokens", "2"],
        "hi\n/exit\n",
    );
    assert!(r.status.success());
    assert!(text(&r).contains("no answer was written"));
}
#[test]
fn serve_port_in_use() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port().to_string();
    let r = run(&["serve", "--engine", "stub", "--port", &port], "");
    assert!(!r.status.success());
    assert!(text(&r).contains("cannot listen"));
}
struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn http(addr: &str, request: &str) -> String {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(PATIENCE)).unwrap();
    s.write_all(request.as_bytes()).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}
#[test]
fn serve_end_to_end() {
    let child = command()
        .args([
            "serve",
            "--engine",
            "stub",
            "--port",
            "0",
            "--ctx",
            "4096",
            "--api-key",
            "k3y",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut child = Child(child);
    let mut err = BufReader::new(child.0.stderr.take().unwrap());
    let mut line = String::new();
    err.read_line(&mut line).unwrap();
    assert!(line.starts_with("listening on http://"), "{line}");
    let addr = line
        .trim_start_matches("listening on http://")
        .split_whitespace()
        .next()
        .unwrap();
    // Another process: /health is the only readiness signal. Generous for a loaded machine.
    let deadline = std::time::Instant::now() + PATIENCE;
    while !http(
        addr,
        &format!("GET /health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"),
    )
    .contains("\"status\":\"ok\"")
    {
        assert!(
            std::time::Instant::now() < deadline,
            "server never became ready"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"enable_thinking\":false}";
    let r = http(
        addr,
        &format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer k3y\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ),
    );
    assert!(r.starts_with("HTTP/1.1 200"));
    assert!(r.contains("\"completion_tokens\":8"));
    assert!(r.contains("\"finish_reason\":\"length\""));
    assert!(
        http(
            addr,
            &format!("GET /v1/models HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n")
        )
        .starts_with("HTTP/1.1 401")
    );
    let status = Command::new("/bin/kill")
        .args(["-INT", &child.0.id().to_string()])
        .status()
        .unwrap();
    assert!(status.success());
    let mut rest = String::new();
    err.read_to_string(&mut rest).unwrap();
    assert!(child.0.wait().unwrap().success());
    assert!(rest.contains("stopping"));
    assert!(rest.contains("API key required"));
}
/// Waits for the child to exit; kills it and fails the test after PATIENCE.
fn exits_within_patience(child: &mut Child) -> std::process::ExitStatus {
    let deadline = std::time::Instant::now() + PATIENCE;
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            return status;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the process did not exit after the interrupt"
        );
        std::thread::sleep(Duration::from_millis(10)); // polling the exit only
    }
}
// An idle terminal worker must not hold the output lock needed by the signal handler.
#[test]
fn interrupt_at_the_prompt_exits() {
    let child = command()
        .args(["chat", "--engine", "stub", "--ctx", "4096"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut child = Child(child);
    // Kept open: the chat waits at its prompt for a line that never comes.
    let _stdin = child.0.stdin.take().unwrap();
    let mut out = child.0.stdout.take().unwrap();
    let mut seen = Vec::new();
    let mut byte = [0_u8; 1];
    while !seen.ends_with(b"\n> ") {
        assert_eq!(out.read(&mut byte).unwrap(), 1, "no prompt appeared");
        seen.push(byte[0]);
    }
    let status = Command::new("/bin/kill")
        .args(["-INT", &child.0.id().to_string()])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(exits_within_patience(&mut child).success());
}
// Loader paths and invalid arguments can contain terminal escape sequences.
#[test]
fn top_level_errors_show_control_characters_visibly() {
    let folder = "/nonexistent/\u{1b}]0;owned\u{7}\u{1b}[2J\u{202e}x";
    for args in [
        vec!["info", "--engine", "stub", "--model-dir", folder],
        vec!["chat", "--engine", "stub", "--model-dir", folder],
        vec!["chat", "--engine", "stub", "--raw", "--model-dir", folder],
        vec!["info", "--engine", "stub", "--bad\u{1b}[2J"],
    ] {
        let r = run(&args, "");
        assert!(!r.status.success());
        let err = String::from_utf8(r.stderr).unwrap();
        assert!(err.starts_with("error: "), "{err:?}");
        assert!(err.contains("\\x1b[2J"), "{err:?}");
        assert!(
            !err.contains(|c: char| c.is_control() && c != '\n' || c == '\u{202e}'),
            "{err:?}"
        );
    }
}

#[test]
fn command_help_lists_options_and_separates_diagnostics() {
    for (command, option) in [
        ("serve", "--exit-when-stdin-closes"),
        ("chat", "--hide-thinking"),
        ("info", "--model-dir"),
        ("bench", "--json"),
        ("selfcheck", "--reference"),
    ] {
        let result = run(&[command, "--help"], "");
        assert!(result.status.success(), "{}", text(&result));
        let help = String::from_utf8(result.stdout).unwrap();
        assert!(help.contains(&format!("Usage: mackani {command} [OPTIONS]")));
        for expected in [
            option,
            "Diagnostics:",
            "--kernel portable",
            "--cpu-stages reference",
            "development aid",
        ] {
            assert!(
                help.contains(expected),
                "{command}: missing {expected}: {help}"
            );
        }
        for removed in [
            "--max-ctx",
            "--metal-mac",
            "MACKANI_BACKEND",
            "MACKANI_KERNEL",
            "MACKANI_CPU_STAGES",
            "MACKANI_METAL_CPU_SHARE",
        ] {
            assert!(
                !help.contains(removed),
                "{command}: stale {removed}: {help}"
            );
        }
    }
    for removed in ["--max-ctx", "--metal-mac"] {
        let result = run(&["info", "--engine", "stub", removed, "1"], "");
        assert!(!result.status.success());
        assert!(text(&result).contains(&format!("unknown option: {removed}")));
    }
}

#[cfg(unix)]
#[test]
fn help_version_and_info_exit_successfully_when_stdout_is_closed() {
    use std::os::{fd::OwnedFd, unix::net::UnixStream};

    for args in [
        vec!["--help"],
        vec!["serve", "--help"],
        vec!["--version"],
        vec!["info", "--engine", "stub"],
    ] {
        let (reader, writer) = UnixStream::pair().unwrap();
        drop(reader);
        let result = command()
            .args(&args)
            .stdout(Stdio::from(OwnedFd::from(writer)))
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(result.status.success(), "{args:?}: {}", text(&result));
        assert!(result.stderr.is_empty(), "{args:?}: {}", text(&result));
    }
}

#[test]
fn stdin_closure_stops_only_a_server_with_the_lifeline_flag() {
    for lifeline in [true, false] {
        let mut cmd = command();
        cmd.args(["serve", "--engine", "stub", "--port", "0"]);
        if lifeline {
            cmd.arg("--exit-when-stdin-closes");
        }
        let mut child = Child(
            cmd.stdin(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let input = child.0.stdin.take().unwrap();
        let mut errors = BufReader::new(child.0.stderr.take().unwrap());
        let mut line = String::new();
        errors.read_line(&mut line).unwrap();
        assert!(line.starts_with("listening on http://"), "{line}");
        let address = line
            .trim_start_matches("listening on http://")
            .split_whitespace()
            .next()
            .unwrap();
        let health =
            format!("GET /health HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n");
        let deadline = std::time::Instant::now() + PATIENCE;
        while !http(address, &health).contains("\"status\":\"ok\"") {
            assert!(
                std::time::Instant::now() < deadline,
                "server never became ready"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        drop(input);
        if lifeline {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            loop {
                if let Some(status) = child.0.try_wait().unwrap() {
                    assert!(status.success(), "stdin closure failed: {status}");
                    break;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "server did not exit after stdin closed"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        } else {
            let deadline = std::time::Instant::now() + Duration::from_secs(1);
            while std::time::Instant::now() < deadline {
                assert!(
                    child.0.try_wait().unwrap().is_none(),
                    "closed stdin stopped server without flag"
                );
                assert!(http(address, &health).starts_with("HTTP/1.1 200"));
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
