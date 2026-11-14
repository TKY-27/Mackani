# Security policy

## Scope

Mackani loads local binary model files and runs a local HTTP server. Reports about the Rust model reader, engine, CLI, HTTP server or native app belong here.

The server binds to loopback by default. A non-loopback bind is refused without an API key. `--api-key` enables bearer-key authentication; `/health` does not require a key.

On loopback, the server checks Host against the local address and bound port. For POST requests it checks any Origin and Sec-Fetch-Site headers to reject cross-site requests. A keyed non-loopback bind bypasses these Host/Origin/Fetch checks and still requires the key for the API. This distinction matters when choosing how to run it.

The server bounds request bodies, parsed JSON, prompt size, pending memory, connections and queue waits. It applies transport and streaming timeouts. One engine request runs at a time. See the [HTTP reference and exact limits](crates/mackani-cli/README.md#security-and-transport-limits).

Prefer the `MACKANI_API_KEY` environment variable to `--api-key`: command-line arguments are visible to other users of the machine. If the engine thread panics, the server process exits. The app starts its server on a free loopback port with a fresh key per launch and opens only `http` and `https` links from replies.

Load model files from a source you trust. Do not change or replace them while Mackani is running; the engine maps them read only.

## What is not a goal

Mackani is intended for a local user. Multi-user isolation and exposure to the internet are not goals. It provides neither separate user accounts nor TLS. An API key does not provide user isolation. Do not expose the server directly to the internet.

## Private reports

Use GitHub's private vulnerability reporting for [TKY-27/Mackani](https://github.com/TKY-27/Mackani/security/advisories/new): open the repository's Security tab and choose “Report a vulnerability”. Do not put vulnerability details or credentials in a public issue.

Include the affected version, model if relevant, reproduction steps and expected impact. Remove secrets and private data. There is no guaranteed acknowledgement, fix or support timeline.
