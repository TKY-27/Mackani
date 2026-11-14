# Mackani for Mac

A native macOS chat app (SwiftUI) for the local `mackani serve` server. macOS 26 or later,
Apple Silicon.

![Mackani in the light appearance](docs/screenshot-light.png)

## Build

```sh
cargo build --release -p mackani   # from the repository root: the server
cd app
./build-app.sh                     # or: ./build-app.sh /path/to/mackani
open build/Mackani.app
```

`build-app.sh` runs `swift build -c release`, assembles `build/Mackani.app` with the server copied
next to the app's executable (`Contents/MacOS/mackani`), and signs the bundle ad hoc. The server
path can also be given with `MACKANI_SERVER_BINARY`.

The package also opens in Xcode (`open Package.swift`). Tests: `swift test`.

## What the app starts

On launch the app starts one `mackani serve --model-dir FOLDER --port PORT --exit-when-stdin-closes`
on a free loopback port, with a fresh API key per launch in `MACKANI_API_KEY`. The server's standard
input is a pipe held by the app, so the server exits when the app quits, crashes or is killed.
Changing the model folder stops the server and waits for it to exit before starting a new one.

Loading the model takes 10 to 40 seconds. If the server stops, the app shows its last lines of
output, which explain missing files or a memory shortfall.

## Where data lives

- Conversations: `~/Library/Application Support/Mackani/`, one JSON file per conversation.
- Settings (model folder, thinking on or off): the `io.github.tky-27.mackani` defaults domain.

## Keyboard

Return sends, Shift-Return starts a new line, ⌘N starts a new chat, ⌘. stops a reply, ⌘, opens
Settings.
