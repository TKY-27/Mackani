import Foundation
import Observation

enum ServerState: Equatable {
    case noFolder
    /// Loading the model; this takes 10 to 40 seconds.
    case starting
    case ready(ServerEndpoint)
    /// The server could not start or stopped on its own. `output` holds its last lines.
    case failed(reason: String, output: [String])
}

/// Starts, watches and stops the bundled `mackani serve`. There is never more than one server:
/// the machine cannot hold the model twice, so a new one starts only after the old one has exited.
@MainActor
@Observable
final class ServerController {
    private(set) var state: ServerState = .noFolder
    private(set) var modelFolder: URL?

    @ObservationIgnored private var server: ServerProcess?

    private static let folderKey = "modelFolderPath"
    private let environment = ProcessInfo.processInfo.environment

    init() {
        modelFolder = UserDefaults.standard.string(forKey: Self.folderKey).map { URL(filePath: $0, directoryHint: .isDirectory) }
    }

    var endpoint: ServerEndpoint? {
        if case .ready(let endpoint) = state { endpoint } else { nil }
    }

    func setModelFolder(_ folder: URL) async {
        modelFolder = folder
        UserDefaults.standard.set(folder.path(percentEncoded: false), forKey: Self.folderKey)
        await restart()
    }

    func restart() async {
        await stop()
        start()
    }

    func start() {
        guard server == nil else { return }
        let modelArguments: [String]
        if let devArguments = Self.developmentArguments(environment) {
            modelArguments = devArguments
        } else if let modelFolder {
            modelArguments = ["--model-dir", modelFolder.path(percentEncoded: false)]
        } else {
            state = .noFolder
            return
        }

        let binary = serverBinary
        guard FileManager.default.isExecutableFile(atPath: binary.path(percentEncoded: false)), !isOwnExecutable(binary) else {
            state = .failed(reason: "The server program was not found at \(binary.path(percentEncoded: false)).", output: [])
            return
        }

        let launched: ServerProcess
        do {
            launched = try ServerProcess(
                binary: binary,
                arguments: ["serve"] + modelArguments,
                port: ServerProcess.freePort(),
                apiKey: ServerProcess.makeAPIKey()
            )
        } catch {
            state = .failed(reason: "The server could not be started: \(error.localizedDescription)", output: [])
            return
        }
        server = launched
        state = .starting
        Task { [weak self] in await self?.watch(launched) }
    }

    /// Stops the server and waits until it has exited, killing it if it ignores SIGTERM for 10 seconds.
    private func stop() async {
        guard let running = server else { return }
        server = nil
        running.terminate()
        let killer = Task {
            try await Task.sleep(for: .seconds(10))
            running.kill()
        }
        _ = await running.exit.value
        killer.cancel()
    }

    /// For quitting: asks the server to exit without waiting and returns it, so the caller can wait
    /// for it off the main actor.
    func terminateForQuit() -> ServerProcess? {
        guard let running = server else { return nil }
        server = nil
        running.terminate()
        return running
    }

    private func watch(_ launched: ServerProcess) async {
        let health = Task {
            while !Task.isCancelled {
                if await ChatClient.isHealthy(port: launched.port) { return true }
                try? await Task.sleep(for: .milliseconds(400))
            }
            return false
        }
        // Polling ends when the server answers or when it exits, whichever comes first.
        Task { _ = await launched.exit.value; health.cancel() }
        if await health.value, server === launched {
            state = .ready(ServerEndpoint(port: launched.port, apiKey: launched.apiKey))
        }
        let status = await launched.exit.value
        guard server === launched else { return } // Stopped on purpose.
        server = nil
        state = .failed(reason: "The server stopped (exit status \(status)).", output: launched.output.snapshot)
    }

    /// Debug builds only: `MACKANI_DEV_SERVER_ARGS` replaces `--model-dir FOLDER`, for example
    /// with `--engine stub`.
    private static func developmentArguments(_ environment: [String: String]) -> [String]? {
        #if DEBUG
        // shortcut: split on spaces only; enough for flags without quoted values.
        return environment["MACKANI_DEV_SERVER_ARGS"]?.split(separator: " ").map(String.init)
        #else
        return nil
        #endif
    }

    /// The server sits next to the app's own executable. Debug builds let
    /// `MACKANI_SERVER_BINARY` name another.
    private var serverBinary: URL {
        #if DEBUG
        if let override = environment["MACKANI_SERVER_BINARY"] {
            return URL(filePath: override)
        }
        #endif
        let executable = Bundle.main.executableURL ?? URL(filePath: CommandLine.arguments[0])
        return executable.deletingLastPathComponent().appending(path: "mackani")
    }

    /// On a case-insensitive volume, "mackani" next to an executable called "Mackani" (as in
    /// `.build/release`) is the app itself; launching it would open a second app, not a server.
    private func isOwnExecutable(_ binary: URL) -> Bool {
        guard let executable = Bundle.main.executableURL,
              let own = try? executable.resourceValues(forKeys: [.fileResourceIdentifierKey]).fileResourceIdentifier,
              let other = try? binary.resourceValues(forKeys: [.fileResourceIdentifierKey]).fileResourceIdentifier
        else { return false }
        return own.isEqual(other)
    }
}
