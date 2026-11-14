import Darwin
import Foundation
import Synchronization

/// One launched `mackani serve` process and the pipes that tie it to the app.
///
/// The server's standard input is a pipe whose write end only this app holds. The server is started
/// with `--exit-when-stdin-closes`, so it exits when the app closes the pipe or dies, even by SIGKILL.
@MainActor
final class ServerProcess {
    let port: Int
    let apiKey: String
    let output = OutputTail()
    /// Completes with the termination status once the process has exited and its output is read.
    let exit: Task<Int32, Never>
    /// Signalled once the process has exited, for waiting without the main actor (when quitting).
    nonisolated let exited = DispatchSemaphore(value: 0)
    nonisolated let pid: pid_t

    private let process: Process
    private let stdinWriter: FileHandle

    init(binary: URL, arguments: [String], port: Int, apiKey: String) throws {
        let process = Process()
        process.executableURL = binary
        process.arguments = arguments + ["--port", String(port), "--exit-when-stdin-closes"]
        var environment = ProcessInfo.processInfo.environment
        environment["MACKANI_API_KEY"] = apiKey
        process.environment = environment

        let stdin = Pipe()
        let output = Pipe()
        process.standardInput = stdin
        process.standardOutput = output
        process.standardError = output

        let (exitStream, exitContinuation) = AsyncStream<Int32>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let exited = self.exited
        process.terminationHandler = { finished in
            exitContinuation.yield(finished.terminationStatus)
            exitContinuation.finish()
            exited.signal()
        }

        try process.run()
        // Only the server may hold these ends; keeping them open here would hide EOF from each side.
        try? stdin.fileHandleForReading.close()
        try? output.fileHandleForWriting.close()

        let tail = self.output
        let reader = Task.detached {
            do {
                for try await line in output.fileHandleForReading.bytes.lines { tail.append(line) }
            } catch {
                tail.append("(could not read the server's output: \(error.localizedDescription))")
            }
        }
        self.exit = Task {
            var status: Int32 = 0
            for await value in exitStream { status = value }
            await reader.value
            return status
        }
        self.process = process
        self.pid = process.processIdentifier
        self.stdinWriter = stdin.fileHandleForWriting
        self.port = port
        self.apiKey = apiKey
    }

    /// Asks the server to exit: SIGTERM, and end of file on its standard input.
    func terminate() {
        process.terminate()
        try? stdinWriter.close()
    }

    nonisolated func kill() {
        Darwin.kill(pid, SIGKILL)
    }

    /// Asks a free TCP port of the loopback interface from the system.
    nonisolated static func freePort() throws -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        defer { close(fd) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        address.sin_port = 0
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { generic in
                bind(fd, generic, length) == 0 && getsockname(fd, generic, &length) == 0
            }
        }
        guard bound else { throw POSIXError(.init(rawValue: errno) ?? .EADDRINUSE) }
        return Int(UInt16(bigEndian: address.sin_port))
    }

    /// A fresh 256-bit key for this launch.
    nonisolated static func makeAPIKey() -> String {
        var generator = SystemRandomNumberGenerator()
        return (0..<32).map { _ in String(format: "%02x", UInt8.random(in: .min ... .max, using: &generator)) }.joined()
    }
}

/// The last lines the server printed, kept for the "failed" state.
final class OutputTail: Sendable {
    private let lines = Mutex<[String]>([])
    private let limit = 40

    func append(_ line: String) {
        lines.withLock { lines in
            lines.append(line)
            if lines.count > limit { lines.removeFirst(lines.count - limit) }
        }
    }

    var snapshot: [String] { lines.withLock { $0 } }
}
