import Foundation
import os

/// Keeps one JSON file per conversation in a directory. Writes are atomic and run in order on a
/// private serial queue, so the main thread never waits on the disk.
final class ConversationStore: Sendable {
    let directory: URL
    private let queue = DispatchQueue(label: "io.github.tky-27.mackani.store")
    private let log = Logger(subsystem: "io.github.tky-27.mackani", category: "store")

    static var defaultDirectory: URL {
        URL.applicationSupportDirectory.appending(path: "Mackani", directoryHint: .isDirectory)
    }

    init(directory: URL = ConversationStore.defaultDirectory) {
        self.directory = directory
    }

    func loadAll() async -> [Conversation] {
        await withCheckedContinuation { continuation in
            queue.async { continuation.resume(returning: self.readAll()) }
        }
    }

    func save(_ conversation: Conversation) {
        queue.async {
            do { try self.write(conversation) } catch {
                self.log.error("Could not save conversation \(conversation.id): \(error)")
            }
        }
    }

    func delete(id: UUID) {
        queue.async {
            do { try FileManager.default.removeItem(at: self.fileURL(for: id)) } catch CocoaError.fileNoSuchFile {
                // Never saved (it had no messages yet).
            } catch {
                self.log.error("Could not delete conversation \(id): \(error)")
            }
        }
    }

    /// Waits until every queued write and deletion has finished.
    func flush() async {
        await withCheckedContinuation { continuation in queue.async { continuation.resume() } }
    }

    /// The blocking form of `flush`, for quitting.
    func waitForPendingWrites() {
        queue.sync {}
    }

    func write(_ conversation: Conversation) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = try Self.encoder.encode(conversation)
        try data.write(to: fileURL(for: conversation.id), options: .atomic)
    }

    func readAll() -> [Conversation] {
        let files: [URL]
        do {
            files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        } catch CocoaError.fileReadNoSuchFile {
            return []
        } catch {
            log.error("Could not list conversations: \(error)")
            return []
        }
        return files.filter { $0.pathExtension == "json" }.compactMap { url in
            do {
                return try Self.decoder.decode(Conversation.self, from: Data(contentsOf: url))
            } catch {
                log.error("Skipping unreadable conversation \(url.lastPathComponent): \(error)")
                return nil
            }
        }
    }

    private func fileURL(for id: UUID) -> URL {
        directory.appending(path: "\(id.uuidString).json")
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }()

    private static let decoder = JSONDecoder()
}
