import Foundation

struct Conversation: Codable, Identifiable, Hashable, Sendable {
    var id = UUID()
    /// Set only when the user renames the conversation; otherwise the title follows the first message.
    var customTitle: String?
    var messages: [ChatMessage] = []
    var createdAt = Date()
    var updatedAt = Date()

    static let untitled = "New Chat"

    var title: String {
        if let customTitle, !customTitle.isEmpty { return customTitle }
        guard let first = messages.first(where: { $0.role == .user }) else { return Self.untitled }
        let line = first.content
            .split(whereSeparator: \.isNewline)
            .first { !$0.allSatisfy(\.isWhitespace) }
            .map { $0.trimmingCharacters(in: .whitespaces) } ?? ""
        if line.isEmpty { return Self.untitled }
        return line.count > 60 ? line.prefix(60) + "…" : line
    }
}

struct ChatMessage: Codable, Identifiable, Hashable, Sendable {
    enum Role: String, Codable, Sendable { case user, assistant }

    var id = UUID()
    var role: Role
    var content: String
    var reasoning: String?
    var stats: ReplyStats?
    /// The server's explanation when a reply failed; shown under the message.
    var error: String?
    var createdAt = Date()
}

struct ReplyStats: Codable, Hashable, Sendable {
    enum Ending: String, Codable, Sendable {
        case finished
        /// The reply reached the token limit (`finish_reason` "length").
        case lengthLimit
        case stopped
    }

    var completionTokens: Int
    var tokensPerSecond: Double?
    var thinkingSeconds: Double?
    var ending: Ending
}
