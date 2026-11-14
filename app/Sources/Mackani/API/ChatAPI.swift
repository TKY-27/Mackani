import Foundation

/// Where the running server listens and the key it expects.
struct ServerEndpoint: Sendable, Equatable {
    var port: Int
    var apiKey: String

    var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }
}

/// The body of `POST /v1/chat/completions`.
struct ChatRequest: Encodable, Equatable {
    struct Message: Encodable, Equatable {
        var role: String
        var content: String
    }

    struct StreamOptions: Encodable, Equatable {
        var includeUsage = true
        enum CodingKeys: String, CodingKey { case includeUsage = "include_usage" }
    }

    var model = "qwen3.8-27b"
    var messages: [Message]
    var stream = true
    var streamOptions = StreamOptions()
    var enableThinking: Bool

    enum CodingKeys: String, CodingKey {
        case model, messages, stream
        case streamOptions = "stream_options"
        case enableThinking = "enable_thinking"
    }

    /// Sends the conversation as plain text: reasoning is never resent, and replies that failed
    /// before producing any text are left out.
    init(history: [ChatMessage], enableThinking: Bool) {
        messages = history
            .filter { !$0.content.isEmpty }
            .map { Message(role: $0.role.rawValue, content: $0.content) }
        self.enableThinking = enableThinking
    }
}

/// The `usage` object of the last streamed chunk.
struct Usage: Codable, Equatable, Sendable {
    struct CompletionDetails: Codable, Equatable, Sendable {
        var reasoningTokens: Int?
        enum CodingKeys: String, CodingKey { case reasoningTokens = "reasoning_tokens" }
    }

    struct PromptDetails: Codable, Equatable, Sendable {
        var cachedTokens: Int?
        enum CodingKeys: String, CodingKey { case cachedTokens = "cached_tokens" }
    }

    var promptTokens: Int
    var completionTokens: Int
    var totalTokens: Int
    var completionTokensDetails: CompletionDetails?
    var promptTokensDetails: PromptDetails?

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case completionTokensDetails = "completion_tokens_details"
        case promptTokensDetails = "prompt_tokens_details"
    }
}

/// An error the server reported, as `{"error":{"message":…,"code":…}}`.
struct ServerError: LocalizedError, Equatable {
    var message: String
    /// A string such as `context_length_exceeded`, or the HTTP status as text.
    var code: String?

    var errorDescription: String? { message }

    init(message: String, code: String? = nil) {
        self.message = message
        self.code = code
    }

    /// Decodes an error body; falls back to the raw text when the body is not the expected JSON.
    init(body: Data, status: Int) {
        if let envelope = try? JSONDecoder().decode(ErrorEnvelope.self, from: body) {
            self.init(message: envelope.error.message, code: envelope.error.code?.text)
        } else {
            let text = String(decoding: body, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            self.init(message: text.isEmpty ? "The server answered with status \(status)." : text, code: String(status))
        }
    }
}

struct ErrorEnvelope: Decodable {
    struct Body: Decodable {
        var message: String
        var code: Code?
    }

    /// The server sends `code` as a string (`"context_length_exceeded"`) or a number (`401`).
    enum Code: Decodable {
        case text(String)
        case number(Int)

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let number = try? container.decode(Int.self) {
                self = .number(number)
            } else {
                self = .text(try container.decode(String.self))
            }
        }

        var text: String {
            switch self {
            case .text(let text): text
            case .number(let number): String(number)
            }
        }
    }

    var error: Body
}
