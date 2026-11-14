import Foundation

/// One piece of a streamed reply.
enum StreamEvent: Equatable, Sendable {
    case reasoning(String)
    case content(String)
    case finish(reason: String)
    case usage(Usage)
    case done
}

/// Turns server-sent-event lines from the chat completions stream into `StreamEvent`s.
///
/// The server writes each event as a single `data:` line, so every line is decoded on its own.
enum SSEParser {
    /// Returns the events carried by one line; comments, blank lines and other fields give none.
    /// Throws `ServerError` for an error event and `DecodingError` for malformed JSON.
    static func events(fromLine line: some StringProtocol) throws -> [StreamEvent] {
        guard line.hasPrefix("data:") else { return [] }
        var payload = line.dropFirst(5)
        if payload.first == " " { payload = payload.dropFirst() }
        if payload == "[DONE]" { return [.done] }

        let chunk = try JSONDecoder().decode(Chunk.self, from: Data(payload.utf8))
        if let error = chunk.error {
            throw ServerError(message: error.message, code: error.code?.text)
        }
        var events: [StreamEvent] = []
        for choice in chunk.choices ?? [] {
            if let reasoning = choice.delta?.reasoningContent, !reasoning.isEmpty {
                events.append(.reasoning(reasoning))
            }
            if let content = choice.delta?.content, !content.isEmpty {
                events.append(.content(content))
            }
            if let reason = choice.finishReason {
                events.append(.finish(reason: reason))
            }
        }
        if let usage = chunk.usage {
            events.append(.usage(usage))
        }
        return events
    }

    private struct Chunk: Decodable {
        var choices: [Choice]?
        var usage: Usage?
        var error: ErrorEnvelope.Body?
    }

    private struct Choice: Decodable {
        var delta: Delta?
        var finishReason: String?
        enum CodingKeys: String, CodingKey {
            case delta
            case finishReason = "finish_reason"
        }
    }

    private struct Delta: Decodable {
        var content: String?
        var reasoningContent: String?
        enum CodingKeys: String, CodingKey {
            case content
            case reasoningContent = "reasoning_content"
        }
    }
}
