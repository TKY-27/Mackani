import Foundation
import Observation

/// The assistant reply that is arriving right now. It lives outside the conversation until it ends,
/// so each token only invalidates the views of this one reply.
@MainActor
@Observable
final class StreamingReply {
    let conversationID: UUID
    private(set) var answer = ""
    private(set) var reasoning = ""
    /// Number of deltas received; each delta is about one token.
    private(set) var deltaCount = 0
    private(set) var firstDeltaAt: Date?
    private(set) var answerStartedAt: Date?
    private(set) var usage: Usage?
    private(set) var finishReason: String?

    @ObservationIgnored private var markdown = IncrementalMarkdown()

    init(conversationID: UUID) {
        self.conversationID = conversationID
    }

    func apply(_ event: StreamEvent, at now: Date = Date()) {
        switch event {
        case .reasoning(let text):
            noteDelta(at: now)
            reasoning += text
        case .content(let text):
            noteDelta(at: now)
            if answerStartedAt == nil { answerStartedAt = now }
            answer += text
        case .finish(let reason):
            finishReason = reason
        case .usage(let usage):
            self.usage = usage
        case .done:
            break
        }
    }

    /// The answer's Markdown blocks; only the part after the last finished block is parsed again.
    func answerBlocks() -> [MarkdownBlock] {
        markdown.blocks(for: answer)
    }

    var isThinking: Bool { !reasoning.isEmpty && answerStartedAt == nil }

    func thinkingSeconds(now: Date = Date()) -> Double? {
        guard !reasoning.isEmpty, let start = firstDeltaAt else { return nil }
        return (answerStartedAt ?? now).timeIntervalSince(start)
    }

    /// While streaming, an estimate from the deltas received so far.
    func liveTokensPerSecond(now: Date = Date()) -> Double? {
        guard let start = firstDeltaAt, deltaCount > 1 else { return nil }
        let elapsed = now.timeIntervalSince(start)
        return elapsed > 0 ? Double(deltaCount) / elapsed : nil
    }

    /// The finished message. Tokens per second is `completion_tokens` over the time from the first
    /// delta to the end of the stream; without usage (a stopped reply) the delta count stands in.
    func message(ending: ReplyStats.Ending, error: String?, endedAt: Date = Date()) -> ChatMessage {
        let tokens = usage?.completionTokens ?? deltaCount
        var tokensPerSecond: Double?
        if let start = firstDeltaAt, endedAt > start, tokens > 0 {
            tokensPerSecond = Double(tokens) / endedAt.timeIntervalSince(start)
        }
        let stats = tokens > 0
            ? ReplyStats(completionTokens: tokens, tokensPerSecond: tokensPerSecond,
                         thinkingSeconds: thinkingSeconds(now: endedAt), ending: ending)
            : nil
        return ChatMessage(
            role: .assistant,
            content: answer,
            reasoning: reasoning.isEmpty ? nil : reasoning,
            stats: stats,
            error: error
        )
    }

    private func noteDelta(at now: Date) {
        if firstDeltaAt == nil { firstDeltaAt = now }
        deltaCount += 1
    }
}
