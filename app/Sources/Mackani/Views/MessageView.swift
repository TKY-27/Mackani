import AppKit
import SwiftUI

/// A finished message from the conversation. Equatable, so a streaming reply below it does not make
/// it parse its Markdown again.
struct MessageView: View, Equatable {
    let message: ChatMessage

    var body: some View {
        switch message.role {
        case .user:
            UserMessageView(text: message.content)
        case .assistant:
            AssistantMessageView(
                blocks: MarkdownParser.parse(message.content),
                content: message.content,
                reasoning: message.reasoning,
                thinkingSeconds: message.stats?.thinkingSeconds,
                isThinking: false,
                isStreaming: false,
                footnote: message.stats.map(Self.footnote),
                error: message.error
            )
        }
    }

    static func footnote(for stats: ReplyStats) -> String {
        var parts = [tokenCount(stats.completionTokens)]
        if let rate = stats.tokensPerSecond { parts.append(tokensPerSecond(rate)) }
        switch stats.ending {
        case .finished: break
        case .lengthLimit: parts.append("Reached the length limit")
        case .stopped: parts.append("Stopped")
        }
        return parts.joined(separator: " · ")
    }

    static func tokenCount(_ count: Int) -> String {
        count == 1 ? "1 token" : "\(count.formatted()) tokens"
    }

    static func tokensPerSecond(_ rate: Double) -> String {
        "\(rate.formatted(.number.precision(.fractionLength(1)))) tokens/s"
    }
}

/// The reply that is arriving now. Only this view changes with each token.
struct StreamingMessageView: View {
    let reply: StreamingReply

    var body: some View {
        let estimate = reply.liveTokensPerSecond().map { "≈ " + MessageView.tokensPerSecond($0) }
        AssistantMessageView(
            blocks: reply.answerBlocks(),
            content: reply.answer,
            reasoning: reply.reasoning.isEmpty ? nil : reply.reasoning,
            thinkingSeconds: reply.thinkingSeconds(),
            isThinking: reply.isThinking,
            isStreaming: true,
            footnote: reply.deltaCount == 0
                ? nil
                : ([MessageView.tokenCount(reply.deltaCount)] + [estimate].compactMap { $0 }).joined(separator: " · "),
            error: nil
        )
    }
}

private struct UserMessageView: View {
    let text: String
    @ScaledMetric(relativeTo: .body) private var fontSize: CGFloat = 15

    var body: some View {
        HStack {
            Spacer(minLength: 80)
            Text(text)
                .font(.system(size: fontSize, design: .serif))
                .lineSpacing(fontSize * 0.25)
                .textSelection(.enabled)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(.quinary, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .contextMenu {
                    Button("Copy") { copyToPasteboard(text) }
                }
        }
    }
}

private struct AssistantMessageView: View {
    let blocks: [MarkdownBlock]
    let content: String
    let reasoning: String?
    let thinkingSeconds: Double?
    let isThinking: Bool
    let isStreaming: Bool
    let footnote: String?
    let error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let reasoning {
                ThinkingView(reasoning: reasoning, seconds: thinkingSeconds, isThinking: isThinking)
            }
            if !blocks.isEmpty {
                MarkdownView(blocks: blocks)
            }
            if let error {
                Label {
                    Text(error).textSelection(.enabled)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill").symbolRenderingMode(.multicolor)
                }
                .font(.callout)
            }
            HStack(spacing: 10) {
                if !isStreaming && !content.isEmpty {
                    CopyButton(text: content, label: "Copy message")
                }
                if let footnote {
                    Text(footnote)
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The model's reasoning: open while it thinks, folded away once the answer starts.
private struct ThinkingView: View {
    let reasoning: String
    let seconds: Double?
    let isThinking: Bool

    /// Nil until the reader toggles it; until then the section follows `isThinking`.
    @State private var userExpanded: Bool?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @ScaledMetric(relativeTo: .body) private var fontSize: CGFloat = 13.5

    private var isExpanded: Bool { userExpanded ?? isThinking }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                userExpanded = !isExpanded
            } label: {
                HStack(spacing: 5) {
                    Text(title)
                    Image(systemName: "chevron.right")
                        .font(.caption2.weight(.semibold))
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .font(.callout)
            .foregroundStyle(.secondary)
            .accessibilityLabel(title)
            .accessibilityValue(isExpanded ? "Expanded" : "Collapsed")
            .accessibilityHint("Shows or hides the model's reasoning")

            if isExpanded {
                Text(reasoning)
                    .font(.system(size: fontSize, design: .serif))
                    .lineSpacing(fontSize * 0.25)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding(.leading, 12)
                    .overlay(alignment: .leading) {
                        Rectangle().fill(.separator).frame(width: 2)
                    }
                    .transition(.opacity)
            }
        }
        .animation(reduceMotion ? nil : .easeOut(duration: 0.2), value: isExpanded)
    }

    private var title: String {
        if isThinking { return "Thinking…" }
        guard let seconds else { return "Thoughts" }
        if seconds < 1 { return "Thought for less than a second" }
        let whole = Int(seconds.rounded())
        return whole == 1 ? "Thought for 1 second" : "Thought for \(whole) seconds"
    }
}

struct CopyButton: View {
    let text: String
    let label: String
    @State private var copied = false

    var body: some View {
        Button {
            copyToPasteboard(text)
            copied = true
        } label: {
            Image(systemName: copied ? "checkmark" : "doc.on.doc")
                .frame(width: 14, height: 14)
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help(label)
        .accessibilityLabel(copied ? "Copied" : label)
        .task(id: copied) {
            guard copied else { return }
            try? await Task.sleep(for: .seconds(1.5))
            copied = false
        }
    }
}

@MainActor
func copyToPasteboard(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
}
