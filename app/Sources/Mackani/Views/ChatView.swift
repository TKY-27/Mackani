import SwiftUI

/// The width of the message column and the composer, for comfortable reading.
let readingWidth: CGFloat = 760

struct ChatView: View {
    @Environment(ChatStore.self) private var chat
    let conversation: Conversation?
    @State private var draft = ""

    var body: some View {
        let reply = chat.reply.flatMap { $0.conversationID == conversation?.id ? $0 : nil }
        VStack(spacing: 0) {
            if let conversation, !conversation.messages.isEmpty || reply != nil {
                MessageList(conversation: conversation, reply: reply)
            } else {
                EmptyChatView()
            }
            VStack(spacing: 10) {
                ServerStatusView()
                ComposerView(
                    draft: $draft,
                    isReplying: reply != nil,
                    canSend: chat.canSend(draft),
                    focusID: conversation?.id,
                    send: send,
                    stop: chat.stop
                )
            }
            .frame(maxWidth: readingWidth)
            .padding(.horizontal, 24)
            .padding(.bottom, 16)
            .padding(.top, 8)
        }
        .navigationTitle(conversation?.title ?? "Mackani")
        .navigationSubtitle(chat.server.state.summary)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(action: chat.newConversation) {
                    Label("New Chat", systemImage: "square.and.pencil")
                }
                .help("New Chat (⌘N)")
            }
        }
    }

    private func send() {
        guard chat.canSend(draft) else { return }
        chat.send(draft)
        draft = ""
    }
}

private struct EmptyChatView: View {
    var body: some View {
        VStack(spacing: 6) {
            Text("Mackani")
                .font(.system(.largeTitle, design: .serif))
            Text("Qwen3.8 27B, running on this Mac")
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .combine)
    }
}

private struct MessageList: View {
    let conversation: Conversation
    let reply: StreamingReply?
    @State private var position = ScrollPosition(edge: .bottom)
    @State private var isAtBottom = true

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 28) {
                ForEach(conversation.messages) { message in
                    MessageView(message: message).equatable()
                }
                if let reply {
                    StreamingMessageView(reply: reply)
                }
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 28)
            .frame(maxWidth: readingWidth)
            .frame(maxWidth: .infinity)
        }
        .scrollPosition($position)
        .defaultScrollAnchor(.bottom)
        .onScrollGeometryChange(for: Bool.self) { geometry in
            geometry.contentOffset.y + geometry.containerSize.height >= geometry.contentSize.height - 60
        } action: { _, atBottom in
            isAtBottom = atBottom
        }
        // Follow a streaming reply only while the reader stays at the bottom.
        .onChange(of: reply?.deltaCount) {
            if isAtBottom { position.scrollTo(edge: .bottom) }
        }
        .onChange(of: conversation.messages.count) {
            position.scrollTo(edge: .bottom)
        }
        .onChange(of: conversation.id) {
            position.scrollTo(edge: .bottom)
        }
    }
}
