import Foundation
import Observation

enum Preferences {
    static let thinkingKey = "thinkingEnabled"

    static var isThinkingEnabled: Bool {
        UserDefaults.standard.object(forKey: thinkingKey) as? Bool ?? true
    }
}

/// The conversations, the selection and the one reply that may be streaming.
@MainActor
@Observable
final class ChatStore {
    private(set) var conversations: [Conversation] = []
    var selectedID: UUID?
    private(set) var reply: StreamingReply?

    let server: ServerController
    @ObservationIgnored private let store: ConversationStore
    @ObservationIgnored private var replyTask: Task<Void, Never>?

    init(server: ServerController, store: ConversationStore = ConversationStore()) {
        self.server = server
        self.store = store
    }

    func load() async {
        let loaded = await store.loadAll()
        conversations = loaded.sorted { $0.updatedAt > $1.updatedAt }
        if selectedID == nil { selectedID = conversations.first?.id }
    }

    var selected: Conversation? {
        conversations.first { $0.id == selectedID }
    }

    var isStreaming: Bool { reply != nil }

    func canSend(_ text: String) -> Bool {
        server.endpoint != nil && reply == nil && !text.allSatisfy(\.isWhitespace)
    }

    // MARK: Conversations

    func newConversation() {
        if let selected, selected.messages.isEmpty { return }
        discardEmptyConversations()
        let conversation = Conversation()
        conversations.insert(conversation, at: 0)
        selectedID = conversation.id
    }

    func select(_ id: UUID?) {
        guard id != selectedID else { return }
        selectedID = id
        discardEmptyConversations()
    }

    func rename(_ id: UUID, to title: String) {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        update(id, isActivity: false) { $0.customTitle = trimmed.isEmpty ? nil : trimmed }
    }

    func delete(_ id: UUID) {
        if reply?.conversationID == id { stop() }
        conversations.removeAll { $0.id == id }
        store.delete(id: id)
        if selectedID == id { selectedID = conversations.first?.id }
    }

    /// Empty conversations exist only while selected; they are never written to disk.
    private func discardEmptyConversations() {
        conversations.removeAll { $0.messages.isEmpty && $0.id != selectedID }
    }

    // MARK: Replies

    func send(_ text: String) {
        guard canSend(text), let endpoint = server.endpoint else { return }
        if selected == nil { newConversation() }
        guard let id = selectedID else { return }

        update(id) { $0.messages.append(ChatMessage(role: .user, content: text)) }
        guard let history = selected?.messages else { return }

        let reply = StreamingReply(conversationID: id)
        self.reply = reply
        let request = ChatRequest(history: history, enableThinking: Preferences.isThinkingEnabled)
        replyTask = Task { [weak self] in
            var failure: String?
            do {
                for try await event in ChatClient(endpoint: endpoint).stream(request) {
                    reply.apply(event)
                }
            } catch is CancellationError {
            } catch let error as URLError where error.code == .cancelled {
            } catch {
                failure = error.localizedDescription
            }
            self?.finish(reply, error: failure, stopped: Task.isCancelled)
        }
    }

    /// Stops the streaming reply; closing the connection makes the server stop generating.
    func stop() {
        replyTask?.cancel()
    }

    private func finish(_ reply: StreamingReply, error: String?, stopped: Bool) {
        guard self.reply === reply else { return }
        self.reply = nil
        replyTask = nil
        let ending: ReplyStats.Ending = stopped ? .stopped : reply.finishReason == "length" ? .lengthLimit : .finished
        let message = reply.message(ending: ending, error: error)
        if message.content.isEmpty, message.reasoning == nil, error == nil { return }
        update(reply.conversationID) { $0.messages.append(message) }
    }

    /// Changes the model folder: the reply in progress is stopped first, then the server restarts.
    func changeModelFolder(to folder: URL) async {
        stop()
        await server.setModelFolder(folder)
    }

    func restartServer() async {
        stop()
        await server.restart()
    }

    /// Starts shutting down for quitting and returns the blocking wait for it to finish: pending saves
    /// written, and the server exited (killed if it ignores SIGTERM for 10 seconds). Run the wait off
    /// the main thread.
    func prepareToQuit() -> @Sendable () -> Void {
        stop()
        // Keep a reply in progress now: the main actor may not run again before the app exits.
        if let reply { finish(reply, error: nil, stopped: true) }
        let running = server.terminateForQuit()
        let store = self.store
        return {
            store.waitForPendingWrites()
            guard let running else { return }
            if running.exited.wait(timeout: .now() + 10) == .timedOut {
                running.kill()
                _ = running.exited.wait(timeout: .now() + 2)
            }
        }
    }

    /// Applies a change and saves it. New messages are activity: they move the conversation to the
    /// top of the list, which is ordered most recent first. A rename is not.
    private func update(_ id: UUID, isActivity: Bool = true, _ change: (inout Conversation) -> Void) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        change(&conversations[index])
        if isActivity {
            conversations[index].updatedAt = Date()
            conversations.insert(conversations.remove(at: index), at: 0)
        }
        if let saved = conversations.first(where: { $0.id == id }), !saved.messages.isEmpty {
            store.save(saved)
        }
    }
}
