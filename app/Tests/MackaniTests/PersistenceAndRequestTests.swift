import Foundation
import Testing
@testable import Mackani

struct PersistenceTests {
    @Test func roundTripThroughDisk() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "mackani-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ConversationStore(directory: directory)

        var conversation = Conversation(customTitle: "Renamed")
        conversation.messages = [
            ChatMessage(role: .user, content: "Hello\nthere"),
            ChatMessage(
                role: .assistant, content: "Hi *you*", reasoning: "Greet back.",
                stats: ReplyStats(completionTokens: 12, tokensPerSecond: 21.5, thinkingSeconds: 0.4, ending: .stopped)
            ),
            ChatMessage(role: .assistant, content: "", error: "the prompt is too long"),
        ]
        let other = Conversation(messages: [ChatMessage(role: .user, content: "Second")])
        store.save(conversation)
        store.save(other)
        await store.flush()

        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path(percentEncoded: false))
        #expect(files.count == 2)
        let loaded = await store.loadAll().sorted { $0.messages.count > $1.messages.count }
        #expect(loaded == [conversation, other])

        store.delete(id: other.id)
        await store.flush()
        #expect(await store.loadAll() == [conversation])
    }

    @Test func unreadableFilesAreSkipped() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "mackani-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{".utf8).write(to: directory.appending(path: "broken.json"))
        #expect(await ConversationStore(directory: directory).loadAll().isEmpty)
    }

    @Test func titleComesFromFirstUserMessage() {
        var conversation = Conversation()
        #expect(conversation.title == "New Chat")
        conversation.messages = [ChatMessage(role: .user, content: "\n  How do I bake bread?  \nMore")]
        #expect(conversation.title == "How do I bake bread?")
        conversation.messages = [ChatMessage(role: .user, content: String(repeating: "a", count: 80))]
        #expect(conversation.title == String(repeating: "a", count: 60) + "…")
        conversation.customTitle = "Bread"
        #expect(conversation.title == "Bread")
    }
}

struct RequestEncodingTests {
    @Test func bodyMatchesTheServerContract() throws {
        let history = [
            ChatMessage(role: .user, content: "Hi"),
            ChatMessage(role: .assistant, content: "Hello", reasoning: "secret thoughts"),
            ChatMessage(role: .assistant, content: "", error: "failed"),
            ChatMessage(role: .user, content: "Again"),
        ]
        let request = ChatRequest(history: history, enableThinking: false)
        let json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? NSDictionary
        let expected: NSDictionary = [
            "model": "qwen3.8-27b",
            "stream": true,
            "stream_options": ["include_usage": true],
            "enable_thinking": false,
            "messages": [
                ["role": "user", "content": "Hi"],
                ["role": "assistant", "content": "Hello"],
                ["role": "user", "content": "Again"],
            ],
        ]
        #expect(json == expected)
    }

    @Test func urlRequestCarriesKeyAndNoOrigin() throws {
        let client = ChatClient(endpoint: ServerEndpoint(port: 49152, apiKey: "k"))
        let request = try client.urlRequest(for: ChatRequest(history: [], enableThinking: true))
        #expect(request.url?.absoluteString == "http://127.0.0.1:49152/v1/chat/completions")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer k")
        #expect(request.value(forHTTPHeaderField: "Origin") == nil)
    }
}
