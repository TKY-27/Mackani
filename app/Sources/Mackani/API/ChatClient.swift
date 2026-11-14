import Foundation

/// Talks to the local server's OpenAI-compatible API.
struct ChatClient: Sendable {
    var endpoint: ServerEndpoint

    /// Prefill of a long prompt can take minutes before the first byte, so the idle timeout is long.
    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 3600
        configuration.timeoutIntervalForResource = 24 * 3600
        configuration.connectionProxyDictionary = [:]
        configuration.httpShouldSetCookies = false
        return URLSession(configuration: configuration)
    }()

    func urlRequest(for body: ChatRequest) throws -> URLRequest {
        var request = URLRequest(url: endpoint.baseURL.appending(path: "v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(endpoint.apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONEncoder().encode(body)
        return request
    }

    /// Streams the reply. Cancelling the consuming task cancels the HTTP request, which closes the
    /// connection and makes the server stop generating.
    func stream(_ body: ChatRequest) -> AsyncThrowingStream<StreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await Self.session.bytes(for: urlRequest(for: body))
                    let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                    guard status == 200 else {
                        throw ServerError(body: try await Self.collect(bytes), status: status)
                    }
                    for try await line in bytes.lines {
                        for event in try SSEParser.events(fromLine: line) {
                            if event == .done {
                                continuation.finish()
                                return
                            }
                            continuation.yield(event)
                        }
                    }
                    continuation.finish(throwing: ServerError(message: "The server closed the connection before the reply was complete."))
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Checks `GET /health`; true once the model is loaded.
    static func isHealthy(port: Int) async -> Bool {
        let url = ServerEndpoint(port: port, apiKey: "").baseURL.appending(path: "health")
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        guard let (_, response) = try? await session.data(for: request) else { return false }
        return (response as? HTTPURLResponse)?.statusCode == 200
    }

    private static func collect(_ bytes: URLSession.AsyncBytes) async throws -> Data {
        var data = Data()
        for try await byte in bytes where data.count < 65_536 {
            data.append(byte)
        }
        return data
    }
}
