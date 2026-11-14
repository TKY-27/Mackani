import Foundation
import Testing
@testable import Mackani

struct SSEParserTests {
    @Test func reasoningAndContentDeltas() throws {
        let reasoning = #"data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"Hm"},"finish_reason":null}]}"#
        let content = #"data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#
        #expect(try SSEParser.events(fromLine: reasoning) == [.reasoning("Hm")])
        #expect(try SSEParser.events(fromLine: content) == [.content("Hi")])
    }

    @Test func roleOnlyDeltaAndNonDataLinesGiveNothing() throws {
        #expect(try SSEParser.events(fromLine: #"data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#).isEmpty)
        #expect(try SSEParser.events(fromLine: "").isEmpty)
        #expect(try SSEParser.events(fromLine: ": keep-alive").isEmpty)
        #expect(try SSEParser.events(fromLine: "event: message").isEmpty)
    }

    @Test func finishUsageAndDone() throws {
        let finish = #"data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#
        #expect(try SSEParser.events(fromLine: finish) == [.finish(reason: "length")])

        let usage = #"data: {"choices":[],"usage":{"prompt_tokens":32,"completion_tokens":77,"total_tokens":109,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":15}}}"#
        let expected = Usage(
            promptTokens: 32, completionTokens: 77, totalTokens: 109,
            completionTokensDetails: .init(reasoningTokens: 15),
            promptTokensDetails: .init(cachedTokens: 4)
        )
        #expect(try SSEParser.events(fromLine: usage) == [.usage(expected)])
        #expect(try SSEParser.events(fromLine: "data: [DONE]") == [.done])
        #expect(try SSEParser.events(fromLine: "data:[DONE]") == [.done])
    }

    @Test func errorEventThrowsServerMessage() {
        let line = #"data: {"error":{"message":"the prompt is too long","code":"context_length_exceeded"}}"#
        #expect(throws: ServerError(message: "the prompt is too long", code: "context_length_exceeded")) {
            try SSEParser.events(fromLine: line)
        }
    }

    @Test func malformedJSONThrows() {
        #expect(throws: DecodingError.self) { try SSEParser.events(fromLine: "data: {not json") }
    }

    @Test func errorBodiesWithStringOrNumericCodes() {
        let text = ServerError(body: Data(#"{"error":{"message":"too long","code":"context_length_exceeded"}}"#.utf8), status: 400)
        #expect(text == ServerError(message: "too long", code: "context_length_exceeded"))
        let number = ServerError(body: Data(#"{"error":{"message":"invalid or missing API key","type":"authentication_error","code":401}}"#.utf8), status: 401)
        #expect(number == ServerError(message: "invalid or missing API key", code: "401"))
        let plain = ServerError(body: Data("busy".utf8), status: 503)
        #expect(plain.message == "busy")
    }
}
