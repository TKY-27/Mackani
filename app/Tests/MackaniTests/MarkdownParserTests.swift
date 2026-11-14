import Testing
@testable import Mackani

struct MarkdownParserTests {
    @Test func paragraphsHeadingsAndRules() {
        let blocks = MarkdownParser.parse("# Title\n\nFirst line\nsecond line\n\n---\n\n### Small ###")
        #expect(blocks == [
            .heading(level: 1, text: "Title"),
            .paragraph("First line\nsecond line"),
            .rule,
            .heading(level: 3, text: "Small"),
        ])
    }

    @Test func hashWithoutSpaceIsNotAHeading() {
        #expect(MarkdownParser.parse("#hashtag") == [.paragraph("#hashtag")])
    }

    @Test func listsKeepMarkersDepthAndContinuations() {
        let blocks = MarkdownParser.parse("Intro\n- one\n  - nested\n- two\n  more\n\n3. three\n4) four")
        #expect(blocks == [
            .paragraph("Intro"),
            .list([
                ListItem(depth: 0, marker: "•", text: "one"),
                ListItem(depth: 1, marker: "•", text: "nested"),
                ListItem(depth: 0, marker: "•", text: "two\nmore"),
            ]),
            .list([
                ListItem(depth: 0, marker: "3.", text: "three"),
                ListItem(depth: 0, marker: "4.", text: "four"),
            ]),
        ])
    }

    @Test func blockQuoteParsesItsContent() {
        #expect(MarkdownParser.parse("> # Note\n> said *this*\n\nafter") == [
            .quote([.heading(level: 1, text: "Note"), .paragraph("said *this*")]),
            .paragraph("after"),
        ])
    }

    @Test func closedCodeFenceWithLanguageKeepsBlankLinesAndMarkdown() {
        let source = "Before\n```swift\nlet a = 1\n\n# not a heading\n```\nAfter"
        #expect(MarkdownParser.parse(source) == [
            .paragraph("Before"),
            .code(language: "swift", code: "let a = 1\n\n# not a heading", isClosed: true),
            .paragraph("After"),
        ])
    }

    @Test func codeFenceStillOpenWhileStreaming() {
        #expect(MarkdownParser.parse("Here:\n```python\nprint(1)\n") == [
            .paragraph("Here:"),
            .code(language: "python", code: "print(1)\n", isClosed: false),
        ])
        #expect(MarkdownParser.parse("```") == [.code(language: nil, code: "", isClosed: false)])
    }

    @Test func shorterOrDifferentFenceDoesNotClose() {
        #expect(MarkdownParser.parse("````\n```\n~~~\n````") == [
            .code(language: nil, code: "```\n~~~", isClosed: true),
        ])
    }

    @Test func tables() {
        let source = "| Name | Size |\n|:---|---:|\n| a | 1 |\n| b |\n\nafter"
        #expect(MarkdownParser.parse(source) == [
            .table(header: ["Name", "Size"], rows: [["a", "1"], ["b", ""]]),
            .paragraph("after"),
        ])
    }

    @Test func crlfLineEndings() {
        #expect(MarkdownParser.parse("a\r\n\r\nb") == [.paragraph("a"), .paragraph("b")])
    }

    /// Feeding a reply one character at a time must always match a full parse of the same text.
    @Test func incrementalParsingMatchesFullParseAtEveryPrefix() {
        let reply = """
        # Plan

        Some *text* with `code`.

        - one
        - two

        ```rust
        fn main() {

            println!("hi");
        }
        ```

        > quoted

        | a | b |
        |---|---|
        | 1 | 2 |

        Done.
        """
        var incremental = IncrementalMarkdown()
        var prefix = ""
        for character in reply {
            prefix.append(character)
            #expect(incremental.blocks(for: prefix) == MarkdownParser.parse(prefix), "prefix: \(prefix.debugDescription)")
        }
        // A shorter text (a different reply) starts over.
        #expect(incremental.blocks(for: "x") == [.paragraph("x")])
    }
}
