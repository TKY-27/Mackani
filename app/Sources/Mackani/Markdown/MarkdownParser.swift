import Foundation

/// A top-level piece of a Markdown message. Inline syntax (emphasis, code spans, links) stays in the
/// strings and is rendered by `AttributedString(markdown:)`.
enum MarkdownBlock: Equatable, Sendable {
    case paragraph(String)
    case heading(level: Int, text: String)
    case list([ListItem])
    case quote([MarkdownBlock])
    /// `isClosed` is false while a reply is still streaming inside the fence.
    case code(language: String?, code: String, isClosed: Bool)
    case table(header: [String], rows: [[String]])
    case rule
}

struct ListItem: Equatable, Sendable {
    /// Nesting depth, from the item's indentation.
    var depth: Int
    /// "•" for bullets, or the source number with its punctuation, such as "3.".
    var marker: String
    var text: String
}

/// A small line-based Markdown block parser covering what chat replies use.
///
/// A blank line outside a code fence always ends the current block, and nothing before it can
/// change what follows. `IncrementalMarkdown` relies on that to parse only the growing tail.
enum MarkdownParser {
    static func parse(_ text: String) -> [MarkdownBlock] {
        parse(text[...])
    }

    static func parse(_ text: Substring) -> [MarkdownBlock] {
        let lines = text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
        var parser = State(lines: lines.map(String.init))
        return parser.run()
    }

    /// The offset just past the last blank line that lies outside a code fence and is followed by a
    /// complete line, or nil. Text before this offset parses the same however the text continues.
    static func lastStableBoundary(in text: Substring) -> String.Index? {
        var fence: Fence?
        var boundary: String.Index?
        var lineStart = text.startIndex
        while let newline = text[lineStart...].firstIndex(where: \.isNewline) {
            let line = text[lineStart..<newline]
            let next = text.index(after: newline)
            if let open = fence {
                if open.isClosed(by: line) { fence = nil }
            } else if let opening = Fence(opening: line) {
                fence = opening
            } else if line.allSatisfy(\.isWhitespace) {
                boundary = next
            }
            lineStart = next
        }
        return boundary
    }

    private struct State {
        let lines: [String]
        var index = 0
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []
        var items: [ListItem] = []

        init(lines: [String]) {
            self.lines = lines
        }

        mutating func run() -> [MarkdownBlock] {
            while index < lines.count {
                let line = lines[index]
                if line.allSatisfy(\.isWhitespace) {
                    flush()
                    index += 1
                } else if let fence = Fence(opening: line[...]) {
                    flush()
                    readCode(fence)
                } else if let heading = Self.heading(line) {
                    flush()
                    blocks.append(heading)
                    index += 1
                } else if Self.isRule(line) {
                    flush()
                    blocks.append(.rule)
                    index += 1
                } else if Self.quoteContent(line) != nil {
                    flush()
                    readQuote()
                } else if let item = Self.listItem(line) {
                    if !paragraph.isEmpty { flush() }
                    items.append(item)
                    index += 1
                } else if index + 1 < lines.count, line.contains("|"), Self.isTableSeparator(lines[index + 1]) {
                    flush()
                    readTable()
                } else if !items.isEmpty {
                    // A continuation line of the last list item.
                    items[items.count - 1].text += "\n" + line.trimmingCharacters(in: .whitespaces)
                    index += 1
                } else {
                    paragraph.append(line)
                    index += 1
                }
            }
            flush()
            return blocks
        }

        mutating func flush() {
            if !paragraph.isEmpty {
                blocks.append(.paragraph(paragraph.joined(separator: "\n")))
                paragraph.removeAll()
            }
            if !items.isEmpty {
                blocks.append(.list(items))
                items.removeAll()
            }
        }

        mutating func readCode(_ fence: Fence) {
            index += 1
            var code: [String] = []
            while index < lines.count {
                let line = lines[index]
                index += 1
                if fence.isClosed(by: line[...]) {
                    blocks.append(.code(language: fence.language, code: code.joined(separator: "\n"), isClosed: true))
                    return
                }
                code.append(fence.removingIndent(from: line))
            }
            blocks.append(.code(language: fence.language, code: code.joined(separator: "\n"), isClosed: false))
        }

        mutating func readQuote() {
            var inner: [String] = []
            while index < lines.count, let content = Self.quoteContent(lines[index]) {
                inner.append(content)
                index += 1
            }
            blocks.append(.quote(MarkdownParser.parse(inner.joined(separator: "\n"))))
        }

        mutating func readTable() {
            let header = Self.cells(lines[index])
            index += 2
            var rows: [[String]] = []
            while index < lines.count, lines[index].contains("|"), !lines[index].allSatisfy(\.isWhitespace) {
                var row = Self.cells(lines[index])
                // Pad or trim to the header's width so the grid stays rectangular.
                row = Array((row + Array(repeating: "", count: max(0, header.count - row.count))).prefix(header.count))
                rows.append(row)
                index += 1
            }
            blocks.append(.table(header: header, rows: rows))
        }

        static func heading(_ line: String) -> MarkdownBlock? {
            let trimmed = line.drop { $0 == " " }
            guard line.count - trimmed.count <= 3 else { return nil }
            let hashes = trimmed.prefix { $0 == "#" }
            guard (1...6).contains(hashes.count) else { return nil }
            let rest = trimmed.dropFirst(hashes.count)
            guard rest.isEmpty || rest.first == " " || rest.first == "\t" else { return nil }
            var text = rest.trimmingCharacters(in: .whitespaces)
            // Optional closing hashes: "## Title ##".
            while text.hasSuffix("#") { text.removeLast() }
            return .heading(level: hashes.count, text: text.trimmingCharacters(in: .whitespaces))
        }

        static func isRule(_ line: String) -> Bool {
            let compact = line.filter { $0 != " " && $0 != "\t" }
            guard compact.count >= 3, let first = compact.first, "-*_".contains(first) else { return false }
            return compact.allSatisfy { $0 == first }
        }

        static func quoteContent(_ line: String) -> String? {
            let trimmed = line.drop { $0 == " " }
            guard line.count - trimmed.count <= 3, trimmed.first == ">" else { return nil }
            var content = trimmed.dropFirst()
            if content.first == " " { content = content.dropFirst() }
            return String(content)
        }

        static func listItem(_ line: String) -> ListItem? {
            let expanded = line.replacingOccurrences(of: "\t", with: "    ")
            let body = expanded.drop { $0 == " " }
            let indent = expanded.count - body.count
            let marker: String
            let rest: Substring
            if let first = body.first, "-*+".contains(first) {
                marker = "•"
                rest = body.dropFirst()
            } else {
                let digits = body.prefix { $0.isASCII && $0.isNumber }
                let afterDigits = body.dropFirst(digits.count)
                guard (1...9).contains(digits.count), let punctuation = afterDigits.first, ".)".contains(punctuation) else {
                    return nil
                }
                marker = digits + "."
                rest = afterDigits.dropFirst()
            }
            guard rest.isEmpty || rest.first == " " else { return nil }
            return ListItem(depth: indent / 2, marker: marker, text: rest.trimmingCharacters(in: .whitespaces))
        }

        static func isTableSeparator(_ line: String) -> Bool {
            let cells = cells(line)
            return !cells.isEmpty && cells.allSatisfy { cell in
                let core = cell.trimmingCharacters(in: CharacterSet(charactersIn: ":"))
                return !core.isEmpty && core.allSatisfy { $0 == "-" }
            }
        }

        static func cells(_ line: String) -> [String] {
            var trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("|") { trimmed.removeFirst() }
            if trimmed.hasSuffix("|") { trimmed.removeLast() }
            return trimmed.split(separator: "|", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces) }
        }
    }
}

/// An opening code fence: three or more backticks or tildes, with an optional language.
private struct Fence {
    var character: Character
    var length: Int
    var indent: Int
    var language: String?

    init?(opening line: Substring) {
        let body = line.drop { $0 == " " }
        indent = line.count - body.count
        guard indent <= 3, let first = body.first, first == "`" || first == "~" else { return nil }
        let run = body.prefix { $0 == first }
        guard run.count >= 3 else { return nil }
        let info = body.dropFirst(run.count).trimmingCharacters(in: .whitespaces)
        if first == "`" && info.contains("`") { return nil }
        character = first
        length = run.count
        language = info.split(separator: " ").first.map(String.init)
    }

    func isClosed(by line: Substring) -> Bool {
        let body = line.drop { $0 == " " }
        guard line.count - body.count <= 3 else { return false }
        let run = body.prefix { $0 == character }
        return run.count >= length && body.dropFirst(run.count).allSatisfy(\.isWhitespace)
    }

    func removingIndent(from line: String) -> String {
        let spaces = line.prefix { $0 == " " }.count
        return String(line.dropFirst(min(spaces, indent)))
    }
}

/// Parses a growing reply without re-reading the part that can no longer change: blocks that end
/// before a blank line (outside a code fence) are kept, and only the tail is parsed again.
struct IncrementalMarkdown {
    private var stableBlocks: [MarkdownBlock] = []
    private var stableLength = 0

    mutating func blocks(for text: String) -> [MarkdownBlock] {
        if text.utf8.count < stableLength {
            self = IncrementalMarkdown()
        }
        var tail = Substring(text).dropFirstUTF8(stableLength)
        if let boundary = MarkdownParser.lastStableBoundary(in: tail) {
            stableBlocks += MarkdownParser.parse(tail[..<boundary])
            stableLength += tail.utf8.distance(from: tail.startIndex, to: boundary)
            tail = tail[boundary...]
        }
        return stableBlocks + MarkdownParser.parse(tail)
    }
}

private extension Substring {
    func dropFirstUTF8(_ count: Int) -> Substring {
        self[utf8.index(startIndex, offsetBy: count)...]
    }
}
