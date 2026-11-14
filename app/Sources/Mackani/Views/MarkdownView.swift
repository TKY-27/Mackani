import SwiftUI

/// Renders parsed Markdown blocks: New York for prose, SF Mono for code.
struct MarkdownView: View {
    let blocks: [MarkdownBlock]
    @ScaledMetric(relativeTo: .body) private var fontSize: CGFloat = 15

    var body: some View {
        VStack(alignment: .leading, spacing: fontSize * 0.9) {
            // Blocks only append while streaming, so their position is a stable identity.
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                BlockView(block: block, fontSize: fontSize)
            }
        }
        .textSelection(.enabled)
    }
}

private struct BlockView: View {
    let block: MarkdownBlock
    let fontSize: CGFloat

    var body: some View {
        switch block {
        case .paragraph(let text):
            prose(text)
        case .heading(let level, let text):
            Text(InlineMarkdown.attributed(text, fontSize: fontSize))
                .font(.system(size: fontSize * Self.headingScale(level), weight: .semibold, design: .serif))
                .padding(.top, level <= 2 ? fontSize * 0.4 : 0)
                .accessibilityAddTraits(.isHeader)
        case .list(let items):
            VStack(alignment: .leading, spacing: fontSize * 0.35) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(item.marker)
                            .font(.system(size: fontSize, design: .serif))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                            .frame(minWidth: fontSize, alignment: .trailing)
                        prose(item.text)
                    }
                    .padding(.leading, CGFloat(item.depth) * fontSize * 1.4)
                }
            }
        case .quote(let inner):
            HStack(alignment: .top, spacing: 12) {
                Rectangle().fill(.separator).frame(width: 3)
                // Quotes nest blocks, so the recursive view needs a type-erased wrapper.
                AnyView(MarkdownView(blocks: inner))
                    .foregroundStyle(.secondary)
            }
            .fixedSize(horizontal: false, vertical: true)
        case .code(let language, let code, _):
            CodeBlockView(language: language, code: code, fontSize: fontSize)
        case .table(let header, let rows):
            TableBlockView(header: header, rows: rows, fontSize: fontSize)
        case .rule:
            Divider().padding(.vertical, fontSize * 0.3)
        }
    }

    private func prose(_ text: String) -> some View {
        Text(InlineMarkdown.attributed(text, fontSize: fontSize))
            .font(.system(size: fontSize, design: .serif))
            .lineSpacing(fontSize * 0.3)
            .fixedSize(horizontal: false, vertical: true)
    }

    private static func headingScale(_ level: Int) -> CGFloat {
        switch level {
        case 1: 1.45
        case 2: 1.25
        case 3: 1.1
        default: 1.0
        }
    }
}

private struct CodeBlockView: View {
    let language: String?
    let code: String
    let fontSize: CGFloat

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language ?? "code")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                CopyButton(text: code, label: "Copy code")
            }
            .padding(.leading, 12)
            .padding(.trailing, 8)
            .padding(.vertical, 6)
            Divider()
            ScrollView(.horizontal) {
                Text(code)
                    .font(.system(size: fontSize * 0.85, design: .monospaced))
                    .lineSpacing(fontSize * 0.15)
                    .fixedSize(horizontal: true, vertical: false)
                    .padding(12)
            }
        }
        .background(.quinary, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(.separator))
        .accessibilityElement(children: .contain)
        .accessibilityLabel(language.map { "\($0) code" } ?? "Code")
    }
}

private struct TableBlockView: View {
    let header: [String]
    let rows: [[String]]
    let fontSize: CGFloat

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 6) {
            GridRow {
                ForEach(Array(header.enumerated()), id: \.offset) { _, cell in
                    Text(InlineMarkdown.attributed(cell, fontSize: fontSize)).fontWeight(.semibold)
                }
            }
            Divider()
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                GridRow {
                    ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                        Text(InlineMarkdown.attributed(cell, fontSize: fontSize))
                    }
                }
            }
        }
        .font(.system(size: fontSize * 0.95, design: .serif))
    }
}

/// Emphasis, code spans and links inside a block, via Foundation's Markdown support.
enum InlineMarkdown {
    static func attributed(_ source: String, fontSize: CGFloat) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible
        )
        guard var text = try? AttributedString(markdown: source, options: options) else {
            return AttributedString(source)
        }
        for run in text.runs where run.inlinePresentationIntent?.contains(.code) == true {
            text[run.range].font = .system(size: fontSize * 0.88, design: .monospaced)
            text[run.range].backgroundColor = Color(nsColor: .quaternarySystemFill)
        }
        return text
    }
}
