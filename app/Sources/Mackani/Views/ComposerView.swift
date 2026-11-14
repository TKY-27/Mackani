import AppKit
import SwiftUI

struct ComposerView: View {
    @Binding var draft: String
    let isReplying: Bool
    let canSend: Bool
    /// Focus moves to the composer whenever this changes (another conversation was selected).
    let focusID: UUID?
    let send: () -> Void
    let stop: () -> Void

    @ScaledMetric(relativeTo: .body) private var fontSize: CGFloat = 14

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            // The hidden text sizes the field: it grows with the draft up to ten lines, then scrolls.
            Text(draft + " ")
                .font(.system(size: fontSize))
                .lineLimit(1...10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .hidden()
                .overlay {
                    ComposerTextView(text: $draft, fontSize: fontSize, focusID: focusID, onSubmit: send)
                }
                .overlay(alignment: .topLeading) {
                    if draft.isEmpty {
                        Text("Message Mackani")
                            .font(.system(size: fontSize))
                            .foregroundStyle(.tertiary)
                            .allowsHitTesting(false)
                            .accessibilityHidden(true)
                    }
                }
                .padding(.vertical, 4)

            if isReplying {
                Button(action: stop) {
                    Image(systemName: "stop.fill")
                        .frame(width: 16, height: 16)
                }
                .buttonStyle(.bordered)
                .buttonBorderShape(.circle)
                .controlSize(.large)
                .help("Stop (⌘.)")
                .accessibilityLabel("Stop reply")
            } else {
                Button(action: send) {
                    Image(systemName: "arrow.up")
                        .fontWeight(.semibold)
                        .frame(width: 16, height: 16)
                }
                .buttonStyle(.borderedProminent)
                .buttonBorderShape(.circle)
                .controlSize(.large)
                .disabled(!canSend)
                .help("Send (Return)")
                .accessibilityLabel("Send message")
            }
        }
        .padding(.leading, 16)
        .padding(.trailing, 8)
        .padding(.vertical, 8)
        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).strokeBorder(.separator))
    }
}

/// A plain-text editor where Return sends and Shift-Return inserts a line break. Return that
/// confirms an input-method composition (Japanese, Chinese, …) is left to the input method.
private struct ComposerTextView: NSViewRepresentable {
    @Binding var text: String
    let fontSize: CGFloat
    let focusID: UUID?
    let onSubmit: () -> Void

    func makeNSView(context: Context) -> NSScrollView {
        let textView = SubmittingTextView()
        textView.delegate = context.coordinator
        textView.isRichText = false
        textView.allowsUndo = true
        textView.drawsBackground = false
        textView.textColor = .labelColor
        textView.textContainerInset = .zero
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainer?.widthTracksTextView = true
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: .greatestFiniteMagnitude)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.setAccessibilityLabel("Message")

        let scrollView = NSScrollView()
        scrollView.documentView = textView
        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? SubmittingTextView else { return }
        context.coordinator.parent = self
        textView.onSubmit = onSubmit
        if textView.string != text { textView.string = text }
        if textView.font?.pointSize != fontSize { textView.font = .systemFont(ofSize: fontSize) }
        if context.coordinator.focusedID != focusID || !context.coordinator.hasFocused {
            context.coordinator.focusedID = focusID
            context.coordinator.hasFocused = true
            DispatchQueue.main.async { textView.window?.makeFirstResponder(textView) }
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: ComposerTextView
        var focusedID: UUID?
        var hasFocused = false

        init(parent: ComposerTextView) { self.parent = parent }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            parent.text = textView.string
        }
    }
}

private final class SubmittingTextView: NSTextView {
    var onSubmit: (() -> Void)?

    override func keyDown(with event: NSEvent) {
        let isReturn = event.keyCode == 36 || event.keyCode == 76 // Return, keypad Enter
        let modifiers = event.modifierFlags.intersection([.shift, .option, .command, .control])
        if isReturn, modifiers.isEmpty, !hasMarkedText() {
            onSubmit?()
            return
        }
        super.keyDown(with: event)
    }
}
