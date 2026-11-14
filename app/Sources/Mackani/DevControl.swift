#if DEBUG
import AppKit

/// Debug builds only: with `MACKANI_DEV_CONTROL=1` at launch, the app takes simple
/// commands from distributed notifications named `io.github.tky-27.mackani.dev` (the command is the
/// notification's object), so the running app can be exercised without accessibility access.
/// Without the variable nothing is registered.
@MainActor
enum DevControl {
    static func installIfRequested(chat: ChatStore) {
        guard ProcessInfo.processInfo.environment["MACKANI_DEV_CONTROL"] == "1" else { return }
        DistributedNotificationCenter.default().addObserver(
            forName: .init("io.github.tky-27.mackani.dev"), object: nil, queue: .main
        ) { notification in
            guard let command = notification.object as? String else { return }
            MainActor.assumeIsolated { run(command, chat: chat) }
        }
    }

    private static func run(_ command: String, chat: ChatStore) {
        let (verb, argument) = command.firstIndex(of: " ").map {
            (String(command[..<$0]), String(command[command.index(after: $0)...]))
        } ?? (command, "")
        switch verb {
        case "send": chat.send(argument.replacingOccurrences(of: "\\n", with: "\n"))
        case "stop": chat.stop()
        case "new": chat.newConversation()
        case "select": if let index = Int(argument), chat.conversations.indices.contains(index) { chat.select(chat.conversations[index].id) }
        case "appearance": NSApp.appearance = argument == "dark" ? NSAppearance(named: .darkAqua) : argument == "light" ? NSAppearance(named: .aqua) : nil
        case "thinking": UserDefaults.standard.set(argument == "on", forKey: Preferences.thinkingKey)
        case "rename": if let id = chat.selectedID { chat.rename(id, to: argument) }
        case "delete": if let id = chat.selectedID { chat.delete(id) }
        case "settings": NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
        case "quit": NSApp.terminate(nil)
        case "restart": Task { await chat.restartServer() }
        case "insert": composer?.insertText(argument, replacementRange: NSRange(location: NSNotFound, length: 0))
        case "key": press(argument)
        default: break
        }
    }

    private static var composer: NSTextView? {
        NSApp.windows.lazy.compactMap { $0.firstResponder as? NSTextView }.first { $0.isEditable }
    }

    /// "return", "shift-return", "cmd-n", "cmd-.": Return goes to the composer, Command keys to the menu.
    private static func press(_ key: String) {
        let parts = key.split(separator: "-").map(String.init)
        guard let name = parts.last else { return }
        var flags: NSEvent.ModifierFlags = []
        if parts.contains("shift") { flags.insert(.shift) }
        if parts.contains("cmd") { flags.insert(.command) }
        let characters = name == "return" ? "\r" : name
        guard let event = NSEvent.keyEvent(
            with: .keyDown, location: .zero, modifierFlags: flags, timestamp: ProcessInfo.processInfo.systemUptime,
            windowNumber: composer?.window?.windowNumber ?? 0, context: nil, characters: characters,
            charactersIgnoringModifiers: characters, isARepeat: false, keyCode: name == "return" ? 36 : 0
        ) else { return }
        if flags.contains(.command) {
            _ = NSApp.mainMenu?.performKeyEquivalent(with: event)
        } else {
            composer?.keyDown(with: event)
        }
    }
}
#endif
