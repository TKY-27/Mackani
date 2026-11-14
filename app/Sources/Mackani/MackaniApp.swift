import AppKit
import SwiftUI

@main
struct MackaniApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Window("Mackani", id: "main") {
            ContentView()
                .environment(appDelegate.chat)
                .frame(minWidth: 680, minHeight: 440)
                // Replies are model output: only web links open, whatever their text says.
                .environment(\.openURL, OpenURLAction { url in
                    ["http", "https"].contains(url.scheme?.lowercased() ?? "") ? .systemAction : .discarded
                })
        }
        .defaultSize(width: 1040, height: 760)
        .commands { ChatCommands(chat: appDelegate.chat) }

        Settings {
            SettingsView()
                .environment(appDelegate.chat)
        }
        .windowResizability(.contentSize)
    }
}

struct ChatCommands: Commands {
    let chat: ChatStore

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Chat") { chat.newConversation() }
                .keyboardShortcut("n")
            Button("Stop Reply") { chat.stop() }
                .keyboardShortcut(".")
                .disabled(!chat.isStreaming)
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let chat = ChatStore(server: ServerController())

    func applicationDidFinishLaunching(_ notification: Notification) {
        chat.server.start()
        Task { await chat.load() }
        #if DEBUG
        DevControl.installIfRequested(chat: chat)
        #endif
    }

    /// The server holds about 19 GB, so closing the window quits rather than leaving it loaded.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    /// Waits for the server to exit before quitting, so a relaunch never finds two servers.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let waitForShutdown = chat.prepareToQuit()
        DispatchQueue.global(qos: .userInitiated).async {
            waitForShutdown()
            // Quit may have been requested from inside a main-queue block, which keeps the main queue
            // busy; the run loop still runs blocks while AppKit waits for this reply.
            CFRunLoopPerformBlock(CFRunLoopGetMain(), CFRunLoopMode.commonModes.rawValue) {
                MainActor.assumeIsolated { NSApp.reply(toApplicationShouldTerminate: true) }
            }
            CFRunLoopWakeUp(CFRunLoopGetMain())
        }
        return .terminateLater
    }
}
