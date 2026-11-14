import AppKit
import SwiftUI

extension ServerState {
    /// A few words for the window subtitle and settings.
    var summary: String {
        switch self {
        case .noFolder: "No model folder"
        case .starting: "Loading the model…"
        case .ready: "Ready"
        case .failed: "Server stopped"
        }
    }
}

/// Explains plainly why the composer cannot send yet, and offers the way forward.
struct ServerStatusView: View {
    @Environment(ChatStore.self) private var chat

    var body: some View {
        switch chat.server.state {
        case .ready:
            EmptyView()
        case .noFolder:
            panel {
                HStack(spacing: 12) {
                    Text("Choose the folder that holds the model to start the server.")
                    Spacer(minLength: 0)
                    ChooseModelFolderButton()
                }
            }
        case .starting:
            panel {
                HStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text("Loading the model. This takes 10 to 40 seconds.")
                    Spacer(minLength: 0)
                }
                .accessibilityElement(children: .combine)
            }
        case .failed(let reason, let output):
            panel {
                VStack(alignment: .leading, spacing: 10) {
                    Label {
                        Text(reason)
                    } icon: {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .symbolRenderingMode(.multicolor)
                    }
                    if !output.isEmpty {
                        ScrollView {
                            Text(output.joined(separator: "\n"))
                                .font(.system(.callout, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .defaultScrollAnchor(.bottom)
                        .frame(maxHeight: 160)
                        .fixedSize(horizontal: false, vertical: true)
                    }
                    HStack {
                        Button("Try Again") { Task { await chat.restartServer() } }
                        ChooseModelFolderButton()
                    }
                }
            }
        }
    }

    private func panel(@ViewBuilder content: () -> some View) -> some View {
        content()
            .font(.callout)
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quinary, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

struct ChooseModelFolderButton: View {
    @Environment(ChatStore.self) private var chat

    var body: some View {
        Button("Choose Model Folder…") {
            Task {
                let panel = NSOpenPanel()
                panel.canChooseDirectories = true
                panel.canChooseFiles = false
                panel.allowsMultipleSelection = false
                panel.prompt = "Choose"
                panel.message = "Choose the folder that holds the Qwen3.8 27B model."
                panel.directoryURL = chat.server.modelFolder
                guard await panel.begin() == .OK, let folder = panel.url else { return }
                await chat.changeModelFolder(to: folder)
            }
        }
    }
}
