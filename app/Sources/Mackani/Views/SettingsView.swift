import SwiftUI

struct SettingsView: View {
    @Environment(ChatStore.self) private var chat
    @AppStorage(Preferences.thinkingKey) private var isThinkingEnabled = true

    var body: some View {
        Form {
            Section {
                LabeledContent("Model folder") {
                    Text(chat.server.modelFolder?.path(percentEncoded: false) ?? "Not chosen")
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                        .help(chat.server.modelFolder?.path(percentEncoded: false) ?? "")
                }
                LabeledContent("Server", value: chat.server.state.summary)
                HStack {
                    Spacer()
                    Button("Restart Server") { Task { await chat.restartServer() } }
                        .disabled(chat.server.state == .noFolder)
                    ChooseModelFolderButton()
                }
            }
            Section {
                Toggle("Think before answering", isOn: $isThinkingEnabled)
            } footer: {
                Text("Thinking helps with harder questions but makes replies slower.")
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .scrollDisabled(true)
        .frame(width: 520)
        .fixedSize(horizontal: false, vertical: true)
    }
}
