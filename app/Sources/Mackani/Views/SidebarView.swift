import SwiftUI

struct SidebarView: View {
    @Environment(ChatStore.self) private var chat
    @State private var renamingID: UUID?
    @State private var draftTitle = ""
    @State private var pendingDeletion: Conversation?
    @FocusState private var isRenameFocused: Bool

    var body: some View {
        List(selection: Binding(get: { chat.selectedID }, set: { chat.select($0) })) {
            ForEach(chat.conversations) { conversation in
                row(for: conversation)
                    .tag(conversation.id)
                    .contextMenu {
                        Button("Rename") { beginRename(conversation) }
                        Divider()
                        Button("Delete…", role: .destructive) { pendingDeletion = conversation }
                    }
            }
        }
        .onDeleteCommand {
            if let selected = chat.selected { pendingDeletion = selected }
        }
        .overlay {
            if chat.conversations.isEmpty {
                ContentUnavailableView("No Conversations", systemImage: "bubble.left.and.bubble.right")
            }
        }
        .confirmationDialog(
            "Delete “\(pendingDeletion?.title ?? "")”?",
            isPresented: Binding(get: { pendingDeletion != nil }, set: { if !$0 { pendingDeletion = nil } }),
            presenting: pendingDeletion
        ) { conversation in
            Button("Delete", role: .destructive) { chat.delete(conversation.id) }
        } message: { _ in
            Text("The conversation is removed from this Mac. This cannot be undone.")
        }
    }

    @ViewBuilder
    private func row(for conversation: Conversation) -> some View {
        if renamingID == conversation.id {
            TextField("Title", text: $draftTitle)
                .focused($isRenameFocused)
                .onSubmit { commitRename(conversation) }
                .onExitCommand { renamingID = nil }
                .onChange(of: isRenameFocused) { _, focused in
                    if !focused { commitRename(conversation) }
                }
        } else {
            Text(conversation.title)
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }

    private func beginRename(_ conversation: Conversation) {
        draftTitle = conversation.title
        renamingID = conversation.id
        isRenameFocused = true
    }

    private func commitRename(_ conversation: Conversation) {
        guard renamingID == conversation.id else { return }
        renamingID = nil
        if draftTitle != conversation.title { chat.rename(conversation.id, to: draftTitle) }
    }
}
