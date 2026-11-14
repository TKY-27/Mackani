import SwiftUI

struct ContentView: View {
    @Environment(ChatStore.self) private var chat

    var body: some View {
        NavigationSplitView {
            SidebarView()
                .navigationSplitViewColumnWidth(min: 200, ideal: 240, max: 340)
        } detail: {
            ChatView(conversation: chat.selected)
        }
    }
}
