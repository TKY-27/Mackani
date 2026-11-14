// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Mackani",
    platforms: [.macOS(.v26)],
    targets: [
        .executableTarget(name: "Mackani"),
        .testTarget(name: "MackaniTests", dependencies: ["Mackani"]),
    ]
)
