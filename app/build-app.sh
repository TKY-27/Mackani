#!/bin/sh
# Builds build/Mackani.app: the SwiftUI app with the mackani server next to its executable.
# Usage: ./build-app.sh [path/to/mackani]   (or set MACKANI_SERVER_BINARY)
set -eu

cd "$(dirname "$0")"
server="${1:-${MACKANI_SERVER_BINARY:-../target/release/mackani}}"

if [ ! -x "$server" ]; then
    echo "error: the server binary was not found at $server" >&2
    echo "Build it from the repository root with: cargo build --release -p mackani" >&2
    echo "or pass its path: ./build-app.sh /path/to/mackani" >&2
    exit 1
fi

swift build -c release
executable="$(swift build -c release --show-bin-path)/Mackani"

app=build/Mackani.app
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
# The file system is usually case-insensitive, so the app's executable cannot be called "Mackani"
# next to the server "mackani".
cp "$executable" "$app/Contents/MacOS/MackaniApp"
cp "$server" "$app/Contents/MacOS/mackani"

cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Mackani</string>
    <key>CFBundleDisplayName</key>
    <string>Mackani</string>
    <key>CFBundleIdentifier</key>
    <string>io.github.tky-27.mackani</string>
    <key>CFBundleExecutable</key>
    <string>MackaniApp</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>LSMinimumSystemVersion</key>
    <string>26.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.productivity</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>
PLIST

codesign --force --sign - "$app/Contents/MacOS/mackani"
codesign --force --sign - "$app"
echo "Built $(pwd)/$app"
