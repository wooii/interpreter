#!/bin/bash
# Build the macOS Interpreter.app bundle — a thin bundle whose launcher runs
# the project's host venv (`python -m interpreter app`). Double-click
# dist/Interpreter.app in Finder to launch.
#
# Run on the macOS host (needs `iconutil`, built into macOS):
#     ./scripts/make_app.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "host venv missing — run 'uv sync' first" >&2
    exit 1
fi

"$PY" scripts/make_icon.py
iconutil -c icns "$ROOT/build/Interpreter.iconset" -o "$ROOT/build/Interpreter.icns"

APP="$ROOT/dist/Interpreter.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Interpreter</string>
    <key>CFBundleDisplayName</key>
    <string>Interpreter</string>
    <key>CFBundleIdentifier</key>
    <string>local.interpreter.app</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>interpreter</string>
    <key>CFBundleIconFile</key>
    <string>Interpreter.icns</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/interpreter" <<LAUNCHER
#!/bin/bash
# Launcher for the thin Interpreter.app bundle: run the project's host venv.
# The repo path is baked in at build time, so the bundle works even when
# moved to /Applications (a walk-up from the bundle is the fallback for when
# the repo path changed). All stderr goes to a log file — Finder launches
# have no terminal, so failures must not be silent.
set -euo pipefail
LOG="\$HOME/Library/Logs/Interpreter.log"
mkdir -p "\$(dirname "\$LOG")"
exec 2>>"\$LOG"
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] launching Interpreter.app" >>"\$LOG"

ROOT="$ROOT"
if [ ! -f "\$ROOT/pyproject.toml" ] || [ ! -d "\$ROOT/.venv" ]; then
    echo "baked path \$ROOT is stale, walking up from the bundle" >>"\$LOG"
    ROOT=""
    BUNDLE_DIR="\$(cd "\$(dirname "\$0")/../.." && pwd)"
    DIR="\$BUNDLE_DIR"
    while [ "\$DIR" != "/" ]; do
        if [ -f "\$DIR/pyproject.toml" ] && [ -d "\$DIR/.venv" ]; then
            ROOT="\$DIR"
            break
        fi
        DIR="\$(dirname "\$DIR")"
    done
fi
if [ -z "\$ROOT" ]; then
    echo "could not find the project (pyproject.toml + .venv)" >>"\$LOG"
    exit 1
fi
cd "\$ROOT"
exec "\$ROOT/.venv/bin/python" -m interpreter app
LAUNCHER
chmod +x "$APP/Contents/MacOS/interpreter"

cp "$ROOT/build/Interpreter.icns" "$APP/Contents/Resources/Interpreter.icns"

echo "Built $APP"
echo "Launch: open \"$APP\"  (or double-click it in Finder)"