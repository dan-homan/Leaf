#!/bin/bash
#
# Build a distributable LeafGUI.app with Leaf engine bundled.
#
# Usage: ./scripts/build_release.sh [engine_version]
#   engine_version: name of the binary in engine/run/ (default: Leaf_vcurrent)
#
# The resulting .app is placed in gui/build/macos/Build/Products/Release/
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE_BIN="${1:-Leaf_vcurrent}"
ENGINE_PATH="$REPO_ROOT/engine/run/$ENGINE_BIN"

if [ ! -f "$ENGINE_PATH" ]; then
    echo "Error: Engine binary not found at $ENGINE_PATH"
    exit 1
fi

# Detect which NNUE file the binary expects.
NNUE_FILE=$(strings "$ENGINE_PATH" | grep '\.nnue$' | head -1)
if [ -z "$NNUE_FILE" ]; then
    echo "Warning: Could not detect NNUE file from binary"
fi

echo "=== Building LeafGUI ==="
echo "Engine binary: $ENGINE_BIN"
echo "NNUE file: ${NNUE_FILE:-none detected}"

# Build the Flutter app.
cd "$REPO_ROOT/gui"
export PATH="$HOME/develop/flutter/bin:$PATH"
flutter pub get
flutter build macos --release

# Bundle engine files into the app.
APP="$REPO_ROOT/gui/build/macos/Build/Products/Release/LeafGUI.app"
ENGINES_DIR="$APP/Contents/Resources/engines"
mkdir -p "$ENGINES_DIR"

echo "=== Bundling engine into app ==="
cp "$ENGINE_PATH" "$ENGINES_DIR/Leaf"
chmod +x "$ENGINES_DIR/Leaf"

for DATA_FILE in main_bk.dat search.par; do
    if [ -f "$REPO_ROOT/engine/run/$DATA_FILE" ]; then
        cp "$REPO_ROOT/engine/run/$DATA_FILE" "$ENGINES_DIR/"
        echo "  Copied $DATA_FILE"
    else
        echo "  Warning: $DATA_FILE not found in engine/run/"
    fi
done

if [ -n "$NNUE_FILE" ] && [ -f "$REPO_ROOT/engine/run/$NNUE_FILE" ]; then
    cp "$REPO_ROOT/engine/run/$NNUE_FILE" "$ENGINES_DIR/"
    echo "  Copied $NNUE_FILE"
elif [ -n "$NNUE_FILE" ]; then
    echo "  Warning: $NNUE_FILE not found in engine/run/"
fi

echo ""
echo "=== Build complete ==="
echo "App: $APP"
du -sh "$APP"
