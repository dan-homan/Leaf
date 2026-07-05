#!/bin/zsh
# Bundle the Leaf engine into the built LeafGUI.app.
#
# Builds the engine with the NNUE net embedded (NNUE_EMBED=1), so the only
# data file needed alongside the binary is the opening book. Run after
# `flutter build macos --release`:
#
#   ./bundle_engine.sh [--debug]
#
set -e

GUI_DIR="${0:a:h}"
ENGINE_DIR="$GUI_DIR/../engine"
NNUE_NET="nn-leaf-260414.nnue"

CONFIG=Release
[[ "$1" == "--debug" ]] && CONFIG=Debug
APP="$GUI_DIR/build/macos/Build/Products/$CONFIG/LeafGUI.app"

if [[ ! -d "$APP" ]]; then
  echo "error: $APP not found — run 'flutter build macos --release' first" >&2
  exit 1
fi

# Build the embedded-net engine if it's missing or older than the sources.
BIN="$ENGINE_DIR/run/Leaf_vgui_embed"
if [[ ! -x "$BIN" ]] || [[ -n "$(find "$ENGINE_DIR/src" -name '*.cpp' -newer "$BIN" 2>/dev/null | head -1)" ]]; then
  echo "Building embedded engine..."
  (cd "$ENGINE_DIR/run" && perl comp.pl gui_embed NNUE=1 NNUE_EMBED=1 NNUE_NET="$NNUE_NET" OVERWRITE)
fi

DEST="$APP/Contents/Resources/engines"
mkdir -p "$DEST"
# Remove before copying: overwriting an existing Mach-O in place invalidates
# its code signature and the kernel SIGKILLs it on exec (Apple Silicon).
rm -f "$DEST/Leaf"
cp "$BIN" "$DEST/Leaf"
codesign --force -s - "$DEST/Leaf"
cp "$ENGINE_DIR/run/main_bk.dat" "$DEST/main_bk.dat"

echo "Bundled $(basename "$BIN") + main_bk.dat -> $DEST/"
