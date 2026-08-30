#!/bin/zsh
# Copyright (C) 2026 Daniel C. Homan.  Part of Leaf / LeafGUI, released under
# the GNU General Public License v3 or later; see LICENSE at the repository root.
# Bundle the Leaf engine into the built LeafGUI.app.
#
# Builds the engine with the NNUE net embedded (NNUE_EMBED=1) — the canonical
# form for release, since the only data file needed alongside the binary is then
# the opening book. Run after `flutter build macos --release`:
#
#   ./bundle_engine.sh [--debug] [--net <file.nnue>]
#
# This is the single source of truth for how the engine gets into the .app;
# scripts/build_release.sh drives the Flutter build and then calls it.
set -e

GUI_DIR="${0:a:h}"
ENGINE_DIR="$GUI_DIR/../engine"
NNUE_NET="nn-leaf-260414.nnue"
CONFIG=Release

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) CONFIG=Debug; shift ;;
    --net)   NNUE_NET="$2"; shift 2 ;;
    *) echo "usage: $0 [--debug] [--net <file.nnue>]" >&2; exit 2 ;;
  esac
done

APP="$GUI_DIR/build/macos/Build/Products/$CONFIG/LeafGUI.app"

if [[ ! -d "$APP" ]]; then
  echo "error: $APP not found — run 'flutter build macos --release' first" >&2
  exit 1
fi

if [[ ! -f "$ENGINE_DIR/run/$NNUE_NET" ]]; then
  echo "error: net $NNUE_NET not found in $ENGINE_DIR/run/" >&2
  exit 1
fi

# Build the embedded-net engine if it's missing, older than the sources, or
# older than the net being embedded.
BIN="$ENGINE_DIR/run/Leaf_vgui_embed"
if [[ ! -x "$BIN" ]] \
   || [[ -n "$(find "$ENGINE_DIR/src" -name '*.cpp' -newer "$BIN" 2>/dev/null | head -1)" ]] \
   || [[ "$ENGINE_DIR/run/$NNUE_NET" -nt "$BIN" ]]; then
  echo "Building embedded engine (net: $NNUE_NET)..."
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

# GPL distribution: ship the licence text inside the bundle.
cp "$GUI_DIR/../LICENSE" "$APP/Contents/Resources/LICENSE"

echo "Bundled $(basename "$BIN") + main_bk.dat + LICENSE -> $APP/Contents/Resources/"
