#!/bin/bash
#
# Copyright (C) 2026 Daniel C. Homan.  Part of Leaf / LeafGUI, released under
# the GNU General Public License v3 or later; see LICENSE at the repository root.
#
# Build a distributable LeafGUI.app with the Leaf engine bundled.
#
# Usage: ./scripts/build_release.sh [--net <file.nnue>]
#   --net: network to embed (default: the one gui/bundle_engine.sh names)
#
# The engine is built with NNUE_EMBED=1 — the net is compiled into the binary,
# so the shipped app carries no external .nnue file and cannot be run against a
# mismatched one.  This is the canonical release form.  gui/bundle_engine.sh
# owns that step; this script only drives the Flutter build around it.
#
# The resulting .app is placed in gui/build/macos/Build/Products/Release/
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Building LeafGUI ==="

cd "$REPO_ROOT/gui"
flutter pub get
flutter build macos --release

# Engine build + bundling (embedded net, opening book, LICENSE).
echo "=== Bundling engine into app ==="
./bundle_engine.sh "$@"

APP="$REPO_ROOT/gui/build/macos/Build/Products/Release/LeafGUI.app"

echo ""
echo "=== Build complete ==="
echo "App: $APP"
du -sh "$APP"
