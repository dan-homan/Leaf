# Leaf — Monorepo

This repository contains both the Leaf chess engine (C++) and LeafGUI (Flutter).

## Structure

```
engine/
  src/          C++ source code (unity build)
  docs/         Engine documentation
  scripts/      Python automation scripts
  run/          Compiled binaries and runtime data files
  learn/        Training artifacts (.nnue, .tdleaf.bin, PGN)
gui/            Flutter chess GUI (LeafGUI)
logos/           Shared logo assets
tools/           Third-party tools (cutechess, BayesElo)
testing/         Test suites and opening books
```

## Component Documentation

- **Engine:** See `engine/CLAUDE.md` for build system, architecture, NNUE, TDLeaf, and code conventions.
- **GUI:** See `gui/CLAUDE.md` for Flutter project structure, providers, widgets, and engine communication.

## Quick Build

```sh
# Console engine (from engine/run/)
cd engine/run/ && perl comp.pl <version> NNUE=1

# GUI (from gui/)
cd gui/ && flutter pub get && flutter build macos --release
```

## Key Facts

- Author: Daniel C. Homan
- Engine and GUI developed in collaboration with Claude Code (Anthropic)
- Engine binary: `engine/run/Leaf_v<version>` — requires `main_bk.dat`, `search.par`, and `.nnue` in same directory
- GUI dev path: hardcoded in `gui/lib/models/engine_config.dart`, checks bundled path first
- macOS sandbox disabled to allow engine subprocess spawning
- GUI opens as a Flutter project from the `gui/` directory
