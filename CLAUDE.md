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

# GUI release build (from repo root) — flutter is on PATH via Homebrew cask
./scripts/build_release.sh
```

## Key Facts

- Author: Daniel C. Homan
- Licence: GPL v3-or-later across engine, scripts, and GUI (`LICENSE` at repo root; source files carry short headers)
- Engine and GUI developed in collaboration with Claude Code (Anthropic)
- Engine binary: `engine/run/Leaf_v<version>` — requires `main_bk.dat` and the `.nnue` net in the same directory (unless built `NNUE_EMBED=1`).  `search.par` no longer exists; search defaults are compiled in
- Release engine builds are always `NNUE_EMBED=1` (net compiled in); `gui/bundle_engine.sh` owns that step, `scripts/build_release.sh` drives the Flutter build around it
- GUI dev path: hardcoded in `gui/lib/models/engine_config.dart`, checks bundled path first
- macOS sandbox disabled to allow engine subprocess spawning
- GUI opens as a Flutter project from the `gui/` directory
