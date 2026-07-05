# LeafGUI

A Flutter chess GUI for the Leaf chess engine.

## Build & Run

Flutter is installed via Homebrew (`brew install --cask flutter`) and is on the PATH.
Building/running the macOS app requires **full Xcode** (not just Command Line Tools);
`flutter analyze` and `flutter test` work without it.

```bash
flutter pub get
flutter build macos --release
./bundle_engine.sh          # copies the embedded-net engine into the .app
flutter run -d macos
```

**Conda gotcha:** miniforge's base env exports a full cross-toolchain into every
shell (`CC`/`LD`/`LIPO`/`AR`/`STRIP`/... = `arm64-apple-darwin20.0.0-*`, plus
`SDKROOT`, `LDFLAGS`), and Xcode spawns whichever of those tools the environment
names — the build fails with errors like `unable to spawn process
'arm64-apple-darwin20.0.0-ld'`. Scrub before building:

```bash
unset $(env | grep -E "arm64-apple-darwin|miniforge3" | cut -d= -f1 | grep -v '^PATH$') SDKROOT
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
flutter build macos --release
```

Static analysis: `flutter analyze` — Tests: `flutter test`

## Project Structure

```
lib/
  main.dart              - Entry point, wraps app in ProviderScope
  app.dart               - MaterialApp with dark theme
  models/
    game_state.dart      - Game options, time control, new game config
    engine_config.dart   - Engine path and UCI settings
  providers/
    game_provider.dart   - Core game state (GameNotifier/GameState), clocks, mode
    engine_provider.dart - Engine lifecycle, streams, bestmove forwarding
  services/
    uci_engine.dart      - UCI protocol client (process spawn, command parsing)
    engine_registry.dart - Persists known engines to ~/.leafgui/engines.json
    pv_formatter.dart    - Converts UCI PV (long algebraic) to SAN for display
  ui/
    screens/
      home_screen.dart   - Main layout: board left, info panel right
    widgets/
      board_widget.dart  - BoardController wrapper with theme
      clock_widget.dart  - Game clock display
      engine_output.dart - Depth, score, PV, nodes/sec panel (dual in eve mode)
      engine_picker.dart - Dropdown of registered engines + native file browse
      engine_settings_dialog.dart - Hash, threads, skill, ponder, engine selection
      game_controls.dart - New game, undo, resign, FEN copy/load
      move_list.dart     - Move history in algebraic notation with scrollbar
      new_game_dialog.dart - Color, time control, Chess960, engine selection
    theme/
      app_theme.dart     - Dark theme colors
      board_theme.dart   - Black-and-white board + marker theme
```

## Key Dependencies

- **bishop** (1.4.4, MIT) - Chess logic, move generation, Chess960 support
- **squares** (1.2.1, MIT) - Flutter chessboard widget
- **square_bishop** (0.3.0, MIT) - Bridge between bishop and squares
- **flutter_riverpod** (3.3.1) - State management

All three chess packages are by Alex Baker. We use `StateNotifierProvider` and `StateProvider` from `package:flutter_riverpod/legacy.dart`.

## Architecture Notes

### Engine Communication
UCI protocol over stdin/stdout pipes. `UciEngine` class spawns the engine process (no flags — engine auto-detects UCI), sets `workingDirectory` to the engine's directory (belt-and-braces; the engine also resolves its data files via its own executable path), and parses responses. Streams expose `UciInfo`, bestmove, and engine state changes to the UI via Riverpod providers. Aspiration fail-high/low lines (`score ... lowerbound/upperbound`) are filtered out by the parser so the eval display only shows settled scores.

**Position commands always send the game's initial FEN plus the full move list** (`_sendPosition` in `game_provider.dart`), never a bare FEN of the current position. The engine rebuilds its repetition history (`plist`) from the replayed move list; a bare FEN would leave its threefold-repetition detection blind. `GameNotifier._initialFen` is captured at `newGame`/`loadFen` (bishop's `BishopState` has no FEN getter, so it can't be recovered later).

**Eval display is normalized to white's POV.** UCI scores are side-to-move relative; `_sendPosition` records the STM of each searched position (including ponder positions, which are one ply ahead) in `engineSearchStmProvider` / `engine2SearchStmProvider`, and `engine_output.dart` flips the sign accordingly.

`UCI_AnalyseMode` is set true on entering analysis (and on `loadFen`, which enters analysis and starts an infinite search), false on exit and at new-game init.

### Engine Registry & Picker
`EngineRegistry` (singleton) persists known engines to `~/.leafgui/engines.json`. Engines are auto-registered after a successful UCI handshake. The `EnginePicker` widget provides a dropdown of registered engines plus a browse button using a native macOS file picker (via `MethodChannel('leaf_gui/file_picker')` → `NSOpenPanel` in `MainFlutterWindow.swift`).

### Engine vs Engine Mode
`GameMode.engineVsEngine` runs two engines: engine 1 (white) via `engineProvider`, engine 2 (black) via `engine2Provider`. Each has its own config, info, state, skill level, and name providers. A 5-second watchdog timer (`_eveWatchdog`) detects stalls where an engine goes idle unexpectedly and nudges it with a `sync()` + `go` command. The `_ignoreNextBestMove` / `_ignoreNextBestMove2` flags prevent stale bestmove responses from being applied after a `stop` command. A monotonic `_gameId` counter guards against stale async init callbacks across game boundaries.

### Game State
`GameNotifier` wraps a mutable `bishop.Game` inside `GameState` (which adds a version counter to trigger Riverpod rebuilds). The game object is mutated in place; `_notify()` increments the version to signal changes.

`bishop.Game` does NOT have `copyWith()`. Do not try to copy it — mutate and notify.

### Board Widget
`BoardController` from squares takes `SquaresState` (produced by `game.squaresState(player)` via square_bishop). The `onMove` callback receives a `squares.Move` where `promo` is a `String?` (uppercase letter like "Q"), not an int.

### Chess960
Created via `Game(variant: Variant.chess960(), startPosSeed: n)` where n is 0-959. Random position: omit startPosSeed or pass `Random().nextInt(960)`.

### Game Result Checking
Use `game.checkmate`, `game.stalemate`, `game.gameOver` (NOT `inCheckmate`/`inStalemate` — those don't exist).

### History Access
Move SAN notation: `game.history[i].meta?.moveMeta?.formatted` (NOT `meta?.formatted`).
`history[0]` is the initial position. `BishopState` does NOT have a `fen` getter — only `Game.fen` produces FEN.

## Leaf Engine

- Source: ~/Leaf/engine (C++)
- Dev binary: /Users/homand/Leaf/engine/run/Leaf_vcurrent
- Supports UCI and xboard protocols with auto-detection
- Full Chess960/Fischer Random support
- UCI options: Hash (1-4096 MB), Threads (1-32), Ponder, UCI_AnalyseMode, UCI_Chess960, Skill (1-100)
- Auto-detects UCI mode (no `--uci` flag needed)
- Data files (`main_bk.dat` opening book, `.nnue` net) are resolved via the engine executable's own directory; `search.par` no longer exists (defaults are compiled in)
- The bundled engine is built with `NNUE_EMBED=1` (net baked into the binary), so only `main_bk.dat` ships alongside it — see `bundle_engine.sh`

`defaultEnginePath()` checks for a bundled engine at `Contents/Resources/engines/Leaf` first, then falls back to the dev binary path for development.

## Platform Configuration

- macOS sandbox is disabled (both Debug and Release entitlements) to allow spawning the engine subprocess
- Window defaults: 1200x800, minimum 900x600
- Targets: macOS (primary), Windows, Linux. iOS planned.

## Current State (MVP / Tier 1)

Working: board display, piece movement, engine communication, new game dialog, Chess960, time controls (game+inc, fixed time, fixed depth), clocks, move list with navigation (back/forward/jump), engine output (dual display in engine-vs-engine, white-POV eval), FEN copy/load, undo, ponder support (setting mirrored to the engine's Ponder option), engine-vs-engine mode with watchdog recovery, engine registry with persistent storage, engine picker UI with native file browser, engine names displayed on board clocks, skill level per engine, UCI_Chess960 support, UCI_AnalyseMode, full move-history position commands (engine repetition detection), bundled engine distribution (embedded NNUE net via `bundle_engine.sh`).

## TODO

- **PGN import/export** — move history and SAN are already available in `GameNotifier`; needs headers, result tags, clipboard/file I/O.
- **MultiPV in analysis mode** — the GUI parser already captures the `multipv` index, but the engine doesn't offer a MultiPV option yet; engine-side work first.
- Board themes/piece sets, resign button, sound effects.
