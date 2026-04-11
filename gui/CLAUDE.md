# LeafGUI

A Flutter chess GUI for the Leaf chess engine.

## Build & Run

```bash
export PATH="$HOME/develop/flutter/bin:$PATH"
flutter pub get
flutter build macos
flutter run -d macos
```

Static analysis: `flutter analyze`

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
UCI protocol over stdin/stdout pipes. `UciEngine` class spawns the engine process (no flags — engine auto-detects UCI), sets `workingDirectory` to the engine's directory so it can find data files (book, NNUE, etc.), and parses responses. Streams expose `UciInfo`, bestmove, and engine state changes to the UI via Riverpod providers.

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

- Source: ~/Leaf (C++, ~17,800 lines)
- Dev binary: /Users/danielhoman/Leaf/run/Leaf_vcurrent
- Supports UCI and xboard protocols with auto-detection
- Full Chess960/Fischer Random support
- UCI options: Hash (1-4096 MB), Threads (1-32), Ponder, UCI_AnalyseMode
- Auto-detects UCI mode (no `--uci` flag needed)
- Requires data files alongside the binary: `main_bk.dat`, `search.par`, and the compiled-in NNUE net

`defaultEnginePath()` checks for a bundled engine at `Contents/Resources/engines/Leaf` first, then falls back to the dev binary path for development.

## Platform Configuration

- macOS sandbox is disabled (both Debug and Release entitlements) to allow spawning the engine subprocess
- Window defaults: 1200x800, minimum 900x600
- Targets: macOS (primary), Windows, Linux. iOS planned.

## Current State (MVP / Tier 1)

Working: board display, piece movement, engine communication, new game dialog, Chess960, time controls (game+inc, fixed time, fixed depth), clocks, move list with navigation (back/forward/jump), engine output (dual display in engine-vs-engine), FEN copy/load, undo, ponder support, engine-vs-engine mode with watchdog recovery, engine registry with persistent storage, engine picker UI with native file browser, engine names displayed on board clocks, skill level per engine, UCI_Chess960 support, bundled engine distribution.

Not yet implemented: PGN import/export, board themes/piece sets, resign button, sound effects.
