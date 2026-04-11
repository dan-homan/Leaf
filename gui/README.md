# LeafGUI

A cross-platform chess GUI for the [Leaf](https://github.com/user/Leaf) chess engine, built with [Flutter](https://flutter.dev/).

LeafGUI supports standard chess and Fischer Random (Chess960), with UCI protocol compatibility for engine communication.

## Features

- Play against any UCI chess engine, or watch engine-vs-engine matches
- Standard chess and Chess960 (Fischer Random) support
  - Random or specific position selection (0-959)
  - UCI_Chess960 option sent to engines automatically
- Multiple time controls: game time + increment, fixed time per move, fixed depth
- Engine pondering (thinks on your time)
- Engine output display (depth, score, PV, nodes/sec)
  - Dual engine output in engine-vs-engine mode with engine names
- Move list with full game history and navigation (back, forward, jump to move)
- FEN copy/paste and position loading
- Engine registry: remembers previously used engines for quick selection
- Engine picker with native file browser and dropdown of known engines
- Per-engine skill level adjustment
- Engine names displayed on board clocks
- Watchdog recovery for engine-vs-engine stalls
- Analysis mode
- Clean, minimal black-and-white board design

## Platforms

- macOS (primary development target)
- Windows
- Linux
- iOS (planned)

## Building

### Prerequisites

- [Flutter SDK](https://docs.flutter.dev/get-started/install) (3.41+)
- A compiled Leaf binary

### Setup

1. Clone this repository
2. Install dependencies:
   ```
   flutter pub get
   ```
3. Place the Leaf engine binary:
   - **macOS (development):** The app will look for the engine at the bundled path. For development, you can modify `EngineConfig.defaultEnginePath()` in `lib/models/engine_config.dart` to point to your local Leaf binary.
   - **Production:** Place the Leaf binary in the appropriate platform location:
     - macOS: `YourApp.app/Contents/Resources/engines/Leaf`
     - Windows: `<app-dir>/engines/Leaf.exe`
     - Linux: `<app-dir>/engines/Leaf`

4. Run:
   ```
   flutter run -d macos
   ```

## Project Structure

```
lib/
  main.dart              - App entry point
  app.dart               - MaterialApp configuration
  models/
    game_state.dart      - Game options, time control models
    engine_config.dart   - Engine path and settings
  providers/
    game_provider.dart   - Game state management (Riverpod)
    engine_provider.dart - Engine lifecycle management
  services/
    uci_engine.dart      - UCI protocol client
    engine_registry.dart - Persistent engine registry (~/.leafgui/engines.json)
    pv_formatter.dart    - PV line formatting (long algebraic to SAN)
  ui/
    screens/
      home_screen.dart   - Main playing screen
    widgets/
      board_widget.dart  - Chess board (squares package)
      clock_widget.dart  - Game clocks
      engine_output.dart - Engine info display (dual in engine-vs-engine)
      engine_picker.dart - Engine selection dropdown + file browser
      engine_settings_dialog.dart - Engine configuration dialog
      game_controls.dart - New game, undo, resign, FEN
      move_list.dart     - Move history panel with navigation
      new_game_dialog.dart - New game configuration
    theme/
      app_theme.dart     - Application theme
      board_theme.dart   - Board colors and markers
```

## Dependencies and Credits

LeafGUI is built on top of several excellent open-source libraries:

### Chess Logic and Board UI

- **[bishop](https://pub.dev/packages/bishop)** (MIT License) by Alex Baker
  Chess logic engine with full variant support including Chess960.
  Repository: https://github.com/alexobviously/bishop

- **[squares](https://pub.dev/packages/squares)** (MIT License) by Alex Baker
  Flutter chessboard widget with drag-and-drop, animations, and theming.
  Repository: https://github.com/alexobviously/squares

- **[square_bishop](https://pub.dev/packages/square_bishop)** (MIT License) by Alex Baker
  Bridge package connecting bishop game logic to squares board UI.
  Repository: https://github.com/alexobviously/square_bishop

### State Management

- **[flutter_riverpod](https://pub.dev/packages/flutter_riverpod)** (MIT License) by Remi Rousselet
  Reactive state management for Flutter.
  Repository: https://github.com/rrousselGit/riverpod

### Engine

- **Leaf** by Daniel Homan
  UCI-compatible chess engine supporting standard chess and Chess960.

## License

Copyright (c) 2026 Daniel Homan. All rights reserved.
