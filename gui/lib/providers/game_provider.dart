import 'dart:async';
import 'dart:math';
import 'package:bishop/bishop.dart' as bp;
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_riverpod/legacy.dart';
import 'package:squares/squares.dart' as sq;
import 'package:square_bishop/square_bishop.dart';
import '../models/engine_config.dart';
import '../models/game_state.dart';
import '../services/uci_engine.dart';
import 'engine_provider.dart';

/// The player orientation (which color the human is playing).
final playerColorProvider = StateProvider<int>((ref) => sq.Squares.white);

/// Whether we are in analysis mode.
final gameModeProvider = StateProvider<GameMode>((ref) => GameMode.play);

/// Current time control settings.
final timeControlProvider = StateProvider<TimeControl>((ref) {
  return const TimeControl.gameTime();
});

/// Whether pondering is enabled.
final ponderEnabledProvider = StateProvider<bool>((ref) => true);

/// Whether the current game is Chess960.
final chess960Provider = StateProvider<bool>((ref) => false);

/// Clock state: remaining time for white and black in milliseconds.
final whiteClockProvider = StateProvider<int>((ref) => 300000);
final blackClockProvider = StateProvider<int>((ref) => 300000);

/// Whether engine vs engine match is paused.
final matchPausedProvider = StateProvider<bool>((ref) => false);

/// Wrapper around a bishop Game to use as notifier state.
/// We need this because Game is mutable and doesn't support copyWith.
class GameState {
  final bp.Game game;
  final int version; // incremented on every change to trigger rebuilds
  /// Total number of plies (half-moves) in the game, including undone moves.
  final int totalPlies;
  /// Current ply being viewed (0 = initial position, totalPlies = latest).
  final int viewPly;

  const GameState(this.game, this.version, {this.totalPlies = 0, this.viewPly = 0});

  /// Whether we're viewing a historical position (not the latest).
  bool get isViewingHistory => viewPly < totalPlies;
}

/// The core game state notifier wrapping a bishop Game.
class GameNotifier extends StateNotifier<GameState> {
  final Ref ref;
  Timer? _clockTimer;
  DateTime? _lastClockTick;
  int _version = 0;

  /// Stack of undone moves for redo/navigation.
  final List<({String algebraic, String san})> _redoStack = [];

  /// Guard to prevent concurrent _restartAnalysis calls from interleaving.
  bool _analysisRestarting = false;

  /// Tracks whether we already retried analysis after an unexpected bestmove.
  bool _analysisRetried = false;

  /// Set when we send "stop" — the resulting bestmove must be ignored.
  bool _ignoreNextBestMove = false;

  /// Same flag for engine 2 in engine-vs-engine mode.
  bool _ignoreNextBestMove2 = false;

  /// Monotonic game ID — incremented on every newGame to detect stale async inits.
  int _gameId = 0;

  /// Watchdog timer for engine-vs-engine stalls.
  Timer? _eveWatchdog;

  GameNotifier(this.ref) : super(GameState(bp.Game(variant: bp.Variant.standard()), 0));

  bp.Game get game => state.game;

  int get _currentPly => game.history.length - 1;
  int get _totalPlies => _currentPly + _redoStack.length;

  void _notify() {
    _version++;
    state = GameState(game, _version,
        totalPlies: _totalPlies, viewPly: _currentPly);
  }

  /// Start the clock if using game time and the clock isn't already running.
  void startClockIfNeeded() {
    final tc = ref.read(timeControlProvider);
    if (tc.type != TimeControlType.gameTime || _clockTimer != null || game.gameOver) return;
    // In engine-vs-engine, always start immediately.
    if (ref.read(gameModeProvider) == GameMode.engineVsEngine) {
      _startClock();
      return;
    }
    // Don't auto-start at the beginning — let the human move first.
    if (_currentPly == 0) return;
    _startClock();
  }

  /// Start a new standard chess game.
  void newGame(NewGameOptions options) {
    _stopClock();
    _redoStack.clear();

    // Stop any running engines BEFORE clearing flags.
    // The ignore flags must stay true until the stale bestmoves are consumed.
    _stopBothEngines();
    _eveWatchdog?.cancel();
    _eveWatchdog = null;

    // Now clear analysis flags (but NOT the ignore flags — those were just set
    // by _stopBothEngines and need to stay until the stale bestmoves arrive).
    _analysisRestarting = false;
    _analysisRetried = false;
    _gameId++;
    ref.read(matchPausedProvider.notifier).state = false;

    final bp.Variant variant;
    int? seed;

    if (options.chess960) {
      variant = bp.Variant.chess960();
      seed = options.chess960Position ?? Random().nextInt(960);
    } else {
      variant = bp.Variant.standard();
    }

    final newGame = bp.Game(variant: variant, startPosSeed: seed);

    // Set player orientation.
    final playerColor = options.playerColor == PlayerColor.white
        ? sq.Squares.white
        : sq.Squares.black;
    ref.read(playerColorProvider.notifier).state = playerColor;
    ref.read(chess960Provider.notifier).state = options.chess960;

    // Set game mode.
    if (options.engineVsEngine) {
      ref.read(gameModeProvider.notifier).state = GameMode.engineVsEngine;
    } else {
      ref.read(gameModeProvider.notifier).state = GameMode.play;
    }

    // Reset clocks.
    final tc = options.timeControl;
    ref.read(timeControlProvider.notifier).state = tc;
    if (tc.type == TimeControlType.gameTime) {
      ref.read(whiteClockProvider.notifier).state = tc.whiteTime.inMilliseconds;
      ref.read(blackClockProvider.notifier).state = tc.blackTime.inMilliseconds;
    }

    _version++;
    state = GameState(newGame, _version, totalPlies: 0, viewPly: 0);

    if (options.engineVsEngine) {
      _initEngineVsEngine(options);
    } else {
      _initEngine1AndGame(options);
    }
  }

  Future<void> _initEngine1AndGame(NewGameOptions options) async {
    final initGameId = _gameId;
    // If a different engine path was requested, update config and restart.
    if (options.engine1Path != null) {
      final currentConfig = ref.read(engineConfigProvider);
      if (currentConfig.path != options.engine1Path) {
        ref.read(engineConfigProvider.notifier).state = EngineConfig(
          path: options.engine1Path!,
          hashMb: currentConfig.hashMb,
          threads: currentConfig.threads,
        );
        await ref.read(engineProvider.notifier).startEngine();
        if (_gameId != initGameId) return;
      }
    }
    final engine = ref.read(engineProvider);
    if (engine != null && engine.isRunning) {
      _initEngineForGame(engine, options);
    }
  }

  Future<void> _initEngineForGame(UciEngine engine, NewGameOptions options) async {
    // Tell engine about Chess960.
    if (options.chess960) {
      engine.setOption('UCI_Chess960', 'true');
    } else {
      engine.setOption('UCI_Chess960', 'false');
    }

    await engine.newGame();
    // Clear stale flags — engine is now fresh and ready.
    _ignoreNextBestMove = false;
    _ignoreNextBestMove2 = false;
    engine.setPosition(fen: game.fen);

    final humanIsWhite = options.playerColor == PlayerColor.white;
    if (!humanIsWhite) {
      _engineGo(engine);
    }

    if (options.timeControl.type == TimeControlType.gameTime && !humanIsWhite) {
      _startClock();
    }
  }

  /// Initialize engine vs engine game.
  Future<void> _initEngineVsEngine(NewGameOptions options) async {
    final initGameId = _gameId;

    // Restart engine 1 if a different path was requested.
    if (options.engine1Path != null) {
      final currentConfig = ref.read(engineConfigProvider);
      if (currentConfig.path != options.engine1Path) {
        ref.read(engineConfigProvider.notifier).state = EngineConfig(
          path: options.engine1Path!,
          hashMb: currentConfig.hashMb,
          threads: currentConfig.threads,
        );
        await ref.read(engineProvider.notifier).startEngine();
        if (_gameId != initGameId) return; // new game started during await
      }
    }

    // Configure and start engine 2 if needed.
    final e2Path = options.engine2Path ?? EngineConfig.defaultEnginePath();
    ref.read(engine2ConfigProvider.notifier).state = EngineConfig(
      path: e2Path,
      hashMb: options.engine2Hash,
      threads: options.engine2Threads,
    );
    ref.read(engine2SkillLevelProvider.notifier).state = options.engine2Skill;

    // Start engine 2.
    final e2Started = await ref.read(engine2Provider.notifier).startEngine();
    if (!e2Started || _gameId != initGameId) return;

    // Prepare engine 1 (white).
    final engine1 = ref.read(engineProvider);
    final engine2 = ref.read(engine2Provider);
    if (engine1 == null || !engine1.isRunning) return;
    if (engine2 == null || !engine2.isRunning) return;

    // Tell engines about Chess960.
    if (options.chess960) {
      engine1.setOption('UCI_Chess960', 'true');
      engine2.setOption('UCI_Chess960', 'true');
    } else {
      engine1.setOption('UCI_Chess960', 'false');
      engine2.setOption('UCI_Chess960', 'false');
    }

    await engine1.newGame();
    if (_gameId != initGameId) return;
    await engine2.newGame();
    if (_gameId != initGameId) return;

    // Clear stale flags — both engines are now fresh and ready.
    _ignoreNextBestMove = false;
    _ignoreNextBestMove2 = false;

    engine1.setPosition(fen: game.fen);

    // Start the clock and tell engine 1 (white) to go.
    if (options.timeControl.type == TimeControlType.gameTime) {
      _startClock();
    }
    _engineGo(engine1);
    _resetEveWatchdog();
  }

  /// Make a move from the squares UI. Returns true if successful.
  bool makeMove(sq.Move move) {
    if (game.gameOver) return false;
    // No human moves in engine vs engine.
    if (ref.read(gameModeProvider) == GameMode.engineVsEngine) return false;

    _redoStack.clear();

    final success = game.makeSquaresMove(move);
    if (!success) return false;

    _notify();
    _addIncrement();
    startClockIfNeeded();

    if (game.gameOver) {
      _stopClock();
      return true;
    }

    final engine = ref.read(engineProvider);
    if (engine != null && engine.isRunning) {
      if (ref.read(gameModeProvider) == GameMode.analysis) {
        _restartAnalysis(engine);
      } else {
        final moveStr = _squaresMoveToUci(move);
        if (engine.state == UciEngineState.pondering &&
            engine.ponderMove == moveStr) {
          engine.ponderHit();
        } else {
          if (engine.state != UciEngineState.idle) {
            _ignoreNextBestMove = true;
            engine.stopSearch();
          }
          engine.setPosition(fen: game.fen);
          _engineGo(engine);
        }
      }
    }

    return true;
  }

  /// Undo the last move (and the engine's move before it in play mode).
  void undoMove() {
    if (!game.canUndo) return;

    final mode = ref.read(gameModeProvider);

    // In engine vs engine, stop both engines and undo.
    if (mode == GameMode.engineVsEngine) {
      _stopBothEngines();
      ref.read(matchPausedProvider.notifier).state = true;
      game.undo();
      _notify();
      return;
    }

    final engine = ref.read(engineProvider);
    if (engine != null && engine.state != UciEngineState.idle) {
      _ignoreNextBestMove = true;
      engine.stopSearch();
    }

    if (mode == GameMode.play) {
      _redoStack.clear();
      game.undo();
      if (game.canUndo) {
        game.undo();
      }
    } else {
      _pushUndoToRedo();
    }

    _notify();

    if (mode == GameMode.analysis && engine != null && engine.isRunning) {
      _restartAnalysis(engine);
    }
  }

  /// Push the current last move onto the redo stack before undoing.
  void _pushUndoToRedo() {
    if (!game.canUndo) return;
    final lastState = game.history.last;
    final algebraic = lastState.meta?.algebraic ?? '';
    final san = lastState.meta?.moveMeta?.formatted ?? '?';
    _redoStack.add((algebraic: algebraic, san: san));
    game.undo();
  }

  /// Navigate one move backward in history.
  void goBack() {
    if (!game.canUndo) return;

    _pushUndoToRedo();
    _notify();

    final engine = ref.read(engineProvider);
    if (ref.read(gameModeProvider) == GameMode.analysis &&
        engine != null && engine.isRunning) {
      _restartAnalysis(engine);
    }
  }

  /// Navigate one move forward in history.
  void goForward() {
    if (_redoStack.isEmpty) return;

    final entry = _redoStack.removeLast();
    game.makeMoveString(entry.algebraic);
    _notify();

    final engine = ref.read(engineProvider);
    if (ref.read(gameModeProvider) == GameMode.analysis &&
        engine != null && engine.isRunning) {
      _restartAnalysis(engine);
    }
  }

  /// Navigate to a specific ply (history index). 0 = initial position.
  void goToMove(int targetPly) {
    if (targetPly == _currentPly) return;
    if (targetPly < 0 || targetPly > _totalPlies) return;

    if (targetPly < _currentPly) {
      while (_currentPly > targetPly && game.canUndo) {
        _pushUndoToRedo();
      }
    } else {
      while (_currentPly < targetPly && _redoStack.isNotEmpty) {
        final entry = _redoStack.removeLast();
        game.makeMoveString(entry.algebraic);
      }
    }

    _notify();

    final engine = ref.read(engineProvider);
    if (ref.read(gameModeProvider) == GameMode.analysis &&
        engine != null && engine.isRunning) {
      _restartAnalysis(engine);
    }
  }

  void goToStart() => goToMove(0);
  void goToEnd() => goToMove(_totalPlies);

  bool get canGoForward => _redoStack.isNotEmpty;

  List<String> get redoMoveSans =>
      _redoStack.reversed.map((e) => e.san).toList();

  /// Swap sides with the engine.
  void swapSides() {
    if (game.gameOver) return;

    final engine = ref.read(engineProvider);
    if (engine != null && engine.state != UciEngineState.idle) {
      _ignoreNextBestMove = true;
      engine.stopSearch();
    }

    final current = ref.read(playerColorProvider);
    final newColor = current == sq.Squares.white
        ? sq.Squares.black
        : sq.Squares.white;
    ref.read(playerColorProvider.notifier).state = newColor;

    _notify();

    if (engine != null && engine.isRunning) {
      final enginesTurn = game.turn != newColor;
      if (enginesTurn) {
        _engineGo(engine);
      }
    }
  }

  /// Enter analysis mode.
  void enterAnalysis() {
    _stopClock();

    // Stop engine 2 if running (leaving engine-vs-engine).
    _stopEngine2IfRunning();

    ref.read(gameModeProvider.notifier).state = GameMode.analysis;
    _notify();

    final engine = ref.read(engineProvider);
    if (engine != null && engine.isRunning) {
      _restartAnalysis(engine);
    }
  }

  Future<void> _restartAnalysis(UciEngine engine) async {
    _analysisRestarting = true;
    _analysisRetried = false;
    try {
      if (engine.state != UciEngineState.idle) {
        _ignoreNextBestMove = true;
        engine.stopSearch();
      }
      await engine.isReady();
      if (ref.read(gameModeProvider) != GameMode.analysis) return;
      engine.setPosition(fen: game.fen);
      engine.go(infinite: true);
    } finally {
      _analysisRestarting = false;
    }
  }

  /// Exit analysis mode back to play mode.
  void exitAnalysis() {
    final engine = ref.read(engineProvider);
    if (engine != null && engine.state != UciEngineState.idle) {
      _ignoreNextBestMove = true;
      engine.stopSearch();
    }

    if (_redoStack.isNotEmpty) {
      while (_redoStack.isNotEmpty) {
        final entry = _redoStack.removeLast();
        game.makeMoveString(entry.algebraic);
      }
    }

    ref.read(gameModeProvider.notifier).state = GameMode.play;
    _notify();

    final tc = ref.read(timeControlProvider);
    if (tc.type == TimeControlType.gameTime && !game.gameOver) {
      _startClock();
    }
  }

  /// Load a position from FEN.
  void loadFen(String fen, {bool chess960 = false}) {
    _stopClock();
    _redoStack.clear();
    _stopEngine2IfRunning();
    final variant = chess960 ? bp.Variant.chess960() : bp.Variant.standard();
    final newGame = bp.Game(variant: variant, fen: fen);
    ref.read(chess960Provider.notifier).state = chess960;
    ref.read(gameModeProvider.notifier).state = GameMode.analysis;
    _version++;
    state = GameState(newGame, _version, totalPlies: 0, viewPly: 0);
  }

  String get fen => game.fen;

  List<String> get moveHistory {
    final moves = <String>[];
    for (int i = 1; i < game.history.length; i++) {
      final meta = game.history[i].meta;
      final algebraic = meta?.algebraic;
      if (algebraic != null) {
        moves.add(algebraic);
      }
    }
    return moves;
  }

  /// Get the SquaresState for the board widget.
  SquaresState get squaresState {
    final player = ref.read(playerColorProvider);
    final mode = ref.read(gameModeProvider);
    if (mode == GameMode.analysis) {
      final turn = game.turn;
      return SquaresState(
        player: turn,
        state: game.playState(turn),
        size: game.size.toSquares(),
        board: game.boardState(player),
        moves: game.squaresMoves(turn),
        history: game.squaresHistory,
        hands: game.handSymbols,
        gates: game.gateSymbols,
      );
    }
    // Engine vs engine or play mode viewing history: no moves allowed.
    if (mode == GameMode.engineVsEngine || _redoStack.isNotEmpty) {
      return SquaresState(
        player: player,
        state: game.playState(player),
        size: game.size.toSquares(),
        board: game.boardState(player),
        moves: const [],
        history: game.squaresHistory,
        hands: game.handSymbols,
        gates: game.gateSymbols,
      );
    }
    return game.squaresState(player);
  }

  /// Handle bestmove from engine 1.
  void onEngineBestMove(String bestMove, String? ponderMove) {
    if (game.gameOver) return;

    if (_ignoreNextBestMove) {
      _ignoreNextBestMove = false;
      return;
    }

    // In analysis mode, don't execute engine moves.
    if (ref.read(gameModeProvider) == GameMode.analysis) {
      if (!_analysisRestarting && !_analysisRetried) {
        _analysisRetried = true;
        final engine = ref.read(engineProvider);
        if (engine != null && engine.isRunning) {
          _restartAnalysis(engine);
        }
      }
      return;
    }

    // Engine vs engine: engine 1 plays white.
    if (ref.read(gameModeProvider) == GameMode.engineVsEngine) {
      _handleEveBestMove(bestMove, isEngine1: true);
      return;
    }

    // Normal play mode.
    _redoStack.clear();
    final success = game.makeMoveString(bestMove);
    if (!success) return;

    _notify();
    _addIncrement();

    if (game.gameOver) {
      _stopClock();
      return;
    }

    // Start pondering if enabled.
    if (ref.read(ponderEnabledProvider) && ponderMove != null) {
      final engine = ref.read(engineProvider);
      if (engine != null && engine.isRunning) {
        engine.setPosition(fen: game.fen, moves: [ponderMove]);
        final tc = ref.read(timeControlProvider);
        if (tc.type == TimeControlType.gameTime) {
          engine.goPonder(
            wtime: ref.read(whiteClockProvider),
            btime: ref.read(blackClockProvider),
            winc: tc.increment.inMilliseconds,
            binc: tc.increment.inMilliseconds,
            ponderMove: ponderMove,
          );
        } else {
          engine.go(ponder: true);
        }
      }
    }
  }

  /// Handle bestmove from engine 2.
  void onEngine2BestMove(String bestMove, String? ponderMove) {
    if (game.gameOver) return;

    if (_ignoreNextBestMove2) {
      _ignoreNextBestMove2 = false;
      return;
    }

    if (ref.read(gameModeProvider) != GameMode.engineVsEngine) return;

    _handleEveBestMove(bestMove, isEngine1: false);
  }

  /// Common handler for engine-vs-engine bestmove.
  void _handleEveBestMove(String bestMove, {required bool isEngine1}) {
    final success = game.makeMoveString(bestMove);
    if (!success) return;

    _notify();
    _addIncrement();

    if (game.gameOver) {
      _stopClock();
      _eveWatchdog?.cancel();
      return;
    }

    // If paused, don't start the next engine.
    if (ref.read(matchPausedProvider)) return;

    // Tell the other engine to go.
    // Send isready first as a sync barrier — some engines (e.g. Arasan) need
    // time to finish post-bestmove cleanup before accepting new commands.
    // The engine processes: isready → readyok → position → go in order.
    final nextEngine = isEngine1
        ? ref.read(engine2Provider)
        : ref.read(engineProvider);
    if (nextEngine != null && nextEngine.isRunning) {
      nextEngine.sync();
      _engineGo(nextEngine);
      _resetEveWatchdog();
    }
  }

  /// Pause the engine-vs-engine match.
  void pauseMatch() {
    ref.read(matchPausedProvider.notifier).state = true;
    _eveWatchdog?.cancel();
    _stopBothEngines();
  }

  /// Resume the engine-vs-engine match.
  void resumeMatch() {
    if (ref.read(gameModeProvider) != GameMode.engineVsEngine) return;
    if (game.gameOver) return;

    ref.read(matchPausedProvider.notifier).state = false;

    // Clear any stale ignore flags from the pause.
    _ignoreNextBestMove = false;
    _ignoreNextBestMove2 = false;

    // Determine whose turn it is and tell that engine to go.
    // Engine 1 = white (turn 0), Engine 2 = black (turn 1).
    final whiteToMove = game.turn == 0;
    final engine = whiteToMove
        ? ref.read(engineProvider)
        : ref.read(engine2Provider);
    if (engine != null && engine.isRunning) {
      engine.sync();
      engine.setPosition(fen: game.fen);
      _engineGo(engine);
      _resetEveWatchdog();
    }
  }

  void _stopBothEngines() {
    final e1 = ref.read(engineProvider);
    if (e1 != null && e1.state != UciEngineState.idle) {
      _ignoreNextBestMove = true;
      e1.stopSearch();
    }
    final e2 = ref.read(engine2Provider);
    if (e2 != null && e2.state != UciEngineState.idle) {
      _ignoreNextBestMove2 = true;
      e2.stopSearch();
    }
  }

  void _stopEngine2IfRunning() {
    final e2 = ref.read(engine2Provider);
    if (e2 != null && e2.state != UciEngineState.idle) {
      _ignoreNextBestMove2 = true;
      e2.stopSearch();
    }
  }

  /// Reset the engine-vs-engine watchdog timer.
  void _resetEveWatchdog() {
    _eveWatchdog?.cancel();
    _eveWatchdog = Timer(const Duration(seconds: 5), _eveWatchdogFired);
  }

  void _eveWatchdogFired() {
    if (ref.read(gameModeProvider) != GameMode.engineVsEngine) return;
    if (game.gameOver || ref.read(matchPausedProvider)) return;

    final whiteToMove = game.turn == 0;
    final activeEngine = whiteToMove
        ? ref.read(engineProvider)
        : ref.read(engine2Provider);

    if (activeEngine == null || !activeEngine.isRunning) return;

    if (activeEngine.state == UciEngineState.idle) {
      // Engine is idle but should be thinking — clear stale ignore flag
      // and nudge it.
      if (whiteToMove) {
        _ignoreNextBestMove = false;
      } else {
        _ignoreNextBestMove2 = false;
      }
      activeEngine.sync();
      _engineGo(activeEngine);
      _eveWatchdog = Timer(const Duration(seconds: 5), _eveWatchdogFired);
    }
    // If the engine is actively thinking, do nothing — it's working normally.
  }

  void _engineGo(UciEngine engine) {
    engine.setPosition(fen: game.fen);
    final tc = ref.read(timeControlProvider);
    switch (tc.type) {
      case TimeControlType.gameTime:
        engine.go(
          wtime: ref.read(whiteClockProvider),
          btime: ref.read(blackClockProvider),
          winc: tc.increment.inMilliseconds,
          binc: tc.increment.inMilliseconds,
        );
        break;
      case TimeControlType.fixedTime:
        engine.go(movetime: tc.whiteTime.inMilliseconds);
        break;
      case TimeControlType.fixedDepth:
        engine.go(depth: tc.depthLimit);
        break;
    }
  }

  void _startClock() {
    _stopClock();
    _lastClockTick = DateTime.now();
    _clockTimer = Timer.periodic(const Duration(milliseconds: 100), (_) {
      _tickClock();
    });
  }

  void _stopClock() {
    _clockTimer?.cancel();
    _clockTimer = null;
  }

  void _tickClock() {
    final now = DateTime.now();
    final elapsed = now.difference(_lastClockTick!).inMilliseconds;
    _lastClockTick = now;

    if (game.gameOver) {
      _stopClock();
      return;
    }

    final whiteToMove = game.turn == 0;
    if (whiteToMove) {
      final current = ref.read(whiteClockProvider);
      final newTime = current - elapsed;
      ref.read(whiteClockProvider.notifier).state = newTime < 0 ? 0 : newTime;
    } else {
      final current = ref.read(blackClockProvider);
      final newTime = current - elapsed;
      ref.read(blackClockProvider.notifier).state = newTime < 0 ? 0 : newTime;
    }
  }

  void _addIncrement() {
    final tc = ref.read(timeControlProvider);
    if (tc.type != TimeControlType.gameTime) return;

    final whiteJustMoved = game.turn == 1;
    if (whiteJustMoved) {
      ref.read(whiteClockProvider.notifier).state =
          ref.read(whiteClockProvider) + tc.increment.inMilliseconds;
    } else {
      ref.read(blackClockProvider.notifier).state =
          ref.read(blackClockProvider) + tc.increment.inMilliseconds;
    }
  }

  String _squaresMoveToUci(sq.Move move) {
    final size = game.size;
    final from = _squareToAlgebraic(move.from, size);
    final to = _squareToAlgebraic(move.to, size);
    final promo = move.promo?.toLowerCase() ?? '';
    return '$from$to$promo';
  }

  String _squareToAlgebraic(int square, bp.BoardSize size) {
    final file = square % size.h;
    final rank = square ~/ size.h;
    return '${String.fromCharCode(97 + file)}${rank + 1}';
  }
}

final gameProvider = StateNotifierProvider<GameNotifier, GameState>((ref) {
  return GameNotifier(ref);
});

/// Derived provider for the SquaresState used by the board widget.
final squaresStateProvider = Provider<SquaresState>((ref) {
  ref.watch(gameProvider);
  return ref.read(gameProvider.notifier).squaresState;
});
