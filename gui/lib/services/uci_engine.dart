import 'dart:async';
import 'dart:convert';
import 'dart:io';

/// Parsed UCI info line from the engine.
class UciInfo {
  final int? depth;
  final int? selectiveDepth;
  final int? score;
  final bool? isMate;
  final int? time;
  final int? nodes;
  final int? nps;
  final List<String>? pv;
  final int? multiPvIndex;

  const UciInfo({
    this.depth,
    this.selectiveDepth,
    this.score,
    this.isMate,
    this.time,
    this.nodes,
    this.nps,
    this.pv,
    this.multiPvIndex,
  });

  String get scoreString {
    if (score == null) return '';
    if (isMate == true) return 'M$score';
    return (score! / 100).toStringAsFixed(2);
  }

  String get pvString => pv?.join(' ') ?? '';

  String get nodesString {
    if (nodes == null) return '';
    if (nodes! >= 1000000) return '${(nodes! / 1000000).toStringAsFixed(1)}M';
    if (nodes! >= 1000) return '${(nodes! / 1000).toStringAsFixed(1)}K';
    return nodes.toString();
  }

  String get npsString {
    if (nps == null) return '';
    if (nps! >= 1000000) return '${(nps! / 1000000).toStringAsFixed(1)}M';
    if (nps! >= 1000) return '${(nps! / 1000).toStringAsFixed(1)}K';
    return nps.toString();
  }
}

/// Represents the current state of the UCI engine.
enum UciEngineState { idle, thinking, pondering }

/// UCI engine process manager. Handles spawning, communication, and parsing.
class UciEngine {
  final String path;
  Process? _process;
  UciEngineState _state = UciEngineState.idle;

  final _infoController = StreamController<UciInfo>.broadcast();
  final _bestMoveController = StreamController<String>.broadcast();
  final _stateController = StreamController<UciEngineState>.broadcast();
  final _rawOutputController = StreamController<String>.broadcast();

  /// Stream of parsed info lines during search.
  Stream<UciInfo> get infoStream => _infoController.stream;

  /// Stream of bestmove responses.
  Stream<String> get bestMoveStream => _bestMoveController.stream;

  /// Stream of engine state changes.
  Stream<UciEngineState> get stateStream => _stateController.stream;

  /// Stream of raw output lines (for debugging).
  Stream<String> get rawOutputStream => _rawOutputController.stream;

  UciEngineState get state => _state;
  bool get isRunning => _process != null;

  String? engineName;
  String? engineAuthor;

  // The ponder move the engine is currently pondering on.
  String? _ponderMove;

  UciEngine({required this.path});

  /// Start the engine process and initialize UCI handshake.
  Future<bool> start() async {
    if (_process != null) return true;

    try {
      // Run in the engine's directory so it can find data files (book, NNUE, etc.)
      final engineDir = File(path).parent.path;
      _process = await Process.start(path, [], workingDirectory: engineDir);
    } catch (e) {
      return false;
    }

    _process!.stdout
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .listen(_handleLine);

    _process!.stderr
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .listen((line) {
      _rawOutputController.add('[stderr] $line');
    });

    _process!.exitCode.then((_) {
      _process = null;
      _setState(UciEngineState.idle);
    });

    // Send UCI init and wait for uciok.
    _send('uci');
    await _waitFor('uciok');
    return true;
  }

  /// Stop the engine process.
  Future<void> stop() async {
    if (_process == null) return;
    _send('quit');
    // Give it a moment to exit gracefully.
    await Future.delayed(const Duration(milliseconds: 200));
    _process?.kill();
    _process = null;
    _setState(UciEngineState.idle);
  }

  /// Set a UCI option.
  void setOption(String name, dynamic value) {
    _send('setoption name $name value $value');
  }

  /// Send 'isready' and wait for 'readyok'.
  Future<void> isReady() async {
    _send('isready');
    await _waitFor('readyok');
  }

  /// Send a new game signal.
  Future<void> newGame() async {
    _send('ucinewgame');
    await isReady();
  }

  /// Set the position. [fen] is the starting position (or 'startpos'),
  /// [moves] is the list of moves in long algebraic notation.
  void setPosition({String? fen, List<String> moves = const []}) {
    final posStr = fen != null ? 'fen $fen' : 'startpos';
    final movesStr = moves.isNotEmpty ? ' moves ${moves.join(' ')}' : '';
    _send('position $posStr$movesStr');
  }

  /// Start searching with the given parameters.
  void go({
    int? wtime,
    int? btime,
    int? winc,
    int? binc,
    int? movetime,
    int? depth,
    bool infinite = false,
    bool ponder = false,
  }) {
    final parts = <String>['go'];
    if (ponder) parts.add('ponder');
    if (infinite) parts.add('infinite');
    if (wtime != null) parts.addAll(['wtime', '$wtime']);
    if (btime != null) parts.addAll(['btime', '$btime']);
    if (winc != null) parts.addAll(['winc', '$winc']);
    if (binc != null) parts.addAll(['binc', '$binc']);
    if (movetime != null) parts.addAll(['movetime', '$movetime']);
    if (depth != null) parts.addAll(['depth', '$depth']);
    _setState(ponder ? UciEngineState.pondering : UciEngineState.thinking);
    _send(parts.join(' '));
  }

  /// Start pondering. Call after setting the position with the expected
  /// ponder move appended.
  void goPonder({
    int? wtime,
    int? btime,
    int? winc,
    int? binc,
    required String ponderMove,
  }) {
    _ponderMove = ponderMove;
    go(wtime: wtime, btime: btime, winc: winc, binc: binc, ponder: true);
  }

  /// Called when the opponent plays the move we were pondering on.
  /// Converts the ponder search into a real search.
  void ponderHit() {
    if (_state == UciEngineState.pondering) {
      _send('ponderhit');
      _setState(UciEngineState.thinking);
    }
  }

  /// Stop the current search (or ponder).
  void stopSearch() {
    _send('stop');
  }

  /// Send "isready" as a synchronization barrier without awaiting the response.
  /// The engine will process all pending work before handling commands that
  /// follow. Useful to ensure an engine is ready after bestmove before
  /// sending a new position + go.
  void sync() {
    _send('isready');
  }

  String? get ponderMove => _ponderMove;

  // --- Private implementation ---

  void _send(String command) {
    _rawOutputController.add('>> $command');
    _process?.stdin.writeln(command);
  }

  void _setState(UciEngineState newState) {
    _state = newState;
    _stateController.add(newState);
  }

  void _handleLine(String line) {
    _rawOutputController.add(line);

    if (line.startsWith('info ')) {
      _handleInfo(line);
    } else if (line.startsWith('bestmove ')) {
      _handleBestMove(line);
    } else if (line.startsWith('id name ')) {
      engineName = line.substring(8);
    } else if (line.startsWith('id author ')) {
      engineAuthor = line.substring(10);
    }
  }

  void _handleInfo(String line) {
    final tokens = line.split(' ');
    int? depth, seldepth, score, time, nodes, nps, multiPv;
    bool? isMate;
    List<String>? pv;

    for (int i = 1; i < tokens.length; i++) {
      switch (tokens[i]) {
        case 'depth':
          depth = int.tryParse(tokens[++i]);
          break;
        case 'seldepth':
          seldepth = int.tryParse(tokens[++i]);
          break;
        case 'score':
          i++;
          if (tokens[i] == 'cp') {
            score = int.tryParse(tokens[++i]);
            isMate = false;
          } else if (tokens[i] == 'mate') {
            score = int.tryParse(tokens[++i]);
            isMate = true;
          }
          break;
        case 'time':
          time = int.tryParse(tokens[++i]);
          break;
        case 'nodes':
          nodes = int.tryParse(tokens[++i]);
          break;
        case 'nps':
          nps = int.tryParse(tokens[++i]);
          break;
        case 'multipv':
          multiPv = int.tryParse(tokens[++i]);
          break;
        case 'pv':
          pv = tokens.sublist(i + 1);
          i = tokens.length; // end loop
          break;
      }
    }

    if (depth != null) {
      _infoController.add(UciInfo(
        depth: depth,
        selectiveDepth: seldepth,
        score: score,
        isMate: isMate,
        time: time,
        nodes: nodes,
        nps: nps,
        pv: pv,
        multiPvIndex: multiPv,
      ));
    }
  }

  void _handleBestMove(String line) {
    final tokens = line.split(' ');
    final bestMove = tokens[1];

    // Extract ponder move if present.
    _ponderMove = null;
    if (tokens.length >= 4 && tokens[2] == 'ponder') {
      _ponderMove = tokens[3];
    }

    _setState(UciEngineState.idle);
    _bestMoveController.add(bestMove);
  }

  /// Wait for a specific response line from the engine.
  Future<void> _waitFor(String expected, {Duration timeout = const Duration(seconds: 10)}) {
    final completer = Completer<void>();
    late StreamSubscription<String> sub;

    sub = rawOutputStream.listen((line) {
      if (line == expected) {
        sub.cancel();
        if (!completer.isCompleted) completer.complete();
      }
    });

    return completer.future.timeout(timeout, onTimeout: () {
      sub.cancel();
    });
  }

  /// Dispose all stream controllers.
  void dispose() {
    stop();
    _infoController.close();
    _bestMoveController.close();
    _stateController.close();
    _rawOutputController.close();
  }
}
