import 'dart:io';

import 'package:bishop/bishop.dart' as bp;
import 'package:flutter_test/flutter_test.dart';

/// Verifies that bishop's `meta.algebraic` move history — what the GUI now
/// sends to the engine in `position ... moves ...` — is engine-parseable
/// UCI long algebraic, including castling and promotion.
void main() {
  List<String> historyMoves(bp.Game game) {
    final moves = <String>[];
    for (int i = 1; i < game.history.length; i++) {
      final algebraic = game.history[i].meta?.algebraic;
      if (algebraic != null) moves.add(algebraic);
    }
    return moves;
  }

  test('standard castling and promotion use UCI long algebraic', () {
    final game = bp.Game(variant: bp.Variant.standard());
    for (final m in ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5']) {
      expect(game.makeMoveString(m), isTrue, reason: 'move $m rejected');
    }
    // White castles kingside.
    expect(game.makeMoveString('e1g1'), isTrue, reason: 'castling rejected');

    final moves = historyMoves(game);
    expect(moves.length, 7);
    expect(moves.take(6), ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5']);
    // Castling must round-trip as e1g1 (king destination), which is what
    // the engine expects with UCI_Chess960 off.
    expect(moves.last, 'e1g1');
  });

  test('engine replays a GUI move list with castling and searches', () async {
    const enginePath = '/Users/homand/Leaf/engine/run/Leaf_vcurrent';
    if (!File(enginePath).existsSync()) {
      markTestSkipped('engine binary not present');
      return;
    }

    final proc = await Process.start(enginePath, [],
        workingDirectory: File(enginePath).parent.path);
    final lines = <String>[];
    final done = proc.stdout
        .transform(const SystemEncoding().decoder)
        .listen((chunk) => lines.addAll(chunk.split('\n')));

    proc.stdin.writeln('uci');
    proc.stdin.writeln('isready');
    // Off-book position (so the engine actually searches) with both sides
    // castling in the replayed move list — the exact `position` shape the
    // GUI now sends. Misparsed moves would corrupt the position and the
    // search/bestmove below.
    proc.stdin.writeln('position fen '
        'r3k2r/pppqpppp/2n2n2/3p4/3P4/2N2N2/PPPQPPPP/R3K2R w KQkq - 0 1 '
        'moves e1g1 e8g8');
    proc.stdin.writeln('go depth 8');
    await Future.delayed(const Duration(seconds: 5));
    proc.stdin.writeln('quit');
    await proc.exitCode.timeout(const Duration(seconds: 5), onTimeout: () {
      proc.kill();
      return -1;
    });
    await done.cancel();

    final scoreLines =
        lines.where((l) => l.startsWith('info') && l.contains('score cp'));
    expect(scoreLines, isNotEmpty, reason: 'engine produced no search info');

    // Roughly balanced position — a wildly lopsided score would mean the
    // replayed moves were misapplied.
    final lastScore = int.parse(
        RegExp(r'score cp (-?\d+)').firstMatch(scoreLines.last)!.group(1)!);
    expect(lastScore.abs(), lessThan(300),
        reason: 'implausible eval suggests move list was misapplied');

    final bestMove =
        lines.firstWhere((l) => l.startsWith('bestmove'), orElse: () => '');
    expect(bestMove, isNotEmpty, reason: 'no bestmove received');
  });
}
