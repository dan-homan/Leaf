import 'package:bishop/bishop.dart' as bp;

/// Converts a list of UCI long-algebraic PV moves into SAN with move numbers.
///
/// Example: ["e2e4", "e7e5", "g1f3"] from a white-to-move position becomes
/// "1. e4 e5 2. Nf3"
String formatPv(List<String> pvMoves, String fen, {bp.Variant? variant, bool chess960 = false}) {
  if (pvMoves.isEmpty) return '';

  try {
    final game = bp.Game(
      variant: variant ?? (chess960 ? bp.Variant.chess960() : bp.Variant.standard()),
      fen: fen,
    );

    final buffer = StringBuffer();
    for (int i = 0; i < pvMoves.length; i++) {
      final isWhiteMove = (game.turn == bp.Bishop.white);
      final moveNum = game.state.fullMoves;

      // Get SAN before making the move.
      // makeMoveString returns true/false; we need the SAN from the
      // generated legal moves first.
      final legalMoves = game.generateLegalMoves();
      final uciMove = pvMoves[i];
      String san = uciMove; // fallback to UCI notation

      // Find the matching legal move to get its SAN.
      for (final lm in legalMoves) {
        final algebraic = game.toAlgebraic(lm);
        if (algebraic == uciMove) {
          // Get the formatted (SAN) version by making the move.
          final success = game.makeMove(lm);
          if (success) {
            final meta = game.history.last.meta;
            san = meta?.moveMeta?.formatted ?? uciMove;
          }
          break;
        }
      }

      // If we didn't find a match through legal moves, try makeMoveString.
      if (san == uciMove) {
        final success = game.makeMoveString(uciMove);
        if (success) {
          final meta = game.history.last.meta;
          san = meta?.moveMeta?.formatted ?? uciMove;
        } else {
          // Invalid move in PV — stop here.
          break;
        }
      }

      if (isWhiteMove) {
        if (i > 0) buffer.write(' ');
        buffer.write('$moveNum. $san');
      } else {
        if (i == 0) {
          // PV starts with black's move.
          buffer.write('$moveNum... $san');
        } else {
          buffer.write(' $san');
        }
      }
    }

    return buffer.toString();
  } catch (_) {
    // If anything goes wrong, fall back to raw UCI moves.
    return pvMoves.join(' ');
  }
}
