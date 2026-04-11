import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:squares/squares.dart';

import '../../providers/game_provider.dart';
import '../theme/board_theme.dart';

class ChessBoardWidget extends ConsumerWidget {
  const ChessBoardWidget({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final sqState = ref.watch(squaresStateProvider);

    return AspectRatio(
      aspectRatio: 1.0,
      child: BoardController(
        state: sqState.board,
        playState: sqState.state,
        pieceSet: PieceSet.merida(),
        theme: LeafBoardTheme.blackAndWhite,
        markerTheme: LeafBoardTheme.markers,
        moves: sqState.moves,
        onMove: (move) {
          ref.read(gameProvider.notifier).makeMove(move);
        },
        onPremove: (move) {
          ref.read(gameProvider.notifier).makeMove(move);
        },
        animatePieces: true,
        animationDuration: const Duration(milliseconds: 200),
        labelConfig: LabelConfig.standard,
        draggable: true,
        dragFeedbackSize: 2.0,
        dragFeedbackOffset: const Offset(0.0, -1.0),
      ),
    );
  }
}
