// LeafGUI — a chess GUI for the Leaf chess engine.
// Copyright (C) 2026 Daniel C. Homan
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.  See the LICENSE file at the root of this repository.

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
