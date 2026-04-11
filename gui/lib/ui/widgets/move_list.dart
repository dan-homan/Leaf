import 'package:bishop/bishop.dart' as bp;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../providers/game_provider.dart';

class MoveListPanel extends ConsumerWidget {
  const MoveListPanel({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final gameState = ref.watch(gameProvider);
    final game = gameState.game;
    final redoSans = ref.read(gameProvider.notifier).redoMoveSans;

    // Build move pairs including redo (future) moves.
    final movePairs = _buildMovePairs(game, redoSans);
    final currentPly = gameState.viewPly;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF242424),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF333333)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Text(
                'Moves',
                style: TextStyle(
                  fontFamily: 'monospace',
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                  fontSize: 14,
                ),
              ),
              if (gameState.isViewingHistory) ...[
                const SizedBox(width: 8),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: const Color(0xFF333344),
                    borderRadius: BorderRadius.circular(4),
                    border: Border.all(color: const Color(0xFF6666AA)),
                  ),
                  child: Text(
                    '$currentPly/${gameState.totalPlies}',
                    style: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 10,
                      color: Color(0xFF6666AA),
                    ),
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 8),
          // Navigation buttons.
          _NavigationBar(gameState: gameState),
          const SizedBox(height: 8),
          Expanded(
            child: movePairs.isEmpty
                ? const Center(
                    child: Text(
                      'No moves yet',
                      style: TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 12,
                        color: Color(0xFF666666),
                      ),
                    ),
                  )
                : Scrollbar(
                    child: ListView.builder(
                    itemCount: movePairs.length,
                    itemBuilder: (context, index) {
                      final pair = movePairs[index];
                      return Padding(
                        padding: const EdgeInsets.symmetric(vertical: 1),
                        child: Row(
                          children: [
                            SizedBox(
                              width: 36,
                              child: Text(
                                '${pair.moveNumber}.',
                                style: const TextStyle(
                                  fontFamily: 'monospace',
                                  fontSize: 12,
                                  color: Color(0xFF666666),
                                ),
                              ),
                            ),
                            _MoveCell(
                              san: pair.whiteMove,
                              ply: pair.whitePly,
                              isCurrentPly: pair.whitePly == currentPly,
                              isInFuture: pair.whitePly > currentPly,
                            ),
                            if (pair.blackMove != null)
                              _MoveCell(
                                san: pair.blackMove!,
                                ply: pair.blackPly!,
                                isCurrentPly: pair.blackPly == currentPly,
                                isInFuture: pair.blackPly! > currentPly,
                              ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
          ),
        ],
      ),
    );
  }

  List<_MovePair> _buildMovePairs(bp.Game game, List<String> redoSans) {
    final pairs = <_MovePair>[];
    final history = game.history;

    // Collect all SANs: from history (ply 1..n) + redo stack.
    final allSans = <String>[];
    for (int i = 1; i < history.length; i++) {
      final meta = history[i].meta;
      allSans.add(meta?.moveMeta?.formatted ?? '?');
    }
    allSans.addAll(redoSans);

    int moveNumber = 1;
    for (int i = 0; i < allSans.length; i++) {
      final ply = i + 1; // ply 1 = first move
      if (i % 2 == 0) {
        // White's move
        pairs.add(_MovePair(
            moveNumber: moveNumber, whiteMove: allSans[i], whitePly: ply));
      } else {
        // Black's move
        if (pairs.isNotEmpty) {
          pairs.last = _MovePair(
            moveNumber: pairs.last.moveNumber,
            whiteMove: pairs.last.whiteMove,
            whitePly: pairs.last.whitePly,
            blackMove: allSans[i],
            blackPly: ply,
          );
        }
        moveNumber++;
      }
    }

    return pairs;
  }
}

class _MoveCell extends ConsumerWidget {
  final String san;
  final int ply;
  final bool isCurrentPly;
  final bool isInFuture;

  const _MoveCell({
    required this.san,
    required this.ply,
    required this.isCurrentPly,
    required this.isInFuture,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return GestureDetector(
      onTap: () => ref.read(gameProvider.notifier).goToMove(ply),
      child: MouseRegion(
        cursor: SystemMouseCursors.click,
        child: Container(
          width: 70,
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
          decoration: isCurrentPly
              ? BoxDecoration(
                  color: const Color(0xFF334455),
                  borderRadius: BorderRadius.circular(3),
                )
              : null,
          child: Text(
            san,
            style: TextStyle(
              fontFamily: 'monospace',
              fontSize: 12,
              color: isCurrentPly
                  ? Colors.white
                  : isInFuture
                      ? const Color(0xFF555555)
                      : Colors.white,
              fontWeight: isCurrentPly ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        ),
      ),
    );
  }
}

class _NavigationBar extends ConsumerWidget {
  final GameState gameState;

  const _NavigationBar({required this.gameState});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final canGoBack = gameState.viewPly > 0;
    final canGoForward = gameState.isViewingHistory;

    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        _NavButton(
          icon: Icons.skip_previous,
          onPressed: canGoBack
              ? () => ref.read(gameProvider.notifier).goToStart()
              : null,
        ),
        _NavButton(
          icon: Icons.chevron_left,
          onPressed: canGoBack
              ? () => ref.read(gameProvider.notifier).goBack()
              : null,
        ),
        _NavButton(
          icon: Icons.chevron_right,
          onPressed: canGoForward
              ? () => ref.read(gameProvider.notifier).goForward()
              : null,
        ),
        _NavButton(
          icon: Icons.skip_next,
          onPressed: canGoForward
              ? () => ref.read(gameProvider.notifier).goToEnd()
              : null,
        ),
      ],
    );
  }
}

class _NavButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback? onPressed;

  const _NavButton({required this.icon, this.onPressed});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 36,
      height: 28,
      child: IconButton(
        icon: Icon(icon, size: 18),
        onPressed: onPressed,
        padding: EdgeInsets.zero,
        constraints: const BoxConstraints(minWidth: 36, minHeight: 28),
        color: Colors.white70,
        disabledColor: const Color(0xFF444444),
      ),
    );
  }
}

class _MovePair {
  final int moveNumber;
  final String whiteMove;
  final int whitePly;
  final String? blackMove;
  final int? blackPly;

  const _MovePair({
    required this.moveNumber,
    required this.whiteMove,
    required this.whitePly,
    this.blackMove,
    this.blackPly,
  });
}
