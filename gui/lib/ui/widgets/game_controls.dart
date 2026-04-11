import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:bishop/bishop.dart' as bp;
import '../../models/game_state.dart';
import '../../providers/game_provider.dart';
import 'engine_settings_dialog.dart';
import 'new_game_dialog.dart';

class GameControlsPanel extends ConsumerWidget {
  const GameControlsPanel({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final gameState = ref.watch(gameProvider);
    final game = gameState.game;
    final mode = ref.watch(gameModeProvider);
    final paused = ref.watch(matchPausedProvider);

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF242424),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF333333)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Text(
                'Game',
                style: TextStyle(
                  fontFamily: 'monospace',
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                  fontSize: 14,
                ),
              ),
              if (mode == GameMode.analysis) ...[
                const SizedBox(width: 8),
                _ModeBadge(label: 'Analysis', color: const Color(0xFF44AAAA)),
              ],
              if (mode == GameMode.engineVsEngine) ...[
                const SizedBox(width: 8),
                _ModeBadge(
                  label: paused ? 'Paused' : 'Engine Match',
                  color: paused
                      ? const Color(0xFFCC8844)
                      : const Color(0xFF44AA44),
                ),
              ],
            ],
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _ControlButton(
                icon: Icons.add,
                label: 'New',
                onPressed: () => _showNewGameDialog(context, ref),
              ),
              if (mode == GameMode.engineVsEngine) ...[
                _ControlButton(
                  icon: paused ? Icons.play_arrow : Icons.pause,
                  label: paused ? 'Resume' : 'Pause',
                  onPressed: game.gameOver
                      ? null
                      : () {
                          if (paused) {
                            ref.read(gameProvider.notifier).resumeMatch();
                          } else {
                            ref.read(gameProvider.notifier).pauseMatch();
                          }
                        },
                ),
              ],
              _ControlButton(
                icon: Icons.undo,
                label: 'Undo',
                onPressed: game.canUndo
                    ? () => ref.read(gameProvider.notifier).undoMove()
                    : null,
              ),
              if (mode == GameMode.play) ...[
                _ControlButton(
                  icon: Icons.swap_horiz,
                  label: 'Swap',
                  onPressed: !game.gameOver
                      ? () => ref.read(gameProvider.notifier).swapSides()
                      : null,
                ),
              ],
              _ControlButton(
                icon: Icons.analytics_outlined,
                label: mode == GameMode.analysis ? 'Play' : 'Analyze',
                onPressed: () {
                  if (mode == GameMode.analysis) {
                    ref.read(gameProvider.notifier).exitAnalysis();
                  } else {
                    ref.read(gameProvider.notifier).enterAnalysis();
                  }
                },
              ),
              _ControlButton(
                icon: Icons.settings_outlined,
                label: 'Engine',
                onPressed: () => _showEngineSettings(context),
              ),
              _ControlButton(
                icon: Icons.content_copy,
                label: 'FEN',
                onPressed: () => _copyFen(context, ref),
              ),
              _ControlButton(
                icon: Icons.content_paste,
                label: 'Load',
                onPressed: () => _loadFen(context, ref),
              ),
            ],
          ),
          if (game.gameOver) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: const Color(0xFF333333),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                _gameResultString(game),
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  void _showNewGameDialog(BuildContext context, WidgetRef ref) {
    showDialog(
      context: context,
      builder: (context) => const NewGameDialog(),
    );
  }

  void _showEngineSettings(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => const EngineSettingsDialog(),
    );
  }

  void _copyFen(BuildContext context, WidgetRef ref) {
    final fen = ref.read(gameProvider.notifier).fen;
    Clipboard.setData(ClipboardData(text: fen));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('FEN copied to clipboard',
            style: TextStyle(fontFamily: 'monospace')),
        duration: Duration(seconds: 2),
      ),
    );
  }

  void _loadFen(BuildContext context, WidgetRef ref) {
    final controller = TextEditingController();
    final chess960 = ref.read(chess960Provider);

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF242424),
        title: const Text('Load FEN',
            style: TextStyle(fontFamily: 'monospace', color: Colors.white)),
        content: TextField(
          controller: controller,
          style: const TextStyle(fontFamily: 'monospace', color: Colors.white),
          decoration: const InputDecoration(
            hintText: 'Paste FEN string...',
            hintStyle: TextStyle(color: Color(0xFF666666)),
            enabledBorder: UnderlineInputBorder(
                borderSide: BorderSide(color: Color(0xFF444444))),
            focusedBorder: UnderlineInputBorder(
                borderSide: BorderSide(color: Colors.white)),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel',
                style: TextStyle(fontFamily: 'monospace')),
          ),
          TextButton(
            onPressed: () {
              final fen = controller.text.trim();
              if (fen.isNotEmpty) {
                ref
                    .read(gameProvider.notifier)
                    .loadFen(fen, chess960: chess960);
                Navigator.pop(context);
              }
            },
            child: const Text('Load',
                style: TextStyle(fontFamily: 'monospace')),
          ),
        ],
      ),
    );
  }

  String _gameResultString(bp.Game game) {
    if (game.checkmate) return 'Checkmate!';
    if (game.stalemate) return 'Stalemate';
    if (game.gameOver) return 'Game Over';
    return '';
  }
}

class _ModeBadge extends StatelessWidget {
  final String label;
  final Color color;

  const _ModeBadge({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontFamily: 'monospace',
          fontSize: 10,
          color: color,
        ),
      ),
    );
  }
}

class _ControlButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onPressed;

  const _ControlButton({
    required this.icon,
    required this.label,
    this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 36,
      child: TextButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, size: 16),
        label: Text(label,
            style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
        style: TextButton.styleFrom(
          foregroundColor: Colors.white,
          disabledForegroundColor: const Color(0xFF555555),
          padding: const EdgeInsets.symmetric(horizontal: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(4),
            side: const BorderSide(color: Color(0xFF444444)),
          ),
        ),
      ),
    );
  }
}
