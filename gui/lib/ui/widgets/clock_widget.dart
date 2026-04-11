import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../providers/game_provider.dart';
import '../../models/game_state.dart';

class ClockWidget extends ConsumerWidget {
  const ClockWidget({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final tc = ref.watch(timeControlProvider);
    if (tc.type != TimeControlType.gameTime) {
      return const SizedBox.shrink();
    }

    final whiteMs = ref.watch(whiteClockProvider);
    final blackMs = ref.watch(blackClockProvider);
    final playerColor = ref.watch(playerColorProvider);

    // Show opponent's clock on top, player's clock on bottom.
    final topMs = playerColor == 0 ? blackMs : whiteMs;
    final bottomMs = playerColor == 0 ? whiteMs : blackMs;
    final topLabel = playerColor == 0 ? 'Black' : 'White';
    final bottomLabel = playerColor == 0 ? 'White' : 'Black';

    return Column(
      children: [
        _ClockDisplay(label: topLabel, milliseconds: topMs),
        const SizedBox(height: 8),
        _ClockDisplay(label: bottomLabel, milliseconds: bottomMs),
      ],
    );
  }
}

class _ClockDisplay extends StatelessWidget {
  final String label;
  final int milliseconds;

  const _ClockDisplay({required this.label, required this.milliseconds});

  @override
  Widget build(BuildContext context) {
    final isLow = milliseconds < 30000; // under 30 seconds
    final timeStr = _formatTime(milliseconds);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: isLow ? const Color(0xFF442222) : const Color(0xFF242424),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: const Color(0xFF333333)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontFamily: 'monospace',
              fontSize: 11,
              color: Color(0xFF888888),
            ),
          ),
          const SizedBox(width: 12),
          Text(
            timeStr,
            style: TextStyle(
              fontFamily: 'monospace',
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: isLow ? const Color(0xFFCC4444) : Colors.white,
            ),
          ),
        ],
      ),
    );
  }

  String _formatTime(int ms) {
    if (ms <= 0) return '0:00';
    final totalSeconds = ms ~/ 1000;
    final minutes = totalSeconds ~/ 60;
    final seconds = totalSeconds % 60;
    if (minutes >= 60) {
      final hours = minutes ~/ 60;
      final mins = minutes % 60;
      return '$hours:${mins.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';
    }
    return '$minutes:${seconds.toString().padLeft(2, '0')}';
  }
}
