import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import '../../models/game_state.dart';
import '../../providers/engine_provider.dart';
import '../../providers/game_provider.dart';
import '../widgets/board_widget.dart';
import '../widgets/engine_output.dart';
import '../widgets/game_controls.dart';
import '../widgets/move_list.dart';
import '../widgets/new_game_dialog.dart';

class HomeScreen extends ConsumerStatefulWidget {
  const HomeScreen({super.key});

  @override
  ConsumerState<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends ConsumerState<HomeScreen> {
  bool _engineStarted = false;
  String? _engineError;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _tryStartEngine());
  }

  Future<void> _tryStartEngine() async {
    final success = await ref.read(engineProvider.notifier).startEngine();
    if (success) {
      ref.read(gameProvider.notifier).startClockIfNeeded();
    }
    setState(() {
      _engineStarted = success;
      if (!success) {
        _engineError = 'Could not start engine. Check engine path in config.';
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            SvgPicture.asset(
              'assets/leaf.svg',
              height: 56,
            ),
            const SizedBox(width: 10),
            const Text(
              'LeafGUI',
              style: TextStyle(
                fontFamily: 'monospace',
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
            ),
          ],
        ),
        actions: [
          if (_engineError != null)
            Padding(
              padding: const EdgeInsets.only(right: 16),
              child: Tooltip(
                message: _engineError!,
                child: const Icon(Icons.warning_amber,
                    color: Color(0xFFCC4444), size: 20),
              ),
            ),
          if (_engineStarted)
            Padding(
              padding: const EdgeInsets.only(right: 16),
              child: Tooltip(
                message: 'Engine connected',
                child: Icon(Icons.check_circle_outline,
                    color: Colors.green.shade400, size: 20),
              ),
            ),
          IconButton(
            icon: const Icon(Icons.add, size: 20),
            tooltip: 'New Game',
            onPressed: () {
              showDialog(
                context: context,
                builder: (context) => const NewGameDialog(),
              );
            },
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: LayoutBuilder(
          builder: (context, constraints) {
            return Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Left side: clock + board + clock
                Column(
                  children: [
                    _ClockDisplay(isTop: true),
                    const SizedBox(height: 8),
                    SizedBox(
                      width: _boardSize(constraints),
                      height: _boardSize(constraints),
                      child: const ChessBoardWidget(),
                    ),
                    const SizedBox(height: 8),
                    _ClockDisplay(isTop: false),
                  ],
                ),
                const SizedBox(width: 16),
                // Right side: controls, engine output, move list
                Expanded(
                  child: Column(
                    children: [
                      const GameControlsPanel(),
                      const SizedBox(height: 12),
                      const EngineOutputPanel(),
                      const SizedBox(height: 12),
                      const Expanded(child: MoveListPanel()),
                    ],
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  double _boardSize(BoxConstraints constraints) {
    final maxHeight = constraints.maxHeight - 100;
    final maxWidth = constraints.maxWidth * 0.6;
    return maxHeight < maxWidth ? maxHeight : maxWidth;
  }
}

class _ClockDisplay extends ConsumerWidget {
  final bool isTop;

  const _ClockDisplay({required this.isTop});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final playerColor = ref.watch(playerColorProvider);
    final gameMode = ref.watch(gameModeProvider);
    final tc = ref.watch(timeControlProvider);
    final hasClock = tc.type == TimeControlType.gameTime;

    // Top = opponent, bottom = player.
    final bool isWhiteSide = isTop
        ? playerColor != 0  // top is white when player is black
        : playerColor == 0; // bottom is white when player is white

    // Engine name for this side.
    String? engineName;
    if (gameMode == GameMode.engineVsEngine) {
      // Engine 1 = white, Engine 2 = black.
      engineName = isWhiteSide
          ? ref.watch(engineNameProvider)
          : ref.watch(engine2NameProvider);
    } else if (gameMode == GameMode.play) {
      // Engine plays the opponent side (top).
      if (isTop) {
        engineName = ref.watch(engineNameProvider);
      }
    }

    final hasContent = hasClock || engineName != null;
    if (!hasContent) return const SizedBox.shrink();

    final String label = isWhiteSide ? 'White' : 'Black';

    // Clock values.
    int ms = 0;
    bool isLow = false;
    if (hasClock) {
      if (isWhiteSide) {
        ms = ref.watch(whiteClockProvider);
      } else {
        ms = ref.watch(blackClockProvider);
      }
      isLow = ms < 30000;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      decoration: BoxDecoration(
        color: hasClock && isLow
            ? const Color(0xFF442222)
            : const Color(0xFF242424),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: const Color(0xFF333333)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (engineName != null) ...[
            Text(
              engineName,
              style: const TextStyle(
                fontFamily: 'monospace',
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(width: 8),
            Text(
              '($label)',
              style: const TextStyle(
                fontFamily: 'monospace',
                fontSize: 14,
                color: Color(0xFF666666),
              ),
            ),
          ] else
            Text(
              label,
              style: const TextStyle(
                fontFamily: 'monospace',
                fontSize: 13,
                color: Color(0xFF888888),
              ),
            ),
          if (hasClock) ...[
            const SizedBox(width: 12),
            Text(
              _formatTime(ms),
              style: TextStyle(
                fontFamily: 'monospace',
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: isLow ? const Color(0xFFCC4444) : Colors.white,
              ),
            ),
          ],
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
