import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_riverpod/legacy.dart';
import '../../models/game_state.dart';
import '../../providers/engine_provider.dart';
import '../../providers/game_provider.dart';
import '../../services/pv_formatter.dart';
import '../../services/uci_engine.dart';

class EngineOutputPanel extends ConsumerWidget {
  const EngineOutputPanel({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final mode = ref.watch(gameModeProvider);
    final isEve = mode == GameMode.engineVsEngine;

    final engine1Name = ref.watch(engineNameProvider);
    final engine2Name = ref.watch(engine2NameProvider);

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF242424),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF333333)),
      ),
      child: isEve
          ? Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _EngineSection(
                  label: engine1Name ?? 'Engine 1',
                  sideLabel: 'White',
                  infoProvider: engineInfoProvider,
                  stateProvider: engineStateProvider,
                  skillProvider: skillLevelProvider,
                ),
                const Divider(color: Color(0xFF333333), height: 16),
                _EngineSection(
                  label: engine2Name ?? 'Engine 2',
                  sideLabel: 'Black',
                  infoProvider: engine2InfoProvider,
                  stateProvider: engine2StateProvider,
                  skillProvider: engine2SkillLevelProvider,
                ),
              ],
            )
          : _EngineSection(
              label: engine1Name ?? 'Engine',
              infoProvider: engineInfoProvider,
              stateProvider: engineStateProvider,
              skillProvider: skillLevelProvider,
            ),
    );
  }

}

class _EngineSection extends ConsumerWidget {
  final String label;
  final String? sideLabel;
  final StateProvider<UciInfo> infoProvider;
  final StateProvider<UciEngineState> stateProvider;
  final StateProvider<int> skillProvider;

  const _EngineSection({
    required this.label,
    this.sideLabel,
    required this.infoProvider,
    required this.stateProvider,
    required this.skillProvider,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final info = ref.watch(infoProvider);
    final engineState = ref.watch(stateProvider);
    final skill = ref.watch(skillProvider);

    String stateLabel;
    if (engineState == UciEngineState.thinking) {
      stateLabel = 'Thinking...';
    } else if (engineState == UciEngineState.pondering) {
      stateLabel = 'Pondering...';
    } else {
      stateLabel = 'Idle';
    }

    final pvDisplay = _formatPvDisplay(info, ref);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Flexible(
              child: Text(
                sideLabel != null ? '$label ($sideLabel)' : label,
                style: const TextStyle(
                  fontFamily: 'monospace',
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                  fontSize: 13,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            if (skill < 100)
              Padding(
                padding: const EdgeInsets.only(left: 8),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: const Color(0xFF443322),
                    borderRadius: BorderRadius.circular(4),
                    border: Border.all(color: const Color(0xFFCC8844)),
                  ),
                  child: Text(
                    'Skill $skill%',
                    style: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 10,
                      color: Color(0xFFCC8844),
                    ),
                  ),
                ),
              ),
            const Spacer(),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: engineState == UciEngineState.thinking
                    ? const Color(0xFF446644)
                    : engineState == UciEngineState.pondering
                        ? const Color(0xFF444466)
                        : const Color(0xFF333333),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                stateLabel,
                style: const TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 11,
                  color: Colors.white70,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        Wrap(
          spacing: 16,
          runSpacing: 4,
          children: [
            _InfoChip(label: 'D', value: _depthString(info)),
            _InfoChip(label: 'Eval', value: info.scoreString),
            _InfoChip(label: 'N', value: info.nodesString),
            _InfoChip(label: 'NPS', value: info.npsString),
          ],
        ),
        const SizedBox(height: 6),
        Text(
          pvDisplay,
          style: const TextStyle(
            fontFamily: 'monospace',
            fontSize: 12,
            color: Color(0xFF999999),
          ),
          maxLines: 3,
          overflow: TextOverflow.ellipsis,
        ),
      ],
    );
  }

  String _formatPvDisplay(UciInfo info, WidgetRef ref) {
    if (info.pv == null || info.pv!.isEmpty) return '';
    final fen = ref.read(gameProvider.notifier).fen;
    final chess960 = ref.read(chess960Provider);
    return formatPv(info.pv!, fen, chess960: chess960);
  }

  String _depthString(UciInfo info) {
    if (info.depth == null) return '';
    if (info.selectiveDepth != null) {
      return '${info.depth}/${info.selectiveDepth}';
    }
    return '${info.depth}';
  }
}

class _InfoChip extends StatelessWidget {
  final String label;
  final String value;

  const _InfoChip({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '$label ',
          style: const TextStyle(
            fontFamily: 'monospace',
            fontSize: 11,
            color: Color(0xFF666666),
          ),
        ),
        Text(
          value,
          style: const TextStyle(
            fontFamily: 'monospace',
            fontSize: 11,
            color: Colors.white,
          ),
        ),
      ],
    );
  }
}

