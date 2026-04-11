import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../models/engine_config.dart';
import '../../providers/engine_provider.dart';
import '../../providers/game_provider.dart';
import '../../services/uci_engine.dart';
import 'engine_picker.dart';

class EngineSettingsDialog extends ConsumerStatefulWidget {
  const EngineSettingsDialog({super.key});

  @override
  ConsumerState<EngineSettingsDialog> createState() =>
      _EngineSettingsDialogState();
}

class _EngineSettingsDialogState extends ConsumerState<EngineSettingsDialog> {
  late String _enginePath;
  late int _hashMb;
  late int _threads;
  late int _skillLevel;
  late bool _ponder;

  @override
  void initState() {
    super.initState();
    final config = ref.read(engineConfigProvider);
    _enginePath = config.path;
    _hashMb = config.hashMb;
    _threads = config.threads;
    _skillLevel = ref.read(skillLevelProvider);
    _ponder = ref.read(ponderEnabledProvider);
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: const Color(0xFF242424),
      title: const Text(
        'Engine Settings',
        style: TextStyle(fontFamily: 'monospace', color: Colors.white),
      ),
      content: SizedBox(
        width: 380,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _sectionLabel('Engine'),
            const SizedBox(height: 4),
            EnginePicker(
              currentPath: _enginePath,
              onChanged: (path) => setState(() => _enginePath = path),
            ),
            const SizedBox(height: 16),

            _sectionLabel('Hash (MB)'),
            const SizedBox(height: 4),
            _numberInput(_hashMb, (v) => setState(() => _hashMb = v),
                min: 1, max: 4096, step: _hashStep()),
            const SizedBox(height: 16),

            _sectionLabel('Threads'),
            const SizedBox(height: 4),
            _numberInput(_threads, (v) => setState(() => _threads = v),
                min: 1, max: 32),
            const SizedBox(height: 16),

            _sectionLabel('Engine Skill'),
            const SizedBox(height: 4),
            Row(
              children: [
                Expanded(
                  child: Slider(
                    value: _skillLevel.toDouble(),
                    min: 1,
                    max: 100,
                    divisions: 99,
                    activeColor: _skillLevel < 100
                        ? const Color(0xFFCC8844)
                        : Colors.white54,
                    inactiveColor: const Color(0xFF333333),
                    onChanged: (v) =>
                        setState(() => _skillLevel = v.round()),
                  ),
                ),
                SizedBox(
                  width: 44,
                  child: Text(
                    '$_skillLevel%',
                    textAlign: TextAlign.right,
                    style: TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 12,
                      color: _skillLevel < 100
                          ? const Color(0xFFCC8844)
                          : Colors.white,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),

            Row(
              children: [
                Checkbox(
                  value: _ponder,
                  onChanged: (v) => setState(() => _ponder = v ?? true),
                  fillColor:
                      WidgetStateProperty.all(const Color(0xFF555555)),
                  checkColor: Colors.white,
                ),
                const Text(
                  'Ponder',
                  style: TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 12,
                      color: Colors.white70),
                ),
              ],
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child:
              const Text('Cancel', style: TextStyle(fontFamily: 'monospace')),
        ),
        TextButton(
          onPressed: _apply,
          child: const Text('Apply',
              style: TextStyle(fontFamily: 'monospace', color: Colors.white)),
        ),
      ],
    );
  }

  int _hashStep() {
    if (_hashMb >= 1024) return 256;
    if (_hashMb >= 256) return 64;
    if (_hashMb >= 64) return 16;
    return 1;
  }

  Widget _sectionLabel(String text) {
    return Text(
      text,
      style: const TextStyle(
        fontFamily: 'monospace',
        fontSize: 12,
        color: Color(0xFF888888),
      ),
    );
  }

  Widget _numberInput(int value, ValueChanged<int> onChanged,
      {int min = 0, int max = 999, int step = 1}) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        IconButton(
          icon: const Icon(Icons.remove, size: 14),
          onPressed: value > min
              ? () => onChanged((value - step).clamp(min, max))
              : null,
          iconSize: 14,
          constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
          color: Colors.white54,
        ),
        SizedBox(
          width: 52,
          child: Text(
            '$value',
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontFamily: 'monospace',
              fontSize: 13,
              color: Colors.white,
            ),
          ),
        ),
        IconButton(
          icon: const Icon(Icons.add, size: 14),
          onPressed: value < max
              ? () => onChanged((value + step).clamp(min, max))
              : null,
          iconSize: 14,
          constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
          color: Colors.white54,
        ),
      ],
    );
  }

  void _apply() async {
    final oldConfig = ref.read(engineConfigProvider);
    final pathChanged = _enginePath != oldConfig.path;

    // Update config.
    ref.read(engineConfigProvider.notifier).state = EngineConfig(
      path: _enginePath,
      hashMb: _hashMb,
      threads: _threads,
    );

    if (pathChanged) {
      // Restart engine with new path.
      await ref.read(engineProvider.notifier).startEngine();
    } else {
      // Send options to running engine.
      final engine = ref.read(engineProvider);
      if (engine != null && engine.isRunning) {
        final wasSearching = engine.state != UciEngineState.idle;
        if (wasSearching) {
          engine.stopSearch();
          await engine.isReady();
        }
        engine.setOption('Hash', _hashMb);
        engine.setOption('Threads', _threads);
        await engine.isReady();
      }
    }

    // Update skill and ponder.
    ref.read(engineProvider.notifier).setSkillLevel(_skillLevel);
    ref.read(ponderEnabledProvider.notifier).state = _ponder;

    if (mounted) Navigator.pop(context);
  }
}
