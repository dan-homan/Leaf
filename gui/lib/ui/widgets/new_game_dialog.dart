import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../models/engine_config.dart';
import '../../models/game_state.dart';
import '../../providers/engine_provider.dart';
import '../../providers/game_provider.dart';
import 'engine_picker.dart';

class NewGameDialog extends ConsumerStatefulWidget {
  const NewGameDialog({super.key});

  @override
  ConsumerState<NewGameDialog> createState() => _NewGameDialogState();
}

class _NewGameDialogState extends ConsumerState<NewGameDialog> {
  bool _engineVsEngine = false;
  PlayerColor _playerColor = PlayerColor.white;
  TimeControlType _timeControlType = TimeControlType.gameTime;
  bool _chess960 = false;
  bool _ponder = true;
  bool _specificPosition = false;

  // Game time + increment defaults
  int _minutes = 5;
  int _increment = 3;

  // Fixed time per move
  int _fixedSeconds = 5;

  // Fixed depth
  int _depth = 10;

  // Engine 1 settings
  String _engine1Path = '';
  int _skillLevel = 100;

  // Engine 2 settings
  String _engine2Path = '';
  int _engine2Skill = 100;
  int _engine2Hash = 256;
  int _engine2Threads = 1;

  // Chess960 position number
  final _positionController = TextEditingController();
  final _engine2PathController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _ponder = ref.read(ponderEnabledProvider);
    _chess960 = ref.read(chess960Provider);
    _engine1Path = ref.read(engineConfigProvider).path;
    _skillLevel = ref.read(skillLevelProvider);
    _engine2Path = EngineConfig.defaultEnginePath();
    _engine2PathController.text = _engine2Path;
    final e2Config = ref.read(engine2ConfigProvider);
    _engine2Hash = e2Config.hashMb;
    _engine2Threads = e2Config.threads;
    _engine2Skill = ref.read(engine2SkillLevelProvider);
  }

  @override
  void dispose() {
    _positionController.dispose();
    _engine2PathController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: const Color(0xFF242424),
      title: const Text(
        'New Game',
        style: TextStyle(fontFamily: 'monospace', color: Colors.white),
      ),
      content: SizedBox(
        width: 380,
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Game type
              _sectionLabel('Game Type'),
              const SizedBox(height: 4),
              Row(
                children: [
                  _choiceChip('Human vs Engine', !_engineVsEngine,
                      () => setState(() => _engineVsEngine = false)),
                  const SizedBox(width: 8),
                  _choiceChip('Engine vs Engine', _engineVsEngine,
                      () => setState(() => _engineVsEngine = true)),
                ],
              ),
              const SizedBox(height: 16),

              // Color selection (human vs engine only)
              if (!_engineVsEngine) ...[
                _sectionLabel('Play as'),
                const SizedBox(height: 4),
                Row(
                  children: [
                    _choiceChip('White', _playerColor == PlayerColor.white,
                        () => setState(() => _playerColor = PlayerColor.white)),
                    const SizedBox(width: 8),
                    _choiceChip('Black', _playerColor == PlayerColor.black,
                        () => setState(() => _playerColor = PlayerColor.black)),
                  ],
                ),
                const SizedBox(height: 16),
              ],

              // Variant
              _sectionLabel('Variant'),
              const SizedBox(height: 4),
              Row(
                children: [
                  _choiceChip('Standard', !_chess960,
                      () => setState(() => _chess960 = false)),
                  const SizedBox(width: 8),
                  _choiceChip('Chess960', _chess960,
                      () => setState(() => _chess960 = true)),
                ],
              ),
              if (_chess960) ...[
                const SizedBox(height: 8),
                Row(
                  children: [
                    Checkbox(
                      value: _specificPosition,
                      onChanged: (v) =>
                          setState(() => _specificPosition = v ?? false),
                      fillColor: WidgetStateProperty.all(
                          const Color(0xFF555555)),
                      checkColor: Colors.white,
                    ),
                    const Text(
                      'Specific position (0-959)',
                      style: TextStyle(
                          fontFamily: 'monospace',
                          fontSize: 12,
                          color: Colors.white70),
                    ),
                  ],
                ),
                if (_specificPosition)
                  Padding(
                    padding: const EdgeInsets.only(left: 40),
                    child: SizedBox(
                      width: 100,
                      child: TextField(
                        controller: _positionController,
                        style: const TextStyle(
                            fontFamily: 'monospace', color: Colors.white),
                        keyboardType: TextInputType.number,
                        inputFormatters: [
                          FilteringTextInputFormatter.digitsOnly,
                        ],
                        decoration: const InputDecoration(
                          hintText: '0-959',
                          hintStyle: TextStyle(color: Color(0xFF666666)),
                          isDense: true,
                          enabledBorder: UnderlineInputBorder(
                              borderSide:
                                  BorderSide(color: Color(0xFF444444))),
                          focusedBorder: UnderlineInputBorder(
                              borderSide: BorderSide(color: Colors.white)),
                        ),
                      ),
                    ),
                  ),
              ],
              const SizedBox(height: 16),

              // Time control
              _sectionLabel('Time Control'),
              const SizedBox(height: 4),
              Row(
                children: [
                  _choiceChip(
                      'Game Time',
                      _timeControlType == TimeControlType.gameTime,
                      () => setState(
                          () => _timeControlType = TimeControlType.gameTime)),
                  const SizedBox(width: 8),
                  _choiceChip(
                      'Per Move',
                      _timeControlType == TimeControlType.fixedTime,
                      () => setState(
                          () => _timeControlType = TimeControlType.fixedTime)),
                  const SizedBox(width: 8),
                  _choiceChip(
                      'Depth',
                      _timeControlType == TimeControlType.fixedDepth,
                      () => setState(
                          () => _timeControlType = TimeControlType.fixedDepth)),
                ],
              ),
              const SizedBox(height: 8),
              _buildTimeControlOptions(),
              const SizedBox(height: 16),

              // Engine 1
              _sectionLabel(_engineVsEngine ? 'Engine 1 (White)' : 'Engine'),
              const SizedBox(height: 4),
              EnginePicker(
                currentPath: _engine1Path,
                onChanged: (path) => setState(() => _engine1Path = path),
              ),
              const SizedBox(height: 8),
              _sectionLabel(_engineVsEngine ? 'Engine 1 Skill' : 'Engine Skill'),
              const SizedBox(height: 4),
              _buildSkillSlider(_skillLevel, (v) => setState(() => _skillLevel = v)),
              const SizedBox(height: 8),

              if (_engineVsEngine) ...[
                // Engine 2 settings
                _sectionLabel('Engine 2 (Black)'),
                const SizedBox(height: 4),
                EnginePicker(
                  currentPath: _engine2Path,
                  onChanged: (path) => setState(() {
                    _engine2Path = path;
                    _engine2PathController.text = path;
                  }),
                ),
                const SizedBox(height: 8),
                _sectionLabel('Engine 2 Skill'),
                const SizedBox(height: 4),
                _buildSkillSlider(_engine2Skill, (v) => setState(() => _engine2Skill = v)),
                const SizedBox(height: 8),
              ],

              // Ponder (human vs engine only)
              if (!_engineVsEngine)
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
                      'Ponder (think on your time)',
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
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child:
              const Text('Cancel', style: TextStyle(fontFamily: 'monospace')),
        ),
        TextButton(
          onPressed: _startGame,
          child: const Text('Start',
              style: TextStyle(fontFamily: 'monospace', color: Colors.white)),
        ),
      ],
    );
  }

  Widget _buildSkillSlider(int value, ValueChanged<int> onChanged) {
    return Row(
      children: [
        Expanded(
          child: Slider(
            value: value.toDouble(),
            min: 1,
            max: 100,
            divisions: 99,
            activeColor: value < 100
                ? const Color(0xFFCC8844)
                : Colors.white54,
            inactiveColor: const Color(0xFF333333),
            onChanged: (v) => onChanged(v.round()),
          ),
        ),
        SizedBox(
          width: 44,
          child: Text(
            '$value%',
            textAlign: TextAlign.right,
            style: TextStyle(
              fontFamily: 'monospace',
              fontSize: 12,
              color: value < 100
                  ? const Color(0xFFCC8844)
                  : Colors.white,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildTimeControlOptions() {
    switch (_timeControlType) {
      case TimeControlType.gameTime:
        return Row(
          children: [
            _numberInput('Min', _minutes, (v) => setState(() => _minutes = v),
                min: 1, max: 180),
            const SizedBox(width: 16),
            _numberInput(
                'Inc(s)', _increment, (v) => setState(() => _increment = v),
                min: 0, max: 60),
          ],
        );
      case TimeControlType.fixedTime:
        return _numberInput(
            'Seconds', _fixedSeconds, (v) => setState(() => _fixedSeconds = v),
            min: 1, max: 300);
      case TimeControlType.fixedDepth:
        return _numberInput(
            'Depth', _depth, (v) => setState(() => _depth = v),
            min: 1, max: 50);
    }
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

  Widget _choiceChip(String label, bool selected, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: MouseRegion(
        cursor: SystemMouseCursors.click,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: selected ? const Color(0xFF444444) : const Color(0xFF2A2A2A),
            borderRadius: BorderRadius.circular(4),
            border: Border.all(
              color: selected ? Colors.white54 : const Color(0xFF444444),
            ),
          ),
          child: Text(
            label,
            style: TextStyle(
              fontFamily: 'monospace',
              fontSize: 12,
              color: selected ? Colors.white : Colors.white54,
            ),
          ),
        ),
      ),
    );
  }

  Widget _numberInput(
      String label, int value, ValueChanged<int> onChanged,
      {int min = 0, int max = 999}) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '$label: ',
          style: const TextStyle(
            fontFamily: 'monospace',
            fontSize: 12,
            color: Color(0xFF888888),
          ),
        ),
        IconButton(
          icon: const Icon(Icons.remove, size: 14),
          onPressed:
              value > min ? () => onChanged(value - 1) : null,
          iconSize: 14,
          constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
          color: Colors.white54,
        ),
        SizedBox(
          width: 36,
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
          onPressed:
              value < max ? () => onChanged(value + 1) : null,
          iconSize: 14,
          constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
          color: Colors.white54,
        ),
      ],
    );
  }

  void _startGame() {
    final TimeControl timeControl;
    switch (_timeControlType) {
      case TimeControlType.gameTime:
        timeControl = TimeControl.gameTime(
          whiteTime: Duration(minutes: _minutes),
          blackTime: Duration(minutes: _minutes),
          increment: Duration(seconds: _increment),
        );
        break;
      case TimeControlType.fixedTime:
        timeControl = TimeControl.fixedTime(
          time: Duration(seconds: _fixedSeconds),
        );
        break;
      case TimeControlType.fixedDepth:
        timeControl = TimeControl.fixedDepth(depth: _depth);
        break;
    }

    int? position;
    if (_chess960 && _specificPosition) {
      position = int.tryParse(_positionController.text);
      if (position != null) {
        position = position.clamp(0, 959);
      }
    }

    ref.read(ponderEnabledProvider.notifier).state = _ponder;
    ref.read(engineProvider.notifier).setSkillLevel(_skillLevel);

    final e2Path = _engine2PathController.text.trim();

    ref.read(gameProvider.notifier).newGame(NewGameOptions(
          playerColor: _engineVsEngine ? PlayerColor.white : _playerColor,
          timeControl: timeControl,
          chess960: _chess960,
          chess960Position: position,
          ponder: _engineVsEngine ? false : _ponder,
          engine1Path: _engine1Path,
          engineVsEngine: _engineVsEngine,
          engine2Path: _engineVsEngine ? (e2Path.isNotEmpty ? e2Path : null) : null,
          engine2Skill: _engine2Skill,
          engine2Hash: _engine2Hash,
          engine2Threads: _engine2Threads,
        ));

    Navigator.pop(context);
  }
}
