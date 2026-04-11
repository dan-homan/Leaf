import 'dart:async';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_riverpod/legacy.dart';
import '../models/engine_config.dart';
import '../services/engine_registry.dart';
import '../services/uci_engine.dart';
import 'game_provider.dart';

/// The current engine configuration (engine 1 / primary).
final engineConfigProvider = StateProvider<EngineConfig>((ref) {
  return EngineConfig(path: EngineConfig.defaultEnginePath());
});

/// Latest engine info line (engine 1).
final engineInfoProvider = StateProvider<UciInfo>((ref) => const UciInfo());

/// Latest bestmove from engine 1.
final bestMoveProvider = StateProvider<String?>((ref) => null);

/// Engine 1 state (idle, thinking, pondering).
final engineStateProvider =
    StateProvider<UciEngineState>((ref) => UciEngineState.idle);

/// Engine skill level (1-100, percentage of knowledge used).
final skillLevelProvider = StateProvider<int>((ref) => 100);

/// Engine 1 name (from UCI "id name" response).
final engineNameProvider = StateProvider<String?>((ref) => null);

/// The engine 1 instance provider. Manages the engine lifecycle.
final engineProvider =
    StateNotifierProvider<EngineNotifier, UciEngine?>((ref) {
  return EngineNotifier(ref, isEngine2: false);
});

// --- Engine 2 providers (for engine vs engine) ---

/// Engine 2 configuration.
final engine2ConfigProvider = StateProvider<EngineConfig>((ref) {
  return EngineConfig(path: EngineConfig.defaultEnginePath());
});

/// Latest engine 2 info line.
final engine2InfoProvider = StateProvider<UciInfo>((ref) => const UciInfo());

/// Engine 2 state.
final engine2StateProvider =
    StateProvider<UciEngineState>((ref) => UciEngineState.idle);

/// Engine 2 skill level.
final engine2SkillLevelProvider = StateProvider<int>((ref) => 100);

/// Engine 2 name.
final engine2NameProvider = StateProvider<String?>((ref) => null);

/// Engine 2 instance provider.
final engine2Provider =
    StateNotifierProvider<EngineNotifier, UciEngine?>((ref) {
  return EngineNotifier(ref, isEngine2: true);
});

class EngineNotifier extends StateNotifier<UciEngine?> {
  final Ref ref;
  final bool isEngine2;
  final List<StreamSubscription> _subscriptions = [];

  EngineNotifier(this.ref, {required this.isEngine2}) : super(null);

  StateProvider<EngineConfig> get _configProvider =>
      isEngine2 ? engine2ConfigProvider : engineConfigProvider;
  StateProvider<UciInfo> get _infoProvider =>
      isEngine2 ? engine2InfoProvider : engineInfoProvider;
  StateProvider<UciEngineState> get _stateProvider =>
      isEngine2 ? engine2StateProvider : engineStateProvider;
  StateProvider<int> get _skillProvider =>
      isEngine2 ? engine2SkillLevelProvider : skillLevelProvider;
  StateProvider<String?> get _nameProvider =>
      isEngine2 ? engine2NameProvider : engineNameProvider;

  /// Start the engine with the current config.
  Future<bool> startEngine() async {
    await stopEngine();

    final config = ref.read(_configProvider);
    final engine = UciEngine(path: config.path);
    final started = await engine.start();

    if (!started) {
      state = null;
      return false;
    }

    // Store the engine name from UCI identification.
    ref.read(_nameProvider.notifier).state = engine.engineName;

    // Register engine in the registry for future use.
    if (engine.engineName != null) {
      EngineRegistry().register(engine.engineName!, config.path);
    }

    // Configure engine options.
    engine.setOption('Hash', config.hashMb);
    engine.setOption('Threads', config.threads);
    if (!isEngine2) {
      engine.setOption('Ponder', 'true');
    }
    final skill = ref.read(_skillProvider);
    if (skill < 100) {
      engine.setOption('Skill', skill);
    }
    await engine.isReady();

    // Subscribe to engine streams.
    _subscriptions.add(engine.infoStream.listen((info) {
      ref.read(_infoProvider.notifier).state = info;
    }));

    _subscriptions.add(engine.bestMoveStream.listen((bestMove) {
      if (isEngine2) {
        ref.read(gameProvider.notifier).onEngine2BestMove(
              bestMove,
              engine.ponderMove,
            );
      } else {
        ref.read(bestMoveProvider.notifier).state = bestMove;
        ref.read(gameProvider.notifier).onEngineBestMove(
              bestMove,
              engine.ponderMove,
            );
      }
    }));

    _subscriptions.add(engine.stateStream.listen((engineState) {
      ref.read(_stateProvider.notifier).state = engineState;
    }));

    state = engine;
    return true;
  }

  /// Update the skill level on a running engine.
  void setSkillLevel(int skill) {
    ref.read(_skillProvider.notifier).state = skill;
    if (state != null && state!.isRunning) {
      state!.setOption('Skill', skill);
    }
  }

  /// Stop the engine.
  Future<void> stopEngine() async {
    for (final sub in _subscriptions) {
      sub.cancel();
    }
    _subscriptions.clear();

    await state?.stop();
    state = null;
    ref.read(_stateProvider.notifier).state = UciEngineState.idle;
  }

  @override
  void dispose() {
    stopEngine();
    super.dispose();
  }
}
