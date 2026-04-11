enum GameMode { play, analysis, engineVsEngine }

enum PlayerColor { white, black }

enum TimeControlType { gameTime, fixedTime, fixedDepth }

class TimeControl {
  final TimeControlType type;
  final Duration whiteTime;
  final Duration blackTime;
  final Duration increment;
  final int? depthLimit;

  const TimeControl({
    this.type = TimeControlType.gameTime,
    this.whiteTime = const Duration(minutes: 5),
    this.blackTime = const Duration(minutes: 5),
    this.increment = const Duration(seconds: 3),
    this.depthLimit,
  });

  const TimeControl.gameTime({
    this.whiteTime = const Duration(minutes: 5),
    this.blackTime = const Duration(minutes: 5),
    this.increment = const Duration(seconds: 3),
  })  : type = TimeControlType.gameTime,
        depthLimit = null;

  const TimeControl.fixedTime({
    Duration time = const Duration(seconds: 5),
  })  : type = TimeControlType.fixedTime,
        whiteTime = time,
        blackTime = time,
        increment = Duration.zero,
        depthLimit = null;

  const TimeControl.fixedDepth({
    int depth = 10,
  })  : type = TimeControlType.fixedDepth,
        whiteTime = Duration.zero,
        blackTime = Duration.zero,
        increment = Duration.zero,
        depthLimit = depth;
}

class NewGameOptions {
  final PlayerColor playerColor;
  final TimeControl timeControl;
  final bool chess960;
  final int? chess960Position;
  final bool ponder;
  final String? engine1Path;
  final bool engineVsEngine;
  final String? engine2Path;
  final int engine2Skill;
  final int engine2Hash;
  final int engine2Threads;

  const NewGameOptions({
    this.playerColor = PlayerColor.white,
    this.timeControl = const TimeControl.gameTime(),
    this.chess960 = false,
    this.chess960Position,
    this.ponder = true,
    this.engine1Path,
    this.engineVsEngine = false,
    this.engine2Path,
    this.engine2Skill = 100,
    this.engine2Hash = 256,
    this.engine2Threads = 1,
  });
}
