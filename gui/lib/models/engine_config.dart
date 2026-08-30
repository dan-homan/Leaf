// LeafGUI — a chess GUI for the Leaf chess engine.
// Copyright (C) 2026 Daniel C. Homan
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.  See the LICENSE file at the root of this repository.

import 'dart:io';

class EngineConfig {
  final String name;
  final String path;
  final int hashMb;
  final int threads;

  const EngineConfig({
    this.name = 'Leaf',
    required this.path,
    this.hashMb = 256,
    this.threads = 1,
  });

  static String defaultEnginePath() {
    // Check for bundled engine first (inside .app bundle).
    if (Platform.isMacOS) {
      final contentsDir = File(Platform.resolvedExecutable).parent.parent.path;
      final bundledPath = '$contentsDir/Resources/engines/Leaf';
      if (File(bundledPath).existsSync()) return bundledPath;
    } else if (Platform.isWindows) {
      final execDir = File(Platform.resolvedExecutable).parent.path;
      final bundledPath = '$execDir/engines/Leaf.exe';
      if (File(bundledPath).existsSync()) return bundledPath;
    } else {
      final execDir = File(Platform.resolvedExecutable).parent.path;
      final bundledPath = '$execDir/engines/Leaf';
      if (File(bundledPath).existsSync()) return bundledPath;
    }

    // Development fallback: use local Leaf binary directly.
    const devPath = '/Users/homand/Leaf/engine/run/Leaf_vcurrent';
    if (File(devPath).existsSync()) return devPath;

    // Last resort: expected bundled path (will show error on start).
    if (Platform.isMacOS) {
      final contentsDir = File(Platform.resolvedExecutable).parent.parent.path;
      return '$contentsDir/Resources/engines/Leaf';
    }
    final execDir = File(Platform.resolvedExecutable).parent.path;
    return '$execDir/engines/Leaf';
  }
}
