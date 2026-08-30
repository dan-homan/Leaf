// LeafGUI — a chess GUI for the Leaf chess engine.
// Copyright (C) 2026 Daniel C. Homan
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.  See the LICENSE file at the root of this repository.

import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart' as p;

class RegisteredEngine {
  final String name;
  final String path;

  const RegisteredEngine({required this.name, required this.path});

  Map<String, dynamic> toJson() => {'name': name, 'path': path};

  factory RegisteredEngine.fromJson(Map<String, dynamic> json) {
    return RegisteredEngine(
      name: json['name'] as String,
      path: json['path'] as String,
    );
  }

  @override
  bool operator ==(Object other) =>
      other is RegisteredEngine && other.path == path;

  @override
  int get hashCode => path.hashCode;
}

class EngineRegistry {
  static final EngineRegistry _instance = EngineRegistry._();
  factory EngineRegistry() => _instance;
  EngineRegistry._();

  List<RegisteredEngine> _engines = [];
  bool _loaded = false;

  List<RegisteredEngine> get engines => List.unmodifiable(_engines);

  /// Home directory for the engine registry.
  ///
  /// `HOME` is not set on Windows, where the equivalents are `USERPROFILE` and
  /// `APPDATA`; without this fallback the registry silently landed in whatever
  /// the current working directory happened to be.  Unix keeps using `HOME` so
  /// existing `~/.leafgui/engines.json` files are found unchanged.
  static String get _homeDir =>
      Platform.environment['HOME'] ??
      Platform.environment['USERPROFILE'] ??
      Platform.environment['APPDATA'] ??
      Directory.current.path;

  static File get _file {
    final dir = Directory(p.join(_homeDir, '.leafgui'));
    if (!dir.existsSync()) dir.createSync(recursive: true);
    return File(p.join(dir.path, 'engines.json'));
  }

  Future<void> load() async {
    if (_loaded) return;
    _loaded = true;
    final file = _file;
    if (!file.existsSync()) return;
    try {
      final json = jsonDecode(await file.readAsString()) as List;
      _engines = json
          .map((e) => RegisteredEngine.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) {
      // Corrupted file — start fresh.
    }
  }

  Future<void> register(String name, String path) async {
    final existing = _engines.indexWhere((e) => e.path == path);
    if (existing >= 0) {
      // Update name if it changed.
      if (_engines[existing].name != name) {
        _engines[existing] = RegisteredEngine(name: name, path: path);
        await _save();
      }
      return;
    }
    _engines.add(RegisteredEngine(name: name, path: path));
    await _save();
  }

  Future<void> _save() async {
    final json = _engines.map((e) => e.toJson()).toList();
    await _file.writeAsString(jsonEncode(json));
  }
}
