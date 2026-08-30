// LeafGUI — a chess GUI for the Leaf chess engine.
// Copyright (C) 2026 Daniel C. Homan
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.  See the LICENSE file at the root of this repository.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'app.dart';
import 'services/engine_registry.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await EngineRegistry().load();
  runApp(const ProviderScope(child: LeafGuiApp()));
}
