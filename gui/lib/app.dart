// LeafGUI — a chess GUI for the Leaf chess engine.
// Copyright (C) 2026 Daniel C. Homan
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.  See the LICENSE file at the root of this repository.

import 'package:flutter/material.dart';
import 'ui/theme/app_theme.dart';
import 'ui/screens/home_screen.dart';

class LeafGuiApp extends StatelessWidget {
  const LeafGuiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LeafGUI',
      theme: AppTheme.dark,
      debugShowCheckedModeBanner: false,
      home: const HomeScreen(),
    );
  }
}
