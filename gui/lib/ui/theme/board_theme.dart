// LeafGUI — a chess GUI for the Leaf chess engine.
// Copyright (C) 2026 Daniel C. Homan
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.  See the LICENSE file at the root of this repository.

import 'package:flutter/material.dart';
import 'package:squares/squares.dart';

class LeafBoardTheme {
  LeafBoardTheme._();

  static BoardTheme get blackAndWhite {
    return const BoardTheme(
      lightSquare: Color(0xFFE8E8E8),
      darkSquare: Color(0xFF606060),
      check: Color(0xFFCC4444),
      checkmate: Color(0xFFCC0000),
      previous: Color(0xFF666688),
      selected: Color(0xFF5577AA),
      premove: Color(0xFF446688),
    );
  }

  static MarkerTheme get markers => MarkerTheme.basic;
}
