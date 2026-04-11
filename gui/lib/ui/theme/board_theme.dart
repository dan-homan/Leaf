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
