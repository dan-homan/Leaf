import 'package:flutter/material.dart';

class AppTheme {
  AppTheme._();

  static ThemeData get dark {
    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: const Color(0xFF1A1A1A),
      colorScheme: const ColorScheme.dark(
        primary: Colors.white,
        secondary: Color(0xFF888888),
        surface: Color(0xFF242424),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: Color(0xFF1A1A1A),
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      cardTheme: const CardThemeData(
        color: Color(0xFF242424),
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(8)),
          side: BorderSide(color: Color(0xFF333333)),
        ),
      ),
      textTheme: const TextTheme(
        bodyMedium: TextStyle(
          fontFamily: 'monospace',
          color: Colors.white,
        ),
        bodySmall: TextStyle(
          fontFamily: 'monospace',
          color: Color(0xFF999999),
        ),
      ),
      iconTheme: const IconThemeData(color: Colors.white),
      dividerColor: const Color(0xFF333333),
    );
  }
}
