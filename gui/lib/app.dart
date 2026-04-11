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
