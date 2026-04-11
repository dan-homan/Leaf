import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'app.dart';
import 'services/engine_registry.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await EngineRegistry().load();
  runApp(const ProviderScope(child: LeafGuiApp()));
}
