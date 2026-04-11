import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../../services/engine_registry.dart';

/// Reusable engine picker: dropdown of registered engines + browse button.
class EnginePicker extends StatelessWidget {
  final String currentPath;
  final ValueChanged<String> onChanged;

  const EnginePicker({
    super.key,
    required this.currentPath,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    final registry = EngineRegistry();
    final engines = registry.engines;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8),
                decoration: BoxDecoration(
                  color: const Color(0xFF2A2A2A),
                  borderRadius: BorderRadius.circular(4),
                  border: Border.all(color: const Color(0xFF444444)),
                ),
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    value: engines.any((e) => e.path == currentPath)
                        ? currentPath
                        : null,
                    isExpanded: true,
                    dropdownColor: const Color(0xFF2A2A2A),
                    style: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 12,
                      color: Colors.white,
                    ),
                    hint: Text(
                      currentPath.isEmpty
                          ? 'Select engine...'
                          : currentPath.split('/').last,
                      style: const TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 12,
                        color: Colors.white70,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                    items: engines
                        .map((e) => DropdownMenuItem(
                              value: e.path,
                              child: Text(
                                e.name,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ))
                        .toList(),
                    onChanged: (path) {
                      if (path != null) onChanged(path);
                    },
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            IconButton(
              icon: const Icon(Icons.folder_open, size: 18),
              color: Colors.white54,
              tooltip: 'Browse...',
              constraints: const BoxConstraints(minWidth: 36, minHeight: 36),
              onPressed: () async {
                const channel = MethodChannel('leaf_gui/file_picker');
                final path =
                    await channel.invokeMethod<String>('pickFile');
                if (path != null) onChanged(path);
              },
            ),
          ],
        ),
        if (currentPath.isNotEmpty &&
            !engines.any((e) => e.path == currentPath))
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              currentPath.split('/').last,
              style: const TextStyle(
                fontFamily: 'monospace',
                fontSize: 10,
                color: Color(0xFF888888),
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
      ],
    );
  }
}
