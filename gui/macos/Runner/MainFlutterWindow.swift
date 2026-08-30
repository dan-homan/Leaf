// LeafGUI — a chess GUI for the Leaf chess engine.
// Copyright (C) 2026 Daniel C. Homan
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.  See the LICENSE file at the root of this repository.

import Cocoa
import FlutterMacOS

class MainFlutterWindow: NSWindow {
  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    self.contentViewController = flutterViewController

    // Set a sensible default window size for the chess GUI.
    let screenFrame = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1280, height: 800)
    let width: CGFloat = min(1200, screenFrame.width * 0.8)
    let height: CGFloat = min(800, screenFrame.height * 0.85)
    let x = screenFrame.origin.x + (screenFrame.width - width) / 2
    let y = screenFrame.origin.y + (screenFrame.height - height) / 2
    self.setFrame(NSRect(x: x, y: y, width: width, height: height), display: true)
    self.minSize = NSSize(width: 900, height: 600)

    // File picking is handled by the file_selector plugin (registered above),
    // which works on macOS, Windows and Linux.  This used to be a hand-rolled
    // NSOpenPanel MethodChannel, which made the Browse button macOS-only.
    RegisterGeneratedPlugins(registry: flutterViewController)

    super.awakeFromNib()
  }
}
