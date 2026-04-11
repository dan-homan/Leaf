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

    RegisterGeneratedPlugins(registry: flutterViewController)

    let channel = FlutterMethodChannel(
      name: "leaf_gui/file_picker",
      binaryMessenger: flutterViewController.engine.binaryMessenger
    )
    channel.setMethodCallHandler { (call, result) in
      if call.method == "pickFile" {
        DispatchQueue.main.async {
          let panel = NSOpenPanel()
          panel.canChooseFiles = true
          panel.canChooseDirectories = false
          panel.allowsMultipleSelection = false
          panel.title = "Select engine executable"
          if panel.runModal() == .OK, let url = panel.url {
            result(url.path)
          } else {
            result(nil)
          }
        }
      } else {
        result(FlutterMethodNotImplemented)
      }
    }

    super.awakeFromNib()
  }
}
