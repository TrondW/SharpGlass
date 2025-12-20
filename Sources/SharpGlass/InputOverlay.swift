import SwiftUI
import AppKit

struct InputOverlay: NSViewRepresentable {
    @MainActor
    class Coordinator: NSObject {
        var parent: InputOverlay
        
        init(_ parent: InputOverlay) {
            self.parent = parent
        }
        
        @objc func scrollWheel(_ event: NSEvent) {
            parent.onScroll(Double(event.deltaY))
        }
    }
    
    // Callbacks
    var onDrag: (_ delta: CGSize) -> Void
    var onScroll: (_ delta: CGFloat) -> Void
    var onKeyDown: (_ code: UInt16) -> Void
    var onKeyUp: (_ code: UInt16) -> Void
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    func makeNSView(context: Context) -> InputView {
        let view = InputView()
        view.onDrag = onDrag
        view.onScroll = onScroll
        view.onKeyDown = onKeyDown
        view.onKeyUp = onKeyUp
        return view
    }
    
    func updateNSView(_ nsView: InputView, context: Context) {
        nsView.onDrag = onDrag
        nsView.onScroll = onScroll
        nsView.onKeyDown = onKeyDown
        nsView.onKeyUp = onKeyUp
    }
    
    class InputView: NSView {
        var onDrag: ((CGSize) -> Void)?
        var onScroll: ((CGFloat) -> Void)?
        var onKeyDown: ((UInt16) -> Void)?
        var onKeyUp: ((UInt16) -> Void)?
        
        override var acceptsFirstResponder: Bool { true }
        
        // 0: Left, 1: Right, 2: Middle/Other
        
        override func mouseDown(with event: NSEvent) {
            // Focus on click
            window?.makeFirstResponder(self)
        }
        
        override func mouseDragged(with event: NSEvent) {
            onDrag?(CGSize(width: event.deltaX, height: event.deltaY))
        }
        
        override func rightMouseDragged(with event: NSEvent) {
            onDrag?(CGSize(width: event.deltaX, height: event.deltaY))
        }
        
        override func otherMouseDragged(with event: NSEvent) {
            onDrag?(CGSize(width: event.deltaX, height: event.deltaY))
        }
        
        override func scrollWheel(with event: NSEvent) {
            onScroll?(event.deltaY)
        }
        
        override func keyDown(with event: NSEvent) {
            onKeyDown?(event.keyCode)
        }
        
        override func keyUp(with event: NSEvent) {
            onKeyUp?(event.keyCode)
        }
    }
}
