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
            // Note: This coordinator method isn't used - InputView handles scroll directly
            parent.onScroll(Double(event.deltaY), CGPoint(x: 0.5, y: 0.5), event.modifierFlags)
        }
    }
    
    // Callbacks
    var onMouseDown: () -> Void
    var onMouseUp: () -> Void
    var onDrag: (_ delta: CGSize, _ button: Int, _ modifiers: NSEvent.ModifierFlags) -> Void
    var onScroll: (_ delta: CGFloat, _ position: CGPoint, _ modifiers: NSEvent.ModifierFlags) -> Void
    var onKeyDown: (_ code: UInt16) -> Void
    var onKeyUp: (_ code: UInt16) -> Void
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    func makeNSView(context: Context) -> InputView {
        let view = InputView()
        view.onMouseDown = onMouseDown
        view.onMouseUp = onMouseUp
        view.onDrag = onDrag
        view.onScroll = onScroll
        view.onKeyDown = onKeyDown
        view.onKeyUp = onKeyUp
        return view
    }
    
    func updateNSView(_ nsView: InputView, context: Context) {
        nsView.onMouseDown = onMouseDown
        nsView.onMouseUp = onMouseUp
        nsView.onDrag = onDrag
        nsView.onScroll = onScroll
        nsView.onKeyDown = onKeyDown
        nsView.onKeyUp = onKeyUp
    }
    
    class InputView: NSView {
        var onMouseDown: (() -> Void)?
        var onMouseUp: (() -> Void)?
        var onDrag: ((CGSize, Int, NSEvent.ModifierFlags) -> Void)?
        var onScroll: ((CGFloat, CGPoint, NSEvent.ModifierFlags) -> Void)?
        var onKeyDown: ((UInt16) -> Void)?
        var onKeyUp: ((UInt16) -> Void)?
        
        // Track whether the current drag started in our view
        private var isDraggingInView = false
        
        override var acceptsFirstResponder: Bool { true }
        override var canBecomeKeyView: Bool { true }
        
        override func mouseDown(with event: NSEvent) {
            if shouldIgnoreEvent(event) {
                isDraggingInView = false
                return
            }
            self.window?.makeFirstResponder(self)
            isDraggingInView = true
            onMouseDown?()
        }
        
        override func mouseDragged(with event: NSEvent) {
            // Only process drag if it started in our view (not title bar)
            guard isDraggingInView else { return }
            onDrag?(CGSize(width: event.deltaX, height: event.deltaY), 0, event.modifierFlags)
        }
        
        override func mouseUp(with event: NSEvent) {
            isDraggingInView = false
            onMouseUp?()
        }
        
        override func rightMouseDown(with event: NSEvent) {
            if shouldIgnoreEvent(event) {
                isDraggingInView = false
                return
            }
            self.window?.makeFirstResponder(self)
            isDraggingInView = true
            onMouseDown?()
        }
        
        override func rightMouseDragged(with event: NSEvent) {
            // Only process drag if it started in our view
            guard isDraggingInView else { return }
            onDrag?(CGSize(width: event.deltaX, height: event.deltaY), 1, event.modifierFlags)
        }
        
        override func rightMouseUp(with event: NSEvent) {
            isDraggingInView = false
            onMouseUp?()
        }
        
        override func otherMouseDown(with event: NSEvent) {
            if shouldIgnoreEvent(event) {
                isDraggingInView = false
                return
            }
            self.window?.makeFirstResponder(self)
            isDraggingInView = true
            onMouseDown?()
        }
        
        override func otherMouseDragged(with event: NSEvent) {
            // Only process drag if it started in our view
            guard isDraggingInView else { return }
            onDrag?(CGSize(width: event.deltaX, height: event.deltaY), event.buttonNumber, event.modifierFlags)
        }
        
        override func otherMouseUp(with event: NSEvent) {
            isDraggingInView = false
            onMouseUp?()
        }
        
        private func shouldIgnoreEvent(_ event: NSEvent) -> Bool {
            let location = event.locationInWindow
            guard let window = self.window else { return false }
            // Ignore events in the top 40 pixels (title bar area)
            // macOS Y coordinate starts from bottom (0)
            return location.y > window.frame.height - 40
        }
        
        override func scrollWheel(with event: NSEvent) {
            let position = convert(event.locationInWindow, from: nil)
            // Normalize position to 0-1 range
            let normalizedPos = CGPoint(x: position.x / bounds.width, y: position.y / bounds.height)
            onScroll?(event.deltaY, normalizedPos, event.modifierFlags)
        }
        
        override func keyDown(with event: NSEvent) {
            onKeyDown?(event.keyCode)
        }
        
        override func keyUp(with event: NSEvent) {
            onKeyUp?(event.keyCode)
        }
    }
}
