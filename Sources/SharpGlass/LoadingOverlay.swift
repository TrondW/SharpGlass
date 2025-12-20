// LoadingOverlay.swift
import SwiftUI

/// A full‑screen overlay that shows a spinner and optional message while the app is busy.
struct LoadingOverlay: View {
    var message: String = "Processing..."
    var body: some View {
        ZStack {
            Rectangle()
                .fill(.ultraThinMaterial)
                .background(Color.black.opacity(0.2))
                .ignoresSafeArea()
            
            VStack(spacing: 24) {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    .scaleEffect(1.3)
                
                Text(message.uppercased())
                    .font(.system(size: 11, weight: .black, design: .rounded))
                    .kerning(4)
                    .foregroundColor(.white.opacity(0.4))
            }
        }
    }
}

// Convenience view modifier to overlay LoadingOverlay when a binding is true.
struct LoadingOverlayModifier: ViewModifier {
    @Binding var isPresented: Bool
    var message: String = "Processing..."
    func body(content: Content) -> some View {
        content
            .overlay(
                Group {
                    if isPresented {
                        LoadingOverlay(message: message)
                    }
                }
            )
    }
}

extension View {
    func loadingOverlay(isPresented: Binding<Bool>, message: String = "Processing...") -> some View {
        self.modifier(LoadingOverlayModifier(isPresented: isPresented, message: message))
    }
}

// End of LoadingOverlay.swift
