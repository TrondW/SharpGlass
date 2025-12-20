// DesignSystem.swift
import SwiftUI

// MARK: - Color Palette
struct AppColors {
    // Cinematic "Nebula" palette (Extreme depth)
    static let liquidBackgrounds = [
        Color(red: 0.01, green: 0.01, blue: 0.02), // Deep Space
        Color(red: 0.02, green: 0.01, blue: 0.05), // Faint Violet
        Color(red: 0.01, green: 0.02, blue: 0.06), // Midnight Blue
        Color(red: 0.00, green: 0.00, blue: 0.00)  // Obsidian
    ]
    
    static let glassBase = Color.black.opacity(0.1)
    static let accent = Color.white
    static let textPrimary = Color.white
    static let textSecondary = Color.white.opacity(0.4)
}

// MARK: - Components

struct LiquidBackground: View {
    @State private var phase: CGFloat = 0
    
    var body: some View {
        let p = phase
        
        // Slower, more atmospheric movement
        let meshPoints: [SIMD2<Float>] = [
            [0, 0], [0.5, 0], [1, 0],
            [0, 0.5], [0.5, 0.5], [1, 0.5],
            [0, 1], [0.5, 1], [1, 1]
        ].map { point in
            let fx = Float(point[0])
            let fy = Float(point[1])
            return SIMD2<Float>(
                fx + Float(sin(p * 0.5 + CGFloat(fx) * 3)) * 0.015,
                fy + Float(cos(p * 0.5 + CGFloat(fy) * 3)) * 0.015
            )
        }
        
        let meshColors: [Color] = [
            AppColors.liquidBackgrounds[0], AppColors.liquidBackgrounds[1], AppColors.liquidBackgrounds[2],
            AppColors.liquidBackgrounds[1], AppColors.liquidBackgrounds[2], AppColors.liquidBackgrounds[3],
            AppColors.liquidBackgrounds[2], AppColors.liquidBackgrounds[3], AppColors.liquidBackgrounds[0]
        ]
        
        return MeshGradient(width: 3, height: 3, points: meshPoints, colors: meshColors)
            .ignoresSafeArea()
            .onAppear {
                withAnimation(.linear(duration: 60).repeatForever(autoreverses: true)) {
                    phase = .pi * 2
                }
            }
    }
}

// MARK: - View Modifiers

struct GlassBackground: ViewModifier {
    func body(content: Content) -> some View {
        content
            .background(.ultraThinMaterial)
            .background(Color.black.opacity(0.15))
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(Color.white.opacity(0.04), lineWidth: 0.5)
            )
    }
}

struct CinematicButtonStyle: ButtonStyle {
    var prominent: Bool = false
    
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(prominent ? Color.white.opacity(0.15) : Color.white.opacity(0.05))
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(Color.white.opacity(configuration.isPressed ? 0.2 : 0.08), lineWidth: 0.5)
            )
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .animation(.spring(response: 0.2, dampingFraction: 0.8), value: configuration.isPressed)
    }
}

struct HoverLiquidEffect: ViewModifier {
    @State private var isHovered = false
    
    func body(content: Content) -> some View {
        content
            .scaleEffect(isHovered ? 1.01 : 1.0)
            .animation(.spring(response: 0.4, dampingFraction: 0.8), value: isHovered)
            .onHover { hovering in
                isHovered = hovering
            }
    }
}

extension View {
    func glassBackground() -> some View { self.modifier(GlassBackground()) }
    func hoverScale() -> some View { self.modifier(HoverLiquidEffect()) }
}

// MARK: - Shadow Tokens (Legacy compatibility)
struct Shadow {
    let color: Color
    let radius: CGFloat
    let x: CGFloat
    let y: CGFloat
}
struct AppShadows {
    static let elevation1 = Shadow(color: Color.black.opacity(0.3), radius: 4, x: 0, y: 2)
    static let elevation2 = Shadow(color: Color.black.opacity(0.4), radius: 8, x: 0, y: 4)
}

