import SwiftUI

public struct MainView: View {
    public init() {}
    @StateObject private var viewModel = SharpViewModel()
    @State private var isDragging = false
    
    // Gesture base states to prevent jumping
    @State private var baseYaw: Double = 0
    @State private var basePitch: Double = 0
    @State private var baseZoom: Double = 0
    
    public var body: some View {

        ZStack {
            // Handle Drops Anywhere
            Color.clear
                .contentShape(Rectangle())
                .onDrop(of: ["public.file-url"], isTargeted: $isDragging) { providers in
                    guard let item = providers.first else { return false }
                    _ = item.loadObject(ofClass: URL.self) { url, error in
                        if let url = url {
                            DispatchQueue.main.async {
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                                    viewModel.importFile(url: url)
                                }
                            }
                        }
                    }
                    return true
                }

            // ---------------------------------------------------------
            // 1. CONTENT LAYER
            // ---------------------------------------------------------
            ZStack {
                // Backgrounds
                Color.black.ignoresSafeArea()
                LiquidBackground()
                
                Group {
                    if let _ = viewModel.gaussians {
                        // 3D Viewport
                        TimelineView(.animation) { timeline in
                            ZStack {
                                MetalSplatView(
                                    gaussians: viewModel.gaussians,
                                    camera: viewModel.camera,
                                    viewModel: viewModel
                                )
                                .edgesIgnoringSafeArea(.all)
                                
                                // View Cube (Top Right)
                                VStack {
                                    HStack {
                                        Spacer()
                                        ViewCube(rotation: viewModel.camera.viewMatrix()) { theta, phi, dist in
                                            viewModel.snapCamera(theta: theta, phi: phi, distance: dist)
                                        }
                                        .padding(.trailing, 24)
                                        .padding(.top, 60)
                                    }
                                    Spacer()
                                }
                            }
                            .onChange(of: timeline.date) { _, newDate in
                                viewModel.updateCamera(time: newDate.timeIntervalSinceReferenceDate)
                            }
                        }
                        .overlay(
                            InputOverlay(
                                onMouseDown: { viewModel.isNavigating = true },
                                onMouseUp: { viewModel.isNavigating = false },
                                onDrag: { delta, button, mods in
                                    viewModel.handleDrag(delta: delta, button: button, modifiers: mods)
                                },
                                onScroll: { delta, position, mods in
                                    viewModel.handleScroll(delta: delta, position: position, modifiers: mods)
                                },
                                onKeyDown: viewModel.handleKeyDown,
                                onKeyUp: viewModel.handleKeyUp
                            )
                        )
                    } else if let original = viewModel.selectedImage {
                        // Full Bleed Image - Constrained to Window Size
                        // GeometryReader ensures we fill the window but NEVER exceed it.
                        GeometryReader { proxy in
                            Image(nsImage: original)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: proxy.size.width, height: proxy.size.height)
                                .clipped()
                        }
                    } else {
                        // Empty State
                        VStack(spacing: 16) {
                            Image(systemName: "photo.on.rectangle.angled")
                                .font(.system(size: 40, weight: .ultraLight))
                                .foregroundStyle(.white.opacity(0.15))
                            Text("Drop an image to begin")
                                .font(.system(size: 13, weight: .regular))
                                .foregroundStyle(.white.opacity(0.3))
                        }
                    }
                }
                
                // Drop Overlay
                if isDragging {
                    Rectangle()
                        .fill(.ultraThinMaterial)
                        .overlay(
                            VStack(spacing: 12) {
                                Image(systemName: "arrow.down.circle")
                                    .font(.system(size: 32, weight: .light))
                                Text("Release to open")
                                    .font(.system(size: 13, weight: .medium))
                            }
                            .foregroundStyle(.white.opacity(0.7))
                        )
                        .transition(.opacity.animation(.easeInOut(duration: 0.2)))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .ignoresSafeArea()
            
            // ---------------------------------------------------------
            // 2. CHROME LAYER (UI) + ONBOARDING OVERLAY
            // ---------------------------------------------------------
            ZStack {
                VStack {
                    // Title Bar (Centered)
                    Text("SharpGlass")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.6))
                        .padding(.top, 12)
                    
                    Spacer()
                    
                    // Bottom Area (Stats + Pill)
                    ZStack(alignment: .bottom) {
                        // Stats (Bottom Left)
                        if let g = viewModel.gaussians {
                            HStack {
                                Text("\(g.pointCount.formatted()) splats")
                                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.35))
                                Spacer()
                            }
                            .padding(.leading, 16)
                            .padding(.bottom, 16)
                        }
                        
                        // Liquid Glass Floating Pill Menu (Bottom Center)
                        LiquidGlassPillMenu(
                            canOpen: true,
                            canGenerate: viewModel.selectedImage != nil && viewModel.gaussians == nil && viewModel.isAvailable,
                            canSave: viewModel.gaussians != nil,
                            onOpen: viewModel.loadImage,
                            onGenerate: viewModel.generate3D,
                            onSave: {
                                let panel = NSSavePanel()
                                panel.allowedContentTypes = [.init(filenameExtension: "ply")!]
                                panel.nameFieldStringValue = "scene.ply"
                                if panel.runModal() == .OK, let url = panel.url {
                                    viewModel.saveSplat(to: url)
                                }
                            }
                        )
                        .padding(.bottom, 30)

                        // Holographic Mode Toggle (Bottom Right)
                        if viewModel.gaussians != nil {
                            HStack {
                                Spacer()
                                Button(action: { viewModel.isHolographicModeEnabled.toggle() }) {
                                    HStack(spacing: 6) {
                                        Image(systemName: viewModel.isHolographicModeEnabled ? "eye.fill" : "eye.slash")
                                            .font(.system(size: 11, weight: .bold))
                                        Text("HOLOGRAPHIC")
                                            .font(.system(size: 9, weight: .black, design: .monospaced))
                                    }
                                    .padding(.horizontal, 10)
                                    .frame(height: 28)
                                    .background(viewModel.isHolographicModeEnabled ? Color.cyan.opacity(0.8) : Color.white.opacity(0.1))
                                    .cornerRadius(14)
                                    .foregroundStyle(viewModel.isHolographicModeEnabled ? .black : .white.opacity(0.6))
                                }
                                .buttonStyle(.plain)
                                .padding(.trailing, 16)
                                .padding(.bottom, 38)
                                .shadow(color: .black.opacity(0.3), radius: 10)
                            }
                        }
                    }
                }
                
                // Onboarding Overlay (Centered Custom Modal)
                if !viewModel.isAvailable {
                    Color.black.opacity(0.7) // Dimmed background
                        .ignoresSafeArea()
                    
                    SetupInstructionsView(viewModel: viewModel)
                        .frame(width: 320)
                        .shadow(color: .black.opacity(0.5), radius: 20, x: 0, y: 10)
                }
            }
            .ignoresSafeArea()
        }
    }
}

// MARK: - Liquid Glass Pill Menu (Tahoe Spec)
struct LiquidGlassPillMenu: View {
    let canOpen: Bool
    let canGenerate: Bool
    let canSave: Bool
    let onOpen: () -> Void
    let onGenerate: () -> Void
    let onSave: () -> Void
    
    var body: some View {
        HStack(spacing: 0) {
            // Open
            Button(action: onOpen) {
                Image(systemName: "folder")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(canOpen ? .white : .white.opacity(0.3))
                    .frame(width: 52, height: 44)
            }
            .buttonStyle(.plain)
            .disabled(!canOpen)
            .help("Open Image")
            
            // Divider
            Rectangle()
                .fill(.white.opacity(0.15))
                .frame(width: 0.5, height: 22)
            
            // Generate
            Button(action: onGenerate) {
                Image(systemName: "cube")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(canGenerate ? .white : .white.opacity(0.3))
                    .frame(width: 52, height: 44)
            }
            .buttonStyle(.plain)
            .disabled(!canGenerate)
            .help("Generate 3D Splats")
            
            // Divider
            Rectangle()
                .fill(.white.opacity(0.15))
                .frame(width: 0.5, height: 22)
            
            // Save
            Button(action: onSave) {
                Image(systemName: "square.and.arrow.down")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(canSave ? .white : .white.opacity(0.3))
                    .frame(width: 52, height: 44)
            }
            .buttonStyle(.plain)
            .disabled(!canSave)
            .help("Save Scene")
        }
        // Liquid Glass: blur with black at 25% opacity (dark mode)
        .background(
            Capsule()
                .fill(.black.opacity(0.15))
                .background(
                    Capsule()
                        .fill(.ultraThinMaterial)
                )
        )
        .clipShape(Capsule())
        // Inner highlight (glass edge) - white gradient top-to-bottom
        .overlay(
            Capsule()
                .strokeBorder(
                    LinearGradient(
                        colors: [.white.opacity(0.35), .white.opacity(0.1)],
                        startPoint: .top,
                        endPoint: .bottom
                    ),
                    lineWidth: 0.5
                )
        )
        // Drop shadow - blur 40px, Y 20px, 15% opacity
        .shadow(color: .black.opacity(0.2), radius: 30, y: 15)
    }
}







struct SetupInstructionsView: View {
    @ObservedObject var viewModel: SharpViewModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            
            if viewModel.setupStatus == .installing {
                installProgress
            } else {
                actionContent
            }
            
            if viewModel.setupStatus != .installing {
                manualLink
            }
        }
        .padding(12)
        .background(Color.black.opacity(0.3))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(Color.white.opacity(0.1), lineWidth: 0.5)
        )
    }
    
    var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "cpu")
                .foregroundStyle(.cyan.opacity(0.8))
            Text("ML BACKEND REQUIRED")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(.white.opacity(0.6))
        }
    }
    
    var installProgress: some View {
        VStack(alignment: .leading, spacing: 6) {
            ProgressView(value: viewModel.setupProgress, total: 1.0)
                .progressViewStyle(.linear)
                .tint(.cyan)
            
            HStack {
                Text(viewModel.setupStep.uppercased())
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(.white.opacity(0.5))
                Spacer()
                Text("\(Int(viewModel.setupProgress * 100))%")
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.3))
            }
        }
    }
    
    var actionContent: some View {
        Group {
            if viewModel.setupStatus == .failed {
                Text("Error: \(viewModel.errorMessage ?? "Unknown error")")
                    .font(.system(size: 9))
                    .foregroundStyle(.red.opacity(0.9))
                    .lineLimit(3)
                    .padding(8)
                    .background(Color.red.opacity(0.1))
                    .cornerRadius(6)
            } else {
                Text("SharpGlass requires a one-time setup to install the AI engine (approx. 2GB).")
                    .font(.system(size: 9))
                    .foregroundStyle(.white.opacity(0.6))
                    .fixedSize(horizontal: false, vertical: true)
            }
            
            Button(action: {
                viewModel.installBackend()
            }) {
                HStack {
                    if viewModel.setupStatus == .failed {
                        Image(systemName: "arrow.clockwise")
                        Text("RETRY INSTALLATION")
                    } else {
                        Image(systemName: "arrow.down.circle.fill")
                        Text("INSTALL ENGINE")
                    }
                }
                .font(.system(size: 9, weight: .bold))
                .frame(maxWidth: .infinity)
                .frame(height: 28)
            }
            .buttonStyle(CinematicButtonStyle(prominent: true))
        }
    }
    
    var manualLink: some View {
        Link("Manual Installation Guide", destination: URL(string: "https://github.com/apple/ml-sharp")!)
           .font(.system(size: 8))
           .tint(.white.opacity(0.3))
           .padding(.top, 4)
    }
}


