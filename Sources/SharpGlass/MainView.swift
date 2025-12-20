import SwiftUI

struct MainView: View {
    @StateObject private var viewModel = SharpViewModel()
    @State private var isDragging = false
    
    // Gesture base states to prevent jumping
    @State private var baseYaw: Double = 0
    @State private var basePitch: Double = 0
    @State private var baseZoom: Double = 0
    
    var body: some View {
        ZStack {
            // Root Background to prevent white bars
            Color.black.ignoresSafeArea()
            

            
            // Cinematic Background (Nebula)
            LiquidBackground()
            
            // Immersive Content Layer
            ZStack(alignment: .trailing) {
                // Main Viewer (Full Bleed)
                ZStack {
                    if let gaussians = viewModel.gaussians {
                        TimelineView(.animation) { timeline in
                            // 2. 3D Viewport
                            ZStack {
                                MetalSplatView(
                                    gaussians: viewModel.gaussians,
                                    camera: viewModel.camera,
                                    viewModel: viewModel
                                )
                                .edgesIgnoringSafeArea(.all)
                            }
                            .onChange(of: timeline.date) { newDate in
                                viewModel.updateCamera(time: newDate.timeIntervalSinceReferenceDate)
                            }
                        }

                        // Input Overlay on top to capture all mouse/keyboard events
                        .overlay(
                            InputOverlay(
                                onDrag: viewModel.handleDrag,
                                onScroll: viewModel.handleScroll,
                                onKeyDown: viewModel.handleKeyDown,
                                onKeyUp: viewModel.handleKeyUp
                            )
                        )
                    } else if let original = viewModel.selectedImage {
                        Image(nsImage: original)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .opacity(viewModel.isProcessing ? 0.05 : 0.4)
                            .padding(100)
                            .blur(radius: viewModel.isProcessing ? 20 : 0)
                            .saturation(0.5)
                    } else {
                        VStack(spacing: 24) {
                            Image(systemName: "plus")
                                .font(.system(size: 24, weight: .ultraLight))
                                .foregroundStyle(.white.opacity(0.1))
                            Text("DRAG AND DROP")
                                .font(.system(size: 11, weight: .bold, design: .rounded))
                                .kerning(3)
                                .foregroundStyle(.white.opacity(0.2))
                        }
                    }
                    
                    if isDragging {
                        Rectangle()
                            .fill(.ultraThinMaterial)
                            .background(Color.black.opacity(0.3))
                            .overlay(
                                RoundedRectangle(cornerRadius: 16)
                                    .strokeBorder(Color.white.opacity(0.15), style: StrokeStyle(lineWidth: 1, dash: [4]))
                                    .padding(60)
                            )
                            .overlay(
                                Text("RELEASE TO SYNTHESIZE")
                                    .font(.system(size: 12, weight: .bold))
                                    .kerning(2)
                                    .foregroundStyle(.white.opacity(0.6))
                            )
                            .transition(.opacity.animation(.easeInOut))
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .contentShape(Rectangle())
                .onDrop(of: ["public.file-url"], isTargeted: $isDragging) { providers in
                    guard let item = providers.first else { return false }
                    
                    // Modern URL loading
                    _ = item.loadObject(ofClass: URL.self) { url, error in
                        if let url = url {
                            DispatchQueue.main.async {
                                viewModel.importFile(url: url)
                            }
                        } else if let error = error {
                            print("Drop Error: \(error.localizedDescription)")
                        }
                    }
                    return true
                }
                
                // Balanced Professional Sidebar
                SidebarView(viewModel: viewModel)
            }
            .loadingOverlay(isPresented: $viewModel.isProcessing)
        }
        .ignoresSafeArea()
}
}



// MARK: - Sidebar View
struct SidebarView: View {
    @ObservedObject var viewModel: SharpViewModel
    
    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 20) {
                // Header
                VStack(alignment: .leading, spacing: 2) {
                    Text("SHARPGLASS")
                        .font(.system(size: 13, weight: .black, design: .rounded))
                        .kerning(4)
                        .foregroundStyle(.white)
                    
                    Text("3D GENERATIVE SYNTHESIS")
                        .font(.system(size: 8, weight: .bold))
                        .kerning(1)
                        .foregroundStyle(.white.opacity(0.3))
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 20)
                
                if let error = viewModel.errorMessage {
                    Text(error)
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.red.opacity(0.8))
                        .padding(10)
                        .background(Color.red.opacity(0.05))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                
                // Action Steps
                VStack(spacing: 24) {
                    WorkflowStep(number: 1, title: "Source") {
                        Button(action: viewModel.loadImage) {
                            Text(viewModel.selectedImage == nil ? "SELECT IMAGE" : "REPLACE")
                                .font(.system(size: 9, weight: .bold))
                                .frame(maxWidth: .infinity)
                                .frame(height: 30)
                        }
                        .buttonStyle(CinematicButtonStyle())
                    }
                    
                    WorkflowStep(number: 2, title: "Process") {
                        Button(action: viewModel.generate3D) {
                            Text(viewModel.gaussians == nil ? "START" : "SYNC")
                                .font(.system(size: 9, weight: .bold))
                                .frame(maxWidth: .infinity)
                                .frame(height: 30)
                        }
                        .buttonStyle(CinematicButtonStyle(prominent: true))
                        .disabled(!viewModel.isAvailable || viewModel.selectedImage == nil || viewModel.isProcessing)
                    }
                    
                    if let _ = viewModel.gaussians {
                        // Stats Overlay
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("SCENE STATS")
                                    .font(.system(size: 8, weight: .bold))
                                    .foregroundColor(.white.opacity(0.4))
                                Spacer()
                            }
                            
                            HStack(spacing: 12) {
                                VStack(alignment: .leading, spacing: 0) {
                                    Text(viewModel.pointCountFormatted)
                                        .font(.system(size: 13, weight: .medium, design: .monospaced))
                                        .foregroundColor(.white)
                                    Text("Splats")
                                        .font(.system(size: 8))
                                        .foregroundColor(.white.opacity(0.5))
                                }
                                
                                Color.white.opacity(0.1)
                                    .frame(width: 1, height: 20)
                                
                                VStack(alignment: .leading, spacing: 0) {
                                    Text(viewModel.memoryUsageFormatted)
                                        .font(.system(size: 13, weight: .medium, design: .monospaced))
                                        .foregroundColor(.white)
                                    Text("VRAM")
                                        .font(.system(size: 8))
                                        .foregroundColor(.white.opacity(0.5))
                                }
                            }
                        }
                        .padding(12)
                        .background(RoundedRectangle(cornerRadius: 6).fill(Color.white.opacity(0.05)))
                        .padding(.bottom, 12)
                        
                        WorkflowStep(number: 3, title: "Refine") {
                            VStack(spacing: 20) {
                                VStack(spacing: 12) {
                                    Picker("Mode", selection: $viewModel.cameraMode) {
                                        Text("Object").tag(SharpViewModel.CameraMode.orbit)
                                        Text("World").tag(SharpViewModel.CameraMode.fly)
                                        Text("Cinema").tag(SharpViewModel.CameraMode.cinema)
                                    }
                                    .pickerStyle(SegmentedPickerStyle())
                                    
                                    // Style & Grading
                                    VStack(alignment: .leading, spacing: 10) {
                                        Text("STYLE")
                                            .font(.system(size: 9, weight: .bold))
                                            .foregroundStyle(.white.opacity(0.4))
                                        
                                        CameraSliderSimplified(value: $viewModel.exposure, label: "EXPOSURE", range: -2...2) {}
                                        CameraSliderSimplified(value: $viewModel.gamma, label: "GAMMA", range: 0.5...2.5) {}
                                        CameraSliderSimplified(value: $viewModel.vignetteStrength, label: "VIGNETTE", range: 0...1) {}
                                    }
                                    .padding(.vertical, 4)
                                    
                                    Divider().background(Color.white.opacity(0.1))
                                    
                                    if viewModel.cameraMode == .cinema {
                                        Picker("Style", selection: $viewModel.selectedStyle) {
                                            Text("Orbit").tag(ParallaxStyle.orbit)
                                            Text("Dolly").tag(ParallaxStyle.dolly)
                                            Text("Pan").tag(ParallaxStyle.horizontal)
                                            Text("Tilt").tag(ParallaxStyle.vertical)
                                            Text("Ken Burns").tag(ParallaxStyle.kenBurns)
                                        }
                                        
                                        HStack {
                                            Text("Duration")
                                                .font(.caption)
                                                .foregroundColor(.white.opacity(0.6))
                                            Slider(value: $viewModel.duration, in: 1...10, step: 0.5)
                                                .controlSize(.mini)
                                            Text(String(format: "%.1fs", viewModel.duration))
                                                .font(.system(size: 9, design: .monospaced))
                                                .frame(width: 36)
                                        }
                                        
                                        Divider().background(Color.white.opacity(0.1))
                                        
                                        Button(action: {
                                            let panel = NSSavePanel()
                                            panel.allowedContentTypes = [.mpeg4Movie]
                                            panel.nameFieldStringValue = "sharp_scene.mp4"
                                            if panel.runModal() == .OK, let url = panel.url {
                                                Task {
                                                    try? await viewModel.exportVideo(url: url, duration: viewModel.duration)
                                                }
                                            }
                                        }) {
                                            HStack {
                                                if viewModel.isExporting {
                                                    ProgressView().controlSize(.small)
                                                    Text("\(Int(viewModel.exportProgress * 100))%")
                                                } else {
                                                    Image(systemName: "film")
                                                    Text("EXPORT VIDEO")
                                                }
                                            }
                                            .font(.system(size: 11, weight: .bold))
                                            .frame(maxWidth: .infinity)
                                            .frame(height: 32)
                                        }
                                        .buttonStyle(CinematicButtonStyle(prominent: true))
                                        .disabled(viewModel.isExporting)
                                    }
                                }
                                
                                if viewModel.cameraMode != .cinema {
                                    InputHelperView(mode: viewModel.cameraMode)
                                }
                            }
                        }
                    }
                    
                    if !viewModel.isAvailable {
                        SetupInstructionsView()
                    }
                }
            }
            .padding(12)
        }
        .background(.ultraThinMaterial)
        .background(Color.black.opacity(0.15))
        .frame(width: 220)
        .overlay(
            Rectangle()
                .fill(Color.white.opacity(0.04))
                .frame(width: 0.5)
                .frame(maxWidth: .infinity, alignment: .leading)
        )
    }
}


struct WorkflowStep<Content: View>: View {
    let number: Int
    let title: String
    let content: Content
    
    init(number: Int, title: String, @ViewBuilder content: () -> Content) {
        self.number = number
        self.title = title
        self.content = content()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Text("\(number)")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.3))
                    .frame(width: 16, height: 16)
                    .overlay(Circle().stroke(Color.white.opacity(0.1), lineWidth: 1))
                
                Text(title.uppercased())
                    .font(.system(size: 10, weight: .bold))
                    .kerning(1)
                    .foregroundStyle(.white.opacity(0.6))
                    .minimumScaleFactor(0.8)
                    .lineLimit(1)
            }
            
            content
        }
    }
}

struct CameraSliderSimplified: View {
    @Binding var value: Double
    let label: String
    let range: ClosedRange<Double>
    let onChanged: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
                    .minimumScaleFactor(0.5)
                    .lineLimit(1)
                Spacer()
                Text(String(format: "%.1f", value))
                    .font(.system(size: 9, design: .monospaced))
            }
            Slider(value: $value, in: range) { editing in
                if !editing { onChanged() }
            }
            .controlSize(.mini)
        }
    }
}

struct SetupInstructionsView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "info.circle")
                Text("Setup Required")
            }
            .font(.system(size: 10, weight: .bold))
            .foregroundStyle(.white.opacity(0.5))
            
            Text("Apple's ml-sharp logic is missing. Install via github.com/apple/ml-sharp")
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
                .minimumScaleFactor(0.5)
            
            Link("Guide", destination: URL(string: "https://github.com/apple/ml-sharp")!)
                .font(.system(size: 9, weight: .medium))
                .tint(.white.opacity(0.8))
        }
        .padding(10)
        .background(Color.white.opacity(0.02))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}


struct InputHelperView: View {
    let mode: SharpViewModel.CameraMode
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                Image(systemName: "keyboard")
                Text("CONTROLS")
            }
            .font(.system(size: 8, weight: .bold))
            .foregroundStyle(.white.opacity(0.4))
            
            Group {
                if mode == .fly {
                    controlRow(key: "WASD", action: "Move")
                    controlRow(key: "Shift/Space", action: "Up/Down")
                    controlRow(key: "Mouse", action: "Look")
                } else {
                    controlRow(key: "LMB", action: "Rotate")
                    controlRow(key: "RMB", action: "Pan")
                    controlRow(key: "Scroll", action: "Zoom")
                }
            }
        }
        .padding(8)
        .background(Color.white.opacity(0.03))
        .cornerRadius(4)
    }
    
    func controlRow(key: String, action: String) -> some View {
        HStack {
            Text(key)
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(.white.opacity(0.8))
            Spacer()
            Text(action)
                .font(.system(size: 9))
                .foregroundStyle(.white.opacity(0.6))
        }
    }
}
