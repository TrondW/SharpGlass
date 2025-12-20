import SwiftUI
import AppKit
import simd

@MainActor
class SharpViewModel: ObservableObject {
    @Published var selectedImage: NSImage?
    @Published var isProcessing = false
    @Published var isAvailable = false
    @Published var gaussians: GaussianSplatData?

    @Published var errorMessage: String?
    
    // Renderer Reference (for Offline Export)
    weak var renderer: MetalSplatRenderer?
    
    // Stats
    var pointCountFormatted: String {
        guard let g = gaussians else { return "0" }
        return NumberFormatter.localizedString(from: NSNumber(value: g.pointCount), number: .decimal)
    }
    
    var memoryUsageFormatted: String {
        guard let g = gaussians else { return "0 MB" }
        // Approx 60 bytes per splat (56 struct + 4 index)
        let bytes = Double(g.pointCount) * 60.0
        let mb = bytes / (1024.0 * 1024.0)
        return String(format: "%.1f MB", mb)
    }
    
    @Published var selectedStyle: ParallaxStyle = .orbit
    @Published var duration: Double = 3.0
    
    // Style & Grading
    @Published var exposure: Double = 0.0
    @Published var gamma: Double = 2.2
    @Published var vignetteStrength: Double = 0.5
    
    // MARK: - Camera Control
    enum CameraMode {
        case orbit
        case fly
        case cinema
    }
    
    @Published var cameraMode: CameraMode = .orbit
    
    // Universal Camera State (The Truth)
    // We compute the final View Matrix from these, but maintain "Mode State" for interactions
    
    // Orbit State
    @Published var orbitTarget: SIMD3<Double> = SIMD3(0, 0, 0) // Default target
    @Published var orbitTheta: Double = .pi
    @Published var orbitPhi: Double = 0
    @Published var orbitDistance: Double = 5.0
    // private var orbitCenter: SIMD3<Double> = SIMD3<Double>(0,0,0) // Dynamic center not yet used
    
    // Fly State
    @Published var flyEye: SIMD3<Double> = SIMD3<Double>(0, 0, -5)
    @Published var flyYaw: Double = 0
    @Published var flyPitch: Double = 0
    
    // Damping Targets (Internal State)
    private var targetOrbitTheta: Double = .pi
    private var targetOrbitPhi: Double = 0
    private var targetOrbitDistance: Double = 5.0
    
    private var targetFlyEye: SIMD3<Double> = SIMD3<Double>(0, 0, -5)
    private var targetFlyYaw: Double = 0
    private var targetFlyPitch: Double = 0
    
    // Smooth Time (Damping Factor)
    private let smoothTime: Double = 0.15
    
    // CINEMA STATE
    @Published var isPlaying: Bool = false
    @Published var playbackProgress: Double = 0.0
    
    // INPUT STATE
    var keysPressed: Set<UInt16> = []
    private var lastUpdateTime: TimeInterval = 0
    
    // Computed Final Camera for Renderer
    var camera: CameraPosition {
        // We output a struct that MetalSplatRenderer consumes.
        // We now have a 'target' property for true LookAt logic.
        
        switch cameraMode {
        case .orbit:
            // Spherical to Cartesian relative to Target
            // Theta = Yaw, Phi = Pitch.
            // Y is Up in our logical camera space.
            
            let x = orbitDistance * cos(orbitPhi) * sin(orbitTheta)
            let y = orbitDistance * sin(orbitPhi)
            let z = orbitDistance * cos(orbitPhi) * cos(orbitTheta)
            
            let eye = orbitTarget + SIMD3(x, y, z)
            
            return CameraPosition(
                x: eye.x, y: eye.y, z: eye.z,
                target: orbitTarget
            )
            
        case .fly:
             // Fly Mode: Eye is explicit. Center is derived from Pitch/Yaw.
             let front = SIMD3<Double>(
                sin(flyYaw) * cos(flyPitch),
                sin(flyPitch),
                cos(flyYaw) * cos(flyPitch)
             )
             let target = flyEye + front // Look forward
             
             return CameraPosition(
                x: flyEye.x, y: flyEye.y, z: flyEye.z,
                target: target
            )
             
        case .cinema:
             // ... legacy for now ...
             return CameraPosition()
        }
    }
    
    // New Input Handlers
    // Input Handling
    func handleDrag(delta: CGSize) {
        if cameraMode == .orbit {
            let sensitivity = 0.005
            targetOrbitTheta -= Double(delta.width) * sensitivity
            targetOrbitPhi   -= Double(delta.height) * sensitivity
            targetOrbitPhi = max(-.pi/2 + 0.1, min(.pi/2 - 0.1, targetOrbitPhi))
        } else if cameraMode == .fly {
            let sensitivity = 0.003
            targetFlyYaw -= Double(delta.width) * sensitivity
            targetFlyPitch -= Double(delta.height) * sensitivity
            targetFlyPitch = max(-.pi/2 + 0.1, min(.pi/2 - 0.1, targetFlyPitch))
        }
    }
    
    func handleScroll(delta: CGFloat) {
        if cameraMode == .orbit {
            let zoomSpeed = 0.01 * targetOrbitDistance
            targetOrbitDistance -= Double(delta) * zoomSpeed
            targetOrbitDistance = max(0.1, targetOrbitDistance)
        } else if cameraMode == .fly {
             let moveSpeed = 0.5
             let forward = SIMD3<Double>(sin(targetFlyYaw), 0, cos(targetFlyYaw))
             targetFlyEye += forward * Double(delta) * moveSpeed
        }
    }
    
    func handleKeyDown(code: UInt16) { keysPressed.insert(code) }
    func handleKeyUp(code: UInt16) { keysPressed.remove(code) }
    
    func updateCamera(time: TimeInterval) {
        let dt = time - lastUpdateTime
        lastUpdateTime = time
        guard dt < 0.1 else { return }
        
        // 1. Update Targets based on Input
        if cameraMode == .fly {
            updateFlyMovement(dt: dt)
        }
        
        // 2. Smooth Damping (Interpolate Current -> Target)
        // Lerp factor: 1 - exp(-lambda * dt) for frame-rate independence
        let lambda = 1.0 / smoothTime
        let t = 1.0 - exp(-lambda * dt)
        
        if cameraMode == .orbit {
            orbitTheta += (targetOrbitTheta - orbitTheta) * t
            orbitPhi += (targetOrbitPhi - orbitPhi) * t
            orbitDistance += (targetOrbitDistance - orbitDistance) * t
        } else if cameraMode == .fly {
            flyEye += (targetFlyEye - flyEye) * t
            flyYaw += (targetFlyYaw - flyYaw) * t
            flyPitch += (targetFlyPitch - flyPitch) * t
        }
    }
    
    private func updateFlyMovement(dt: Double) {
        var speed = 5.0
        if keysPressed.contains(56) { speed *= 5.0 } // Shift
        
        var move = SIMD3<Double>(0,0,0)
        
        // Use Target Yaw for movement direction to feel responsive
        let forward = SIMD3<Double>(sin(targetFlyYaw), 0, cos(targetFlyYaw))
        let right = SIMD3<Double>(cos(targetFlyYaw), 0, -sin(targetFlyYaw))
        let up = SIMD3<Double>(0, 1, 0)
        
        if keysPressed.contains(13) { move -= forward } // W
        if keysPressed.contains(1)  { move += forward } // S
        if keysPressed.contains(0)  { move -= right }   // A
        if keysPressed.contains(2)  { move += right }   // D
        if keysPressed.contains(14) { move += up }      // E
        if keysPressed.contains(12) { move -= up }      // Q
        
        if move != .zero {
            move = normalize(move)
            targetFlyEye += move * speed * dt
        }
    }
    
    // ... Legacy init and load methods keep as is ...
    
    let sharpService = SharpService()
    
    init() {
        checkAvailability()
    }
    
    func checkAvailability() {
        Task {
            let available = await sharpService.isAvailable()
            self.isAvailable = available
        }
    }
    
    func loadImage() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.image]
        
        if panel.runModal() == .OK, let url = panel.url {
            loadImage(url: url)
        }
    }
    
    func loadImage(url: URL) {
        if let image = NSImage(contentsOf: url) {
            self.selectedImage = image
            self.gaussians = nil
            self.errorMessage = nil
        } else {
            self.errorMessage = "Failed to load image from \(url.lastPathComponent)"
        }
    }
    

    
    // Unified Import
    func importFile(url: URL) {
        if url.pathExtension.lowercased() == "ply" || url.pathExtension.lowercased() == "splat" {
            loadSplat(url: url)
        } else {
            loadImage(url: url)
        }
    }
    
    func loadSplat(url: URL) {
        // Load .ply directly
        Task {
            // Handle security scope for regular files dropped onto the app
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }
            
            do {
                let data = try Data(contentsOf: url)
                let splat = try GaussianSplatData(data: data)
                
                // Dispatch UI updates
                await MainActor.run {
                    self.gaussians = splat
                    self.selectedImage = nil // clear image to indicate splat mode
                    self.isProcessing = false
                    self.errorMessage = nil
                    
                    self.focusOnSplat(splat)
                }
                    

            } catch {
                await MainActor.run {
                    self.errorMessage = "Failed to load splat: \(error.localizedDescription)"
                }
            }
        }
    }
    
    // Extracted Helper for Consistent Auto-Focus
    @MainActor
    private func focusOnSplat(_ splat: GaussianSplatData) {
        // Robust Auto-Focus: Use Mean and StdDev to ignore outliers
        var sum = SIMD3<Float>(0, 0, 0)
        var count = 0
        
        let step = max(1, splat.pointCount / 1000) // Sample 1000 points
        
        // 1. Calculate Mean
        for i in stride(from: 0, to: splat.pointCount, by: step) {
            sum += splat.positions[i]
            count += 1
        }
        let mean = sum / Float(count)
        
        // 2. Calculate StdDev (spread)
        var distSum: Float = 0
        for i in stride(from: 0, to: splat.pointCount, by: step) {
            let p = splat.positions[i]
            let d = distance(p, mean)
            distSum += d * d
        }
        let variance = distSum / Float(count)
        let stdDev = sqrt(variance)
        
        print("Sharp: Robust Auto-Focus - Mean: \(mean), StdDev: \(stdDev)")
        
        // Set Target to Mean (Center of mass)
        self.orbitTarget = SIMD3<Double>(Double(mean.x), Double(mean.y), Double(mean.z))
        
        // Set Distance to 3 * StdDev (Covers ~99% of normal distribution)
        // Ensure reasonable bounds (min 2.0)
        self.orbitDistance = max(2.0, Double(stdDev) * 3.0)
        self.targetOrbitDistance = self.orbitDistance
        
        self.targetOrbitTheta = .pi
        self.targetOrbitPhi = 0
        
        self.orbitTheta = self.targetOrbitTheta
        self.orbitPhi = self.targetOrbitPhi
        
        // Sync Fly Mode
        self.flyEye = self.orbitTarget + SIMD3<Double>(0, 0, self.orbitDistance)
        self.targetFlyEye = self.flyEye
        self.flyYaw = .pi
        self.flyPitch = 0
        self.targetFlyYaw = .pi
        self.targetFlyPitch = 0
    }
    
    func saveSplat(to url: URL) {
        guard let gaussians = gaussians else { return }
        do {
            try gaussians.plyData.write(to: url)
        } catch {
            self.errorMessage = "Failed to save splat: \(error.localizedDescription)"
        }
    }
    
    func generate3D() {
        guard let image = selectedImage else { return }
        
        isProcessing = true
        errorMessage = nil
        
        Task {
            do {
                let result = try await sharpService.generateGaussians(from: image)
                self.gaussians = result
                
                // CRITICAL: Apply Auto-Focus here too!
                self.focusOnSplat(result)
                
                self.isProcessing = false
            } catch {
                self.errorMessage = error.localizedDescription
                self.isProcessing = false
            }
        }
    }
    
    func updatePreview() {
        // No-op: Metal view updates automatically via @Published properties
    }
    
    func exportAnimation() {
        guard let image = selectedImage else { return }
        
        isProcessing = true
        errorMessage = nil
        
        Task {
            do {
                let frames = try await sharpService.generateParallaxAnimation(
                    from: image,
                    duration: duration,
                    style: selectedStyle
                )
                
                // For now, just show the last frame as "complete" 
                // In a real app, we'd save as GIF/MOV
                if let _ = frames.first {
                    // self.previewImage = first // Metal view handles preview
                }
                self.isProcessing = false
            } catch {
                self.errorMessage = error.localizedDescription
                self.isProcessing = false
            }
        }
    }
    
    // MARK: - Camera Logic
    
    // Helper to sync legacy sliders to orbit state

    private func cameraPositionForStyle(_ style: ParallaxStyle, at t: Double) -> CameraPosition {
        // t goes from 0 to 1
        let angle = t * 2 * .pi
        
        // Helper: Convert degrees to radians for output if needed (CameraPosition usually expects radians for trig, 
        // but let's check MetalSplatRenderer. It expects Radians in matrix_rotation).
        // The Service implementation returned values directly.
        
        switch style {
        case .orbit:
            // Circular orbit around center
            return CameraPosition(
                x: 0.2 * cos(angle),
                y: 0,
                z: 0.2 * sin(angle),
                rotationY: angle
            )
        case .dolly:
            // Move forward and back
            let z = 0.3 * sin(angle)
            return CameraPosition(x: 0, y: 0, z: z)
            
        case .horizontal:
            // Pan left to right
            let x = 0.3 * sin(angle)
            return CameraPosition(x: x, y: 0, z: 0)
            
        case .vertical:
            // Tilt up and down
            let y = 0.2 * sin(angle)
            return CameraPosition(x: 0, y: y, z: 0, rotationX: y * 0.5)
            
        case .kenBurns:
            // Slow zoom with slight pan
            let z = -t * 0.2  // Zoom in
            let x = t * 0.1 - 0.05
            return CameraPosition(x: x, y: 0, z: z)
        }
    }
    
    // Video Export
    @Published var isExporting = false
    @Published var exportProgress: Double = 0
    
    func exportVideo(url: URL, duration: Double = 5.0) async throws {
        guard let renderer = renderer else { return }
        
        isExporting = true
        exportProgress = 0
        
        // 1. Setup Exporter
        // Resolution: 1080p for high quality
        let size = CGSize(width: 1920, height: 1080)
        let exporter = try VideoExporter(url: url, size: size)
        try exporter.start(url: url)
        
        // 2. Offline Render Loop
        // We need to drive the renderer manually
        // 60 FPS
        let fps = 60
        let totalFrames = Int(duration * Double(fps))
        
        // Save current state
        let originalMode = cameraMode
        let originalSize = renderer.viewportSize 
        let originalAspect = renderer.aspectRatio
        
        // Force Renderer to 1080p aspect
        renderer.viewportSize = SIMD2<Float>(Float(size.width), Float(size.height))
        renderer.aspectRatio = Float(size.width / size.height)
        
        renderer.isCapturingFrame = true
        
        print("Starting Export: \(totalFrames) frames to \(url.lastPathComponent)")
        
        for i in 0..<totalFrames {
            // Update Progress
            await MainActor.run { self.exportProgress = Double(i) / Double(totalFrames) }
            
            // Time interpolation
            let t = Double(i) / Double(totalFrames - 1)
            
            // Set Camera for this frame
            // We reuse the Cinematic logic
            let cam = cameraPositionForStyle(selectedStyle, at: t)
            
            // Update Renderer Camera
            // We need to bypass the smooth damping "updateCamera" loop and set directly
            renderer.cameraPosition = cam
            
            // Force Draw
            // We need to trigger a draw. In MTKView, draw() is called by the loop.
            // But for offline, we want to wait for it.
            // Hack: We wait for the view's refresh rate? No, that's real-time.
            // Since we are on Main Thread (Async), we can just wait for a moment?
            // Correct way: Snapshot waits for lastCapturedTexture to update.
            // But we need to ensure the draw hook fires.
            // We can just rely on the fact that we set state, and the MTKView is running at 60fps.
            // We wait for a short duration to ensure a frame was produced.
            try await Task.sleep(nanoseconds: 20_000_000) // 20ms (1/50s)
            
            // Capture
            if let buffer = renderer.snapshot(at: size) {
                let time = Double(i) / Double(fps)
                exporter.append(pixelBuffer: buffer, at: time)
            }
        }
        
        await exporter.finish()
        
        // Restore State
        renderer.isCapturingFrame = false
        renderer.viewportSize = originalSize
        renderer.aspectRatio = originalAspect
        cameraMode = originalMode
        isExporting = false
        exportProgress = 1.0
        
        print("Export Complete")
    }
    


}
