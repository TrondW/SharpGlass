import SwiftUI
import AppKit
import simd

@MainActor
public class SharpViewModel: ObservableObject {
    @Published var selectedImage: NSImage?
    @Published var selectedImageURL: URL?
    @Published var isProcessing = false
    @Published var isAvailable = false
    @Published var gaussians: GaussianSplatData?
    
    // Generation Settings
    @Published var cleanBackground = false  // Disabled - Vision AI not reliable for all images
    @Published var colorMode: ColorMode = .standard
    @Published var isHolographicModeEnabled: Bool = false {
        didSet {
            if isHolographicModeEnabled {
                headTrackingManager.start()
            } else {
                headTrackingManager.stop()
            }
        }
    }
    @Published var headPosition: SIMD3<Float> = SIMD3<Float>(0, 0, 60) // cm
    private var targetHeadPosition: SIMD3<Float> = SIMD3<Float>(0, 0, 60)
    
    private let headTrackingManager = HeadTrackingManager()

    // localized strings
    private enum Constants {
        static let installStarting = "Starting..."
        static let installComplete = "Installation Complete"
        static let installFailed = "Failed"
    }

    public enum ColorMode {
        case standard
        case filmic
    }

    @Published var errorMessage: String?
    
    // Setup State
    public enum SetupStatus {
        case notStarted
        case installing
        case complete
        case failed
    }
    @Published var setupStatus: SetupStatus = .notStarted
    @Published var setupProgress: Double = 0.0
    @Published var setupStep: String = ""
    
    public func installBackend() {
        guard setupStatus != .installing else { return }
        
        setupStatus = .installing
        setupProgress = 0.0
        setupStep = Constants.installStarting
        errorMessage = nil
        
        Task {
            do {
                try await sharpService.setupBackend { [weak self] step, progress in
                    Task { @MainActor in
                        self?.setupStep = step
                        self?.setupProgress = progress
                    }
                }
                
                self.setupStatus = .complete
                self.setupStep = Constants.installComplete
                self.checkAvailability()
                
            } catch {
                self.setupStatus = .failed
                self.errorMessage = error.localizedDescription
                self.setupStep = Constants.installFailed
            }
        }
    }
    
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
    

    // Style & Grading (Defaults match source image)
    @Published var exposure: Double = 0.0
    @Published var gamma: Double = 2.2  // Match sRGB source
    @Published var vignetteStrength: Double = 0.0  // No vignette by default
    @Published var splatScale: Double = 1.0
    @Published var saturation: Double = 1.0
    
    /// Resets all grading parameters to values that best match the original source image.
    public func resetToMatchSource() {
        self.exposure = 0.0
        self.gamma = 2.2  // Linearize sRGB splat data for the extendedLinearSRGB buffer
        self.saturation = 1.0
        self.vignetteStrength = 0.0
        self.colorMode = .standard
        self.splatScale = 1.0
    }
    
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
    @Published var orbitTarget: SIMD3<Double> = SIMD3(0, 0, 0)
    @Published var isNavigating: Bool = false
    @Published var orbitTheta: Double = .pi  // Start facing front of splat
    @Published var orbitPhi: Double = 0
    @Published var orbitDistance: Double = 5.0
    // private var orbitCenter: SIMD3<Double> = SIMD3<Double>(0,0,0) // Dynamic center not yet used
    
    // Fly State
    @Published var flyEye: SIMD3<Double> = SIMD3<Double>(0, 0, -5)
    @Published var flyYaw: Double = 0
    @Published var flyPitch: Double = 0
    
    // Damping Targets (Internal State)
    var targetOrbitTarget: SIMD3<Double> = SIMD3(0, 0, 0)
    var targetOrbitTheta: Double = .pi  // Start facing front of splat
    var targetOrbitPhi: Double = 0
    var targetOrbitDistance: Double = 5.0
    
    var targetFlyEye: SIMD3<Double> = SIMD3<Double>(0, 0, -5)
    var targetFlyYaw: Double = 0
    var targetFlyPitch: Double = 0
    
    // Smooth Time (Damping Factor)
    private let smoothTime: Double = 0.1 // snappier feel (was 0.15)
    
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
             // Forward vector in RHS: +Z is Backwards, so -Z is Forward.
             let front = SIMD3<Double>(
                sin(flyYaw) * cos(flyPitch),
                sin(flyPitch),
                -cos(flyYaw) * cos(flyPitch)
             )
             let target = flyEye + front // Look forward
             
             return CameraPosition(
                x: flyEye.x, y: flyEye.y, z: flyEye.z,
                target: target
            )
             
        case .cinema:
             // Cinema Mode: Slow Spin-table
             let spinRate = 0.5
             let angle = playbackProgress * 2.0 * .pi * spinRate
             
             let x = orbitDistance * cos(orbitPhi) * sin(angle)
             let y = orbitDistance * sin(orbitPhi)
             let z = orbitDistance * cos(orbitPhi) * cos(angle)
             
             let eye = orbitTarget + SIMD3(x, y, z)
             
             return CameraPosition(
                 x: eye.x, y: eye.y, z: eye.z,
                 target: orbitTarget
             )
        }
    }
    
    // New Input Handlers
    // Input Handling
    /// Processes mouse drag events to update the camera's target orientation or position.
    /// - Parameters:
    ///   - delta: movement in screen coordinates.
    ///   - button: 0=LMB, 1=RMB, 2=MMB.
    ///   - modifiers: NSEvent modifier flags (e.g., .option for Alt).
    func handleDrag(delta: CGSize, button: Int, modifiers: NSEvent.ModifierFlags) {
        let isAlt = modifiers.contains(.option)
        let isShift = modifiers.contains(.shift)
        
        // 1. Orbit: Alt + LMB (0) OR MMB (2) OR simple LMB (0) if in Orbit mode
        let shouldOrbit = (isAlt && button == 0) || (button == 2 && !isAlt && !isShift) || (button == 0 && cameraMode == .orbit)
        
        if shouldOrbit {
            if cameraMode == .orbit {
                let sensitivity = 0.005
                targetOrbitTheta -= Double(delta.width) * sensitivity
                targetOrbitPhi   -= Double(delta.height) * sensitivity
                targetOrbitPhi = max(-.pi/2 + 0.1, min(.pi/2 - 0.1, targetOrbitPhi))
            } else if cameraMode == .fly {
                let sensitivity = 0.003
                targetFlyYaw -= Double(delta.width) * sensitivity // Refined direction
                targetFlyPitch -= Double(delta.height) * sensitivity
                targetFlyPitch = max(-.pi/2 + 0.1, min(.pi/2 - 0.1, targetFlyPitch))
            }
        }
        // 2. Pan: Alt + MMB (2) OR Shift + MMB (2)
        else if (button == 2 && (isAlt || isShift)) {
            let sensitivity = 0.01 * orbitDistance
            let right = SIMD3<Double>(cos(orbitTheta), 0, -sin(orbitTheta))
            let up = SIMD3<Double>(0, 1, 0)
            targetOrbitTarget += right * Double(-delta.width) * sensitivity
            targetOrbitTarget += up * Double(delta.height) * sensitivity
        }
        // 3. Zoom: Alt + RMB (1) OR simple RMB (1) in Orbit Mode
        else if (button == 1) {
            let zoomSpeed = 0.04 * targetOrbitDistance
            targetOrbitDistance += Double(delta.height) * zoomSpeed
            targetOrbitDistance = max(0.01, targetOrbitDistance)  // Allow very close zoom
        }
    }
    
    /// Processes scroll events for zooming (Orbit) or forward/back movement (Fly).
    /// When Option key is held, zooms toward/away from the mouse pointer position.
    func handleScroll(delta: CGFloat, position: CGPoint, modifiers: NSEvent.ModifierFlags) {
        if cameraMode == .orbit {
            let zoomSpeed = 0.05 * targetOrbitDistance
            let oldDistance = targetOrbitDistance
            targetOrbitDistance -= Double(delta) * zoomSpeed
            targetOrbitDistance = max(0.01, targetOrbitDistance)
            
            // Option + scroll = zoom toward pointer
            if modifiers.contains(.option) {
                // Shift orbit target toward the pointer direction
                // Convert normalized position to offset from center (-1 to 1)
                let offsetX = (position.x - 0.5) * 2.0
                let offsetY = (position.y - 0.5) * 2.0
                
                // Calculate shift amount based on zoom delta
                let shiftFactor = (oldDistance - targetOrbitDistance) * 0.3
                
                // Get camera right and up vectors from theta
                let right = SIMD3<Double>(cos(orbitTheta), 0, sin(orbitTheta))
                let up = SIMD3<Double>(0, 1, 0)
                
                // Shift target toward pointer
                targetOrbitTarget += right * Double(offsetX) * shiftFactor
                targetOrbitTarget += up * Double(offsetY) * shiftFactor
            }
        } else if cameraMode == .fly {
             let moveSpeed = 0.5
             let forward = SIMD3<Double>(sin(targetFlyYaw), 0, cos(targetFlyYaw))
             targetFlyEye += forward * Double(delta) * moveSpeed
        }
    }
    
    func handleKeyDown(code: UInt16) { 
        keysPressed.insert(code) 
        
        // 'F' Key for Focus
        if code == 3 { // 'f'
            focusOnScene()
        }
    }
    
    func focusOnScene() {
        // Reset to world center and sensible distance
        targetOrbitTarget = SIMD3(0, 0, 0)
        targetOrbitDistance = 5.0
        targetOrbitTheta = .pi
        targetOrbitPhi = 0
    }
    
    func snapCamera(theta: Double, phi: Double, distance: Double) {
        targetOrbitTheta = theta
        targetOrbitPhi = phi
        targetOrbitDistance = distance
    }
    func handleKeyUp(code: UInt16) { keysPressed.remove(code) }
    
    /// Main camera update loop called by the display link.
    /// Handles smooth damping of all camera parameters (Orbit and Fly).
    func updateCamera(time: TimeInterval) {
        let dt = time - lastUpdateTime
        lastUpdateTime = time
        guard dt < 0.1 else { return }
        
        // 1. Update Targets based on Input
        if cameraMode == .fly {
            updateFlyMovement(dt: dt)
        } else if cameraMode == .cinema {
            // Drive playback at a constant rate only when playing
            if isPlaying {
                playbackProgress += dt * 0.1
                if playbackProgress > 1.0 { playbackProgress -= 1.0 }
            }
        }
        
        // 2. Smooth Damping (Interpolate Current -> Target)
        // Lerp factor: 1 - exp(-lambda * dt) for frame-rate independence
        let lambda = 1.0 / smoothTime
        let t = 1.0 - exp(-lambda * dt)
        
        if cameraMode == .orbit {
            orbitTarget += (targetOrbitTarget - orbitTarget) * t
            orbitTheta += (targetOrbitTheta - orbitTheta) * t
            orbitPhi += (targetOrbitPhi - orbitPhi) * t
            orbitDistance += (targetOrbitDistance - orbitDistance) * t
        } else if cameraMode == .fly {
            flyEye += (targetFlyEye - flyEye) * t
            flyYaw += (targetFlyYaw - flyYaw) * t
            flyPitch += (targetFlyPitch - flyPitch) * t
        }
        
        // --- Smooth Head Position for Holographic ---
        // XY smoothing (Fast for parallax response)
        let headK_xy: Float = 10.0
        let ht_xy = 1.0 - exp(-headK_xy * Float(dt))
        headPosition.x += (targetHeadPosition.x - headPosition.x) * ht_xy
        headPosition.y += (targetHeadPosition.y - headPosition.y) * ht_xy
        
        // Z smoothing (Slow to dampen phantom zoom)
        let headK_z: Float = 2.0
        let ht_z = 1.0 - exp(-headK_z * Float(dt))
        headPosition.z += (targetHeadPosition.z - headPosition.z) * ht_z
    }
    
    private func updateFlyMovement(dt: Double) {
        var speed = 5.0
        if keysPressed.contains(56) { speed *= 5.0 } // Shift
        
        var move = SIMD3<Double>(0,0,0)
        
        // Forward matches the camera's look direction (toward -Z when yaw=pi)
        let forward = SIMD3<Double>(
            sin(targetFlyYaw) * cos(targetFlyPitch),
            sin(targetFlyPitch),
            -cos(targetFlyYaw) * cos(targetFlyPitch)
        )
        let right = SIMD3<Double>(cos(targetFlyYaw), 0, sin(targetFlyYaw))
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
    
    let sharpService: SharpServiceProtocol
    
    public init(service: SharpServiceProtocol = SharpService()) {
        self.sharpService = service
        checkAvailability()
        
        headTrackingManager.onHeadPositionUpdate = { [weak self] pos in
            // Lock Z to 60cm to prevent phantom zoom artifacts as requested.
            // XY tracking remains active for parallax.
            self?.targetHeadPosition = SIMD3<Float>(pos.x, pos.y, 60.0)
        }
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
            self.selectedImageURL = url
            self.gaussians = nil
            self.errorMessage = nil
            
            // Auto-start processing
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 300_000_000) // 300ms delay for drag completion + UI update
                self.generate3D()
            }
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
                var splat = try GaussianSplatData(data: data)
                
                // Auto-prune if exceeds GPU memory limits
                let maxSplatCount = 2_000_000
                if splat.pointCount > maxSplatCount {
                    print("Sharp: ⚠️ Loaded splat exceeds safe limit (\(splat.pointCount) > \(maxSplatCount))")
                    splat = splat.pruned(maxCount: maxSplatCount)
                }
                
                // Dispatch UI updates
                await MainActor.run {
                    var finalSplat = splat
                    finalSplat.evictRawPLYData() // Free up raw data memory
                    
                    self.gaussians = finalSplat
                    self.selectedImage = nil // clear image to indicate splat mode
                    self.isProcessing = false
                    self.errorMessage = nil
                    
                    self.focusOnSplat(finalSplat)
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
        
        // Set Distance based on splat spread - match input photo perspective
        self.orbitDistance = max(0.5, Double(stdDev) * 1.2)
        self.targetOrbitDistance = self.orbitDistance
        self.targetOrbitTarget = self.orbitTarget
        
        // Initial camera matches the input photo view:
        // Camera on -Z axis looking toward +Z (where splat is)
        self.targetOrbitTheta = .pi
        self.targetOrbitPhi = 0
        
        self.orbitTheta = self.targetOrbitTheta
        self.orbitPhi = self.targetOrbitPhi
        
        // Sync Fly Mode - camera at origin looking toward splat
        self.targetFlyEye = SIMD3<Double>(0, 0, 0)
        self.flyEye = self.targetFlyEye
        // Calculate yaw to look at target
        let dx = self.orbitTarget.x
        let dz = self.orbitTarget.z
        self.targetFlyYaw = atan2(dx, dz)
        self.flyYaw = self.targetFlyYaw
        self.targetFlyPitch = 0
        self.flyPitch = 0
    }
    
    func saveSplat(to url: URL) {
        guard let gaussians = gaussians, let data = gaussians.plyData else { 
            self.errorMessage = "No PLY data available to save"
            return 
        }
        do {
            try data.write(to: url)
        } catch {
            self.errorMessage = "Failed to save splat: \(error.localizedDescription)"
        }
    }
    
    func generate3D() {
        guard let image = selectedImage else { return }
        
        isProcessing = true
        errorMessage = nil
        
        Task {
            // Give UI time to update and show the loading overlay
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
            
            do {
                let result = try await sharpService.generateGaussians(
                    from: image,
                    originalURL: selectedImageURL,
                    cleanBackground: cleanBackground
                )
                
                var finalSplat = result
                finalSplat.evictRawPLYData() // Free memory from PLY parsing
                
                self.gaussians = finalSplat
                self.selectedImage = nil // Free up source image memory
                
                // CRITICAL: Apply Auto-Focus here too!
                self.focusOnSplat(finalSplat)
                
                self.isProcessing = false
                
                // Cleanup intermediate files
                sharpService.cleanup()
            } catch {
                self.errorMessage = error.localizedDescription
                self.isProcessing = false
                
                // Show error alert to user
                showErrorAlert(
                    title: "3D Generation Failed",
                    message: error.localizedDescription
                )
            }
        }
    }
    
    /// Show a native macOS alert dialog
    @MainActor
    private func showErrorAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
    
    func updatePreview() {
        // No-op: Metal view updates automatically via @Published properties
    }
    
    


}
