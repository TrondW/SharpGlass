import Foundation
import Vision
@preconcurrency import AVFoundation
import CoreMedia
import AppKit
import simd

/// Manages real-time face tracking using the Vision framework to drive a holographic "off-axis" projection.
/// Maps physical head position to virtual camera offsets to create a window-like depth effect.
class HeadTrackingManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "session.queue")
    private let processingQueue = DispatchQueue(label: "head.tracking.processing", qos: .userInteractive)
    
    // Throttling State
    private var lastFrameTime: TimeInterval = 0
    private let minFrameInterval: TimeInterval = 0.033 // Cap at ~30 FPS. Smoother tracking on CPU.
    
    // Vision Request Reuse
    private lazy var faceRequest: VNDetectFaceLandmarksRequest = {
        let req = VNDetectFaceLandmarksRequest { [weak self] request, error in
            guard let self = self,
                  let results = request.results as? [VNFaceObservation],
                  let face = results.first else { return }
            self.processFace(face)
        }
        #if !targetEnvironment(simulator)
        req.usesCPUOnly = true // CRITICAL: Prevent GPU contention with Metal Splatter
        #endif
        return req
    }()
    
    // Callback is thread-safe for the consumer
    var onHeadPositionUpdate: (@MainActor (SIMD3<Float>) -> Void)?
    
    private var isStarted = false
    
    func start() {
        guard !isStarted else { return }
        isStarted = true
        
        #if DEBUG
        print("⚠️ HeadTracking: Running in DEBUG mode. Performance may be degraded. Use Release for optimal 60fps splatting.")
        #endif
        
        // Add Observers for Stability
        NotificationCenter.default.addObserver(self, selector: #selector(sessionRuntimeError), name: .AVCaptureSessionRuntimeError, object: captureSession)
        NotificationCenter.default.addObserver(self, selector: #selector(sessionWasInterrupted), name: .AVCaptureSessionWasInterrupted, object: captureSession)
        NotificationCenter.default.addObserver(self, selector: #selector(sessionInterruptionEnded), name: .AVCaptureSessionInterruptionEnded, object: captureSession)
        
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                // Priority: Continuity Camera -> Built-in Wide -> Default
                var device: AVCaptureDevice?
                if let continuity = AVCaptureDevice.default(.continuityCamera, for: .video, position: .unspecified) {
                    device = continuity
                    print("HeadTracking: 📱 Using Continuity Camera: \(continuity.localizedName)")
                } else if let builtIn = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) {
                    device = builtIn
                    print("HeadTracking: 💻 Using Built-in Camera: \(builtIn.localizedName)")
                } else {
                    device = AVCaptureDevice.default(for: .video)
                    print("HeadTracking: 📷 Using Default Camera: \(device?.localizedName ?? "Unknown")")
                }
                
                guard let finalDevice = device else {
                    print("HeadTracking: ❌ No suitable camera found")
                    return
                }
                
                // Clear existing inputs if restarting
                self.captureSession.inputs.forEach { self.captureSession.removeInput($0) }
                self.captureSession.outputs.forEach { self.captureSession.removeOutput($0) }
                
                let input = try AVCaptureDeviceInput(device: finalDevice)
                
                self.captureSession.beginConfiguration()
                
                // OPTIMIZATION: Use lower resolution. Face landmarks don't need 1080p/4K.
                self.captureSession.sessionPreset = .vga640x480 
                
                if self.captureSession.canAddInput(input) {
                    self.captureSession.addInput(input)
                }
                
                self.videoOutput.setSampleBufferDelegate(self, queue: self.processingQueue)
                self.videoOutput.alwaysDiscardsLateVideoFrames = true // Drop frames if busy
                
                if self.captureSession.canAddOutput(self.videoOutput) {
                    self.captureSession.addOutput(self.videoOutput)
                }
                self.captureSession.commitConfiguration()
                
                self.captureSession.startRunning()
            } catch {
                print("HeadTracking: Failed to start session: \(error)")
            }
        }
    }
    
    @objc func sessionRuntimeError(notification: NSNotification) {
        guard let error = notification.userInfo?[AVCaptureSessionErrorKey] as? AVError else { return }
        print("HeadTracking: ⚠️ Session Runtime Error: \(error.localizedDescription)")
        
        // Auto-restart if media services were reset
        if error.code == .mediaServicesWereReset {
            sessionQueue.async { [weak self] in
                self?.captureSession.startRunning()
            }
        }
    }
    
    @objc func sessionWasInterrupted(notification: NSNotification) {
        print("HeadTracking: ⏸️ Session Interrupted")
    }
    
    @objc func sessionInterruptionEnded(notification: NSNotification) {
        print("HeadTracking: ▶️ Session Interruption Ended")
        sessionQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
    }
    
    func stop() {
        guard isStarted else { return }
        isStarted = false
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Throttling: Ensure we don't process more than 10 FPS
        let now = Date().timeIntervalSince1970
        guard now - lastFrameTime >= minFrameInterval else { return }
        lastFrameTime = now
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Use .up orientation (raw sensor). Mirroring and mapping handled in processFace
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        
        try? handler.perform([self.faceRequest])
    }
    
    // MARK: - Face Processing
    
    /// Processes a face observation to estimate the user's head position in screen-space centimeters.
    ///
    /// The mapping follows these principles:
    /// 1. **Stability**: Uses fixed-depth baseline and eccentricity correction to eliminate phantom zoom.
    /// 2. **Orientation**: Aligns with Mac camera mirroring and the renderer's Y-down coordinate system.
    /// 3. **Scale**: Maps normalized [0,1] coordinates to physical centimeters for realistic parallax.
    private func processFace(_ face: VNFaceObservation) {
        let box = face.boundingBox // normalized [0, 1], origin is bottom-left
        
        // --- Geometric Eccentricity Correction ---
        // As the head moves away from the camera axis, the observed face appears to shrink by cos²(θ).
        // This math compensates for that perspective shrinkage to maintain a stable depth estimate.
        let hx_norm = 0.5 - Float(box.midX)
        let hy_norm = 0.5 - Float(box.midY)
        
        // Estimate angular offset (Mac camera FOV is approx 60-70 deg)
        let angle_x = hx_norm * 0.52 * 2.0 
        let angle_y = hy_norm * 0.40 * 2.0
        let cos_ecc = cos(angle_x) * cos(angle_y)
        
        // Correct the perceived width to be "axis-aligned"
        let faceWidthCorrected = Float(box.width) / (cos_ecc * cos_ecc + 0.0001)
        
        // Calibrated Heuristic: z (cm) = C / w_corrected
        // At 60cm, corrected box.width is typically ~0.24.
        var z_val = 14.5 / (Double(faceWidthCorrected) + 0.0001)
        z_val = max(15.0, min(120.0, z_val))
        
        // --- Output Coordinate Mapping ---
        // Sensitivity 1.5x provides a responsive but stable depth portal feel.
        // X+: Head right -> moves camera right -> shifts world left.
        // Y+: Head down -> moves camera down -> shifts world up.
        let SCREEN_WIDTH_CM: Float = 30.0
        let SCREEN_HEIGHT_CM: Float = 20.0
        let SENSITIVITY: Float = 1.5
        
        let xCM = hx_norm * SCREEN_WIDTH_CM * SENSITIVITY
        let yCM = hy_norm * SCREEN_HEIGHT_CM * SENSITIVITY * 0.8
        
        let pos = SIMD3<Float>(xCM, yCM, Float(z_val))
        
        Task { @MainActor in
            self.onHeadPositionUpdate?(pos)
        }
    }
}

extension HeadTrackingManager: @unchecked Sendable {}
