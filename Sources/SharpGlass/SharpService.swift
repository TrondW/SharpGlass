import Foundation
import AppKit
import Metal
import MetalKit

// MARK: - Sharp View Synthesis Service

/// Service for Apple's ml-sharp 3D view synthesis
/// ml-sharp generates 3D Gaussian splats from single images for novel view rendering
/// 
/// Prerequisites:
/// - Python 3.13+ with ml-sharp installed: pip install -r requirements.txt
/// - Model checkpoint (auto-downloaded on first run)
///
/// Usage:
/// 1. Call generateGaussians() to create 3DGS from image
/// 2. Use renderNovelView() to render from different camera angles
/// 3. Export as animated GIF or video with exportAnimation()

@MainActor
protocol SharpServiceProtocol {
    func isAvailable() async -> Bool
    func generateGaussians(from image: NSImage) async throws -> GaussianSplatData
    func renderNovelView(_ splats: GaussianSplatData, cameraPosition: CameraPosition) async throws -> NSImage
    func generateParallaxAnimation(from image: NSImage, duration: Double, style: ParallaxStyle) async throws -> [NSImage]
}

// MARK: - Data Types

struct CameraPosition {
    var x: Double = 0  // Left/Right offset (or Eye X)
    var y: Double = 0  // Up/Down offset (or Eye Y)
    var z: Double = 0  // Forward/Back offset (or Eye Z)
    var rotationX: Double = 0  // Pitch
    var rotationY: Double = 0  // Yaw
    
    // Optional look-at target for Arcball/LookAt cameras
    var target: SIMD3<Double>? = nil
    
    static let center = CameraPosition()
}

enum ParallaxStyle: String, CaseIterable, Identifiable {
    case orbit = "Orbit"
    case dolly = "Dolly Zoom"
    case horizontal = "Pan Left-Right"
    case vertical = "Tilt Up-Down"
    case kenBurns = "Ken Burns"
    
    var id: String { rawValue }
    
    var icon: String {
        switch self {
        case .orbit: return "arrow.triangle.2.circlepath"
        case .dolly: return "arrow.up.and.down.and.arrow.left.and.right"
        case .horizontal: return "arrow.left.arrow.right"
        case .vertical: return "arrow.up.arrow.down"
        case .kenBurns: return "camera.viewfinder"
        }
    }
}

/// Represents 3D Gaussian Splat data
struct GaussianSplatData: Identifiable {
    let id = UUID()
    let plyData: Data  // Raw PLY file data
    let positions: [SIMD3<Float>]  // Gaussian centers
    let colors: [SIMD3<Float>]  // RGB colors
    let opacities: [Float]  // Alpha values
    let scales: [SIMD3<Float>]  // Log-scales (converted to exp)
    let rotations: [SIMD4<Float>] // Quaternions
    let shs: [Float] // Spherical Harmonics coefficients (45 per splat)
    let pointCount: Int
    
    @MainActor
    init(plyPath: URL) throws {
        guard let data = try? Data(contentsOf: plyPath) else {
            throw SharpServiceError.failedToLoadPLY
        }
        self.plyData = data
        
        // Parse PLY header and data
        // Parse PLY header and data
        (self.positions, self.colors, self.opacities, self.scales, self.rotations, self.shs) = try GaussianSplatData.parsePLY(data)
        self.pointCount = positions.count
    }
    
    @MainActor
    init(data: Data) throws {
        self.plyData = data
        
        // Parse PLY header and data
        // Parse PLY header and data
        (self.positions, self.colors, self.opacities, self.scales, self.rotations, self.shs) = try GaussianSplatData.parsePLY(data)
        self.pointCount = positions.count
    }
    
    @MainActor
    private static func parsePLY(_ data: Data) throws -> ([SIMD3<Float>], [SIMD3<Float>], [Float], [SIMD3<Float>], [SIMD4<Float>], [Float]) {
        // Robust PLY parser for 3D Gaussian Splat format
        // Supports binary_little_endian 1.0 and identifies 3DGS fields
        
        print("Sharp: Starting PLY parse, data size: \(data.count) bytes")
        
        // Find end_header with different possible line endings
        let endHeaderPatterns = ["end_header\n", "end_header\r\n", "end_header"]
        var headerRange: Range<Data.Index>? = nil
        
        for pattern in endHeaderPatterns {
            if let range = data.range(of: pattern.data(using: .utf8)!) {
                headerRange = range
                break
            }
        }
        
        guard let finalHeaderRange = headerRange else {
            print("Sharp Error: Could not find end_header")
            throw SharpServiceError.invalidPLYFormat
        }
        
        let headerData = data.subdata(in: 0..<finalHeaderRange.upperBound)
        guard let header = String(data: headerData, encoding: .utf8) else {
            print("Sharp Error: Could not decode header as UTF8")
            throw SharpServiceError.invalidPLYFormat
        }
        
        print("Sharp: Header found (\(headerData.count) bytes)")
        
        let lines = header.components(separatedBy: .newlines)
        var vertexCount = 0
        var isBinary = false
        var properties: [(name: String, type: String)] = []
        
        for line in lines {
            let parts = line.trimmingCharacters(in: .whitespaces).components(separatedBy: " ")
            if parts.contains("binary_little_endian") {
                isBinary = true
            } else if line.hasPrefix("element vertex") {
                if parts.count >= 3, let count = Int(parts[2]) {
                    vertexCount = count
                }
            } else if line.hasPrefix("property") {
                if parts.count >= 3 {
                    properties.append((name: parts[2], type: parts[1]))
                }
            }
        }
        
        print("Sharp: Vertex count: \(vertexCount), Binary: \(isBinary), Properties: \(properties.count)")
        
        if isBinary {
            return try parseBinaryPLY(data.subdata(in: finalHeaderRange.upperBound..<data.count), vertexCount: vertexCount, properties: properties)
        } else {
            return try parseAsciiPLY(header, data: data.subdata(in: finalHeaderRange.upperBound..<data.count), vertexCount: vertexCount, properties: properties)
        }
    }
    
    @MainActor
    private static func parseBinaryPLY(_ data: Data, vertexCount: Int, properties: [(name: String, type: String)]) throws -> ([SIMD3<Float>], [SIMD3<Float>], [Float], [SIMD3<Float>], [SIMD4<Float>], [Float]) {
        print("Sharp: Parsing binary data (\(data.count) bytes)")
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD3<Float>] = []
        var opacities: [Float] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        var shs: [Float] = []
        
        // Pre-allocate for performance
        let shCount = 15 * 3 // 45 coeffs
        let estimatedTotalFloats = vertexCount * shCount
        shs.reserveCapacity(estimatedTotalFloats)
        positions.reserveCapacity(vertexCount)
        colors.reserveCapacity(vertexCount)
        opacities.reserveCapacity(vertexCount)
        scales.reserveCapacity(vertexCount)
        rotations.reserveCapacity(vertexCount)
        
        // Find offsets for critical 3DGS properties
        let xIdx = properties.firstIndex { $0.name == "x" } ?? -1
        let yIdx = properties.firstIndex { $0.name == "y" } ?? -1
        let zIdx = properties.firstIndex { $0.name == "z" } ?? -1
        let redIdx = properties.firstIndex { $0.name == "red" || $0.name == "f_dc_0" } ?? -1
        let greenIdx = properties.firstIndex { $0.name == "green" || $0.name == "f_dc_1" } ?? -1
        let blueIdx = properties.firstIndex { $0.name == "blue" || $0.name == "f_dc_2" } ?? -1
        let opacityIdx = properties.firstIndex { $0.name == "opacity" } ?? -1
        let scale0Idx = properties.firstIndex { $0.name == "scale_0" } ?? -1
        let rot0Idx = properties.firstIndex { $0.name == "rot_0" } ?? -1
        
        // SH Indices (f_rest_0 to f_rest_44)
        var shIndices: [Int] = []
        for i in 0..<shCount {
            if let idx = properties.firstIndex(where: { $0.name == "f_rest_\(i)" }) {
                shIndices.append(idx)
            } else {
                // If any are missing, we might assume NO SH data, or partial?
                // Standard 3DGS has all or nothing.
                break
            }
        }
        let hasSH = shIndices.count == shCount
        
        print("Sharp: Property indices - XYZ: (\(xIdx),\(yIdx),\(zIdx)), RGB: (\(redIdx),\(greenIdx),\(blueIdx)), Opacity: \(opacityIdx), Rot: \(rot0Idx), Has SH: \(hasSH)")
        
        // Calculate stride
        var stride = 0
        var propertyOffsets: [Int] = []
        print("Sharp: Property List:")
        for (idx, prop) in properties.enumerated() {
            print("  [\(idx)] \(prop.name) (\(prop.type))")
            propertyOffsets.append(stride)
            let size: Int
            switch prop.type.lowercased() {
            case "float", "float32": size = 4
            case "double", "float64": size = 8
            case "uchar", "uint8", "char", "int8": size = 1
            case "short", "uint16", "int16": size = 2
            case "int", "uint", "int32", "uint32": size = 4
            default: size = 4
            }
            stride += size
        }
        
        print("Sharp: Stride calculated from header: \(stride) bytes")
        
        // Fix for padded/mismatched stride (ml-sharp metadata)
        let actualStride = data.count / vertexCount
        if actualStride < stride {
            print("Sharp Warning: Actual stride (\(actualStride)) < Header stride (\(stride)). Recalculating offsets...")
            
            // "Packed Heuristic":
            // Check if the actual stride matches the size of "Essential" properties only.
            // Essentials: XYZ (12), RGB/DC (12), Opacity (4), Scale (12), Rot (16) -> 56 bytes.
            // If actual == 56, we assume ONLY these properties are present, regardless of header order.
            // We iterate header properties. If prop is Essential, we assign offset. If not, we skip.
            
            // Check if actual stride matches a "known packed size" (e.g. 56 bytes for standard minimal)
            // Or just iterate and see if we can fit strict essentials.
            
            stride = 0
            propertyOffsets = []
            
            // HEURISTIC: If actual stride is exactly 56 bytes (14 floats), we assume it's the standard minimal set.
            // We greedily match only the critical 14 properties in appearance order?
            // Or we assume the header order is preserved but non-essentials are dropped?
            // Let's rely on indices we found earlier.
            
            let criticalIndices = [xIdx, yIdx, zIdx, redIdx, greenIdx, blueIdx, opacityIdx, scale0Idx, scale0Idx+1, scale0Idx+2, rot0Idx, rot0Idx+1, rot0Idx+2, rot0Idx+3].filter { $0 != -1 }
            let criticalSet = Set(criticalIndices)
            
            // Re-map offsets
            for (idx, prop) in properties.enumerated() {
                // Determine size
                let size: Int
                switch prop.type.lowercased() {
                case "float", "float32": size = 4
                case "double", "float64": size = 8
                case "uchar", "uint8", "char", "int8": size = 1
                case "short", "uint16", "int16": size = 2
                case "int", "uint", "int32", "uint32": size = 4
                default: size = 4
                }
                
                // DECISION: Should we include this property in the Packed Layout?
                // If we are in "Missing Data" mode:
                // 1. Is it a critical property? YES -> Include.
                // 2. Is it SH? YES/NO?
                // 3. Is it "Normals" or "Extra"?
                
                // If actual stride is small (56), we probably only have criticals.
                let isCritical = criticalSet.contains(idx)
                
                // If we have room, we add it. But priority is Criticals.
                // If we encounter a Non-Critical property, and we are "squeezed", we skip it.
                // How do we define "squeezed"?
                // If (Current Stride + Size) > Actual Stride, we definitely stop.
                // But what if Non-Critical (nx) is early?
                // If we iterate header, and nx is before Value, and we include nx, we might push Value out of bounds.
                
                // Heuristic: If ActualStride == 56 (Fast Path for Standard Compact)
                // We ONLY increment stride for properties that are in 'criticalSet'.
                
                var included = false
                if actualStride == 56 {
                    if isCritical {
                        included = true
                    }
                } else {
                    // Fallback to "Truncation" logic (keep adding until full)
                    if stride + size <= actualStride {
                        included = true
                    }
                }
                
                if included {
                    print("  -> Included '\(prop.name)' at offset \(stride) (Size: \(size))")
                    propertyOffsets.append(stride)
                    stride += size
                } else {
                    print("  -> Skipped '\(prop.name)' (Not Critical/Packed)")
                    propertyOffsets.append(-1) 
                }
            }
            print("Sharp: Final Packed Stride: \(stride) (Expected ~\(actualStride))")
            stride = actualStride
        }
        
        let expectedSize = vertexCount * stride
        if data.count < expectedSize {
            // Allow minor truncation if it's just a few bytes, but warn
            print("Sharp Warning: Data size mismatch. Expected \(expectedSize), got \(data.count)")
        }
        
        // Optimization: Pre-calculate relative offsets for direct access
        let xOff = (xIdx != -1 && xIdx < propertyOffsets.count) ? propertyOffsets[xIdx] : -1
        let yOff = (yIdx != -1 && yIdx < propertyOffsets.count) ? propertyOffsets[yIdx] : -1
        let zOff = (zIdx != -1 && zIdx < propertyOffsets.count) ? propertyOffsets[zIdx] : -1
        let rOff = (redIdx != -1 && redIdx < propertyOffsets.count) ? propertyOffsets[redIdx] : -1
        let gOff = (greenIdx != -1 && greenIdx < propertyOffsets.count) ? propertyOffsets[greenIdx] : -1
        let bOff = (blueIdx != -1 && blueIdx < propertyOffsets.count) ? propertyOffsets[blueIdx] : -1
        let opOff = (opacityIdx != -1 && opacityIdx < propertyOffsets.count) ? propertyOffsets[opacityIdx] : -1
        let sOff = (scale0Idx != -1 && scale0Idx < propertyOffsets.count) ? propertyOffsets[scale0Idx] : -1
        let rotOff = (rot0Idx != -1 && rot0Idx < propertyOffsets.count) ? propertyOffsets[rot0Idx] : -1
        
        // SH Offsets
        let shOffsets = hasSH ? shIndices.map { propertyOffsets[$0] } : []

        // Unsafe Access for Speed
        try data.withUnsafeBytes { buffer in
            guard let basePtr = buffer.baseAddress else { throw SharpServiceError.invalidPLYFormat }
            
            for i in 0..<vertexCount {
                let vertexStart = basePtr + i * stride
                
                // Helper to read float at offset relative to vertexStart
                // Assume standard float (4 bytes)
                func floatAt(_ offset: Int) -> Float {
                    if offset == -1 { return 0 }
                     // Bounds safety check could be removed for raw speed if we trust stride logic
                     // But for now, let's trust stride.
                    return vertexStart.advanced(by: offset).load(as: Float.self)
                }
                
                // Positions
                let x = floatAt(xOff)
                let y = floatAt(yOff)
                let z = floatAt(zOff)
                positions.append(SIMD3<Float>(x, y, z))
                
                // Colors (DC)
                // Note: If using SH, these are f_dc coefficients, not raw RGB.
                // 3DGS stores 0th order SH (DC) here.
                // If it's a standard point cloud, it might be uint8 RGB.
                var r: Float = 0
                var g: Float = 0
                var b: Float = 0
                
                if redIdx != -1 {
                    if properties[redIdx].name.hasPrefix("f_dc") {
                        // SH 0th order -> RGB conversion
                        // color = 0.5 + 0.28209 * f_dc
                        let shC: Float = 0.28209479177387814
                        r = 0.5 + shC * floatAt(rOff)
                        g = 0.5 + shC * floatAt(gOff)
                        b = 0.5 + shC * floatAt(bOff)
                    } else if properties[redIdx].type.contains("char") || properties[redIdx].type.contains("uint") {
                        // Integer color
                        let rVal = vertexStart.advanced(by: rOff).load(as: UInt8.self)
                        let gVal = vertexStart.advanced(by: gOff).load(as: UInt8.self)
                        let bVal = vertexStart.advanced(by: bOff).load(as: UInt8.self)
                        r = Float(rVal) / 255.0
                        g = Float(gVal) / 255.0
                        b = Float(bVal) / 255.0
                    } else {
                        // Float color
                        r = floatAt(rOff)
                        g = floatAt(gOff)
                        b = floatAt(bOff)
                    }
                }
                colors.append(SIMD3<Float>(max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b))))
                
                // Opacity
                let opacity = opOff != -1 ? 1.0 / (1.0 + exp(-floatAt(opOff))) : 1.0
                opacities.append(opacity)
                
                // Scale (Exp)
                if sOff != -1 {
                    scales.append(SIMD3<Float>(
                        exp(floatAt(sOff)),
                        exp(floatAt(sOff + 4)),
                        exp(floatAt(sOff + 8))
                    ))
                } else {
                    scales.append(SIMD3<Float>(0.01, 0.01, 0.01))
                }
                
                // Rotation
                if rotOff != -1 {
                    let q = SIMD4<Float>(
                        floatAt(rotOff),
                        floatAt(rotOff + 4),
                        floatAt(rotOff + 8),
                        floatAt(rotOff + 12)
                    )
                    // Normalize
                    let len = sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
                    rotations.append(len > 0 ? q / len : SIMD4<Float>(1, 0, 0, 0))
                } else {
                    rotations.append(SIMD4<Float>(1, 0, 0, 0))
                }
                
                // SH Data
                if hasSH {
                    for offset in shOffsets {
                        shs.append(floatAt(offset))
                    }
                }
            }
        }
        
        print("Sharp: Successfully parsed \(positions.count) vertices")
        if hasSH { print("Sharp: Loaded SH data (\(shs.count) floats)") }
        
        // DEBUG: Print first 5 vertices to diagnose mapping errors
        print("Sharp DEBUG: Sample Vertices (First 5):")
        for i in 0..<min(5, positions.count) {
            print("  [\(i)] POS: \(positions[i]), COL: \(colors[i]), OP: \(opacities[i]), SCL: \(scales[i]), ROT: \(rotations[i])")
            if hasSH && !shs.isEmpty {
                // Print first coeff
                // 45 coeffs per splat.
                let shStart = i * 45
                print("      SH[0..2]: \(shs[shStart]), \(shs[shStart+1]), \(shs[shStart+2])")
            }
        }

        return (positions, colors, opacities, scales, rotations, shs)
    }
    
    @MainActor
    private static func parseAsciiPLY(_ header: String, data: Data, vertexCount: Int, properties: [(name: String, type: String)]) throws -> ([SIMD3<Float>], [SIMD3<Float>], [Float], [SIMD3<Float>], [SIMD4<Float>], [Float]) {
        // Fallback for ASCII (keep it simple as most are binary)
        guard let content = String(data: data, encoding: .utf8) else {
            throw SharpServiceError.invalidPLYFormat
        }
        
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD3<Float>] = []
        var opacities: [Float] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [SIMD4<Float>] = []
        
        let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
        for (i, line) in lines.enumerated() {
            if i >= vertexCount { break }
            let values = line.components(separatedBy: " ").compactMap { Float($0) }
            if values.count >= 7 {
                positions.append(SIMD3<Float>(values[0], values[1], values[2]))
                colors.append(SIMD3<Float>(values[3], values[4], values[5]))
                opacities.append(values[6])
                scales.append(SIMD3<Float>(0.01, 0.01, 0.01))
                rotations.append(SIMD4<Float>(1, 0, 0, 0))
            }
        }
        return (positions, colors, opacities, scales, rotations, [])
    }
}

// MARK: - Errors

enum SharpServiceError: Error, LocalizedError {
    case pythonNotFound
    case sharpNotInstalled
    case modelNotFound
    case processingFailed(String)
    case failedToLoadPLY
    case invalidPLYFormat
    case renderingFailed
    
    var errorDescription: String? {
        switch self {
        case .pythonNotFound: return "Python 3.13+ not found"
        case .sharpNotInstalled: return "ml-sharp not installed. Run: pip install -r requirements.txt"
        case .modelNotFound: return "Sharp model checkpoint not found"
        case .processingFailed(let msg): return "Processing failed: \(msg)"
        case .failedToLoadPLY: return "Failed to load PLY file"
        case .invalidPLYFormat: return "Invalid PLY format"
        case .renderingFailed: return "Rendering failed"
        }
    }
}

// MARK: - Implementation

@MainActor
class SharpService: SharpServiceProtocol {
    
    private let tempDirectory: URL
    private var cachedGaussians: GaussianSplatData?
    
    init() {
        self.tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AgentGlass-Sharp-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }
    
    deinit {
        // Cleanup temp directory
        try? FileManager.default.removeItem(at: tempDirectory)
    }
    
    /// Check if ml-sharp is available
    func isAvailable() async -> Bool {
        do {
            let result = try await runCommand("sharp", arguments: ["--help"])
            return result.contains("sharp")
        } catch {
            return false
        }
    }
    
    /// Generate 3D Gaussian splats from a single image
    func generateGaussians(from image: NSImage) async throws -> GaussianSplatData {
        // Save input image
        let inputPath = tempDirectory.appendingPathComponent("input.jpg")
        let outputPath = tempDirectory.appendingPathComponent("output")
        
        try? FileManager.default.createDirectory(at: outputPath, withIntermediateDirectories: true)
        
        guard let tiffData = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let jpegData = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.9]) else {
            throw SharpServiceError.processingFailed("Failed to convert image")
        }
        
        try jpegData.write(to: inputPath)
        
        // Run ml-sharp prediction
        let result = try await runCommand("sharp", arguments: [
            "predict",
            "-i", inputPath.path,
            "-o", outputPath.path
        ])
        
        // Check for output PLY file
        let plyFiles = try FileManager.default.contentsOfDirectory(at: outputPath, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "ply" }
        
        guard let plyPath = plyFiles.first else {
            throw SharpServiceError.processingFailed("No PLY output generated: \(result)")
        }
        
        let gaussians = try GaussianSplatData(plyPath: plyPath)
        self.cachedGaussians = gaussians
        return gaussians
    }
    
    /// Render a novel view from Gaussian splats
    func renderNovelView(_ splats: GaussianSplatData, cameraPosition: CameraPosition) async throws -> NSImage {
        // Use Metal to render the Gaussian splats
        // This is a simplified splatting renderer
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SharpServiceError.renderingFailed
        }
        
        // For now, use a software-based point cloud renderer
        // A full implementation would use Metal compute shaders for Gaussian splatting
        return try renderPointCloud(splats, camera: cameraPosition, device: device)
    }
    
    /// Generate parallax animation frames
    func generateParallaxAnimation(from image: NSImage, duration: Double, style: ParallaxStyle) async throws -> [NSImage] {
        // Generate Gaussians if not cached
        let splats = try await generateGaussians(from: image)
        
        let frameCount = Int(duration * 30)  // 30 fps
        var frames: [NSImage] = []
        
        for i in 0..<frameCount {
            let t = Double(i) / Double(frameCount - 1)
            let camera = cameraPositionForStyle(style, at: t)
            let frame = try await renderNovelView(splats, cameraPosition: camera)
            frames.append(frame)
        }
        
        return frames
    }
    
    // MARK: - Private Helpers
    
    private func runCommand(_ command: String, arguments: [String]) async throws -> String {
        let executablePath = await findExecutable(command)
        
        return try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            let pipe = Pipe()
            
            process.executableURL = URL(fileURLWithPath: executablePath)
            // If we found a venv path, we might need to set up the environment
            if executablePath.contains("/venv/") {
                var env = ProcessInfo.processInfo.environment
                let venvBin = URL(fileURLWithPath: executablePath).deletingLastPathComponent().path
                env["PATH"] = "\(venvBin):\(env["PATH"] ?? "")"
                env["VIRTUAL_ENV"] = URL(fileURLWithPath: venvBin).deletingLastPathComponent().path
                process.environment = env
            }
            
            process.arguments = arguments
            process.standardOutput = pipe
            process.standardError = pipe
            
            do {
                try process.run()
                process.waitUntilExit()
                
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let output = String(data: data, encoding: .utf8) ?? ""
                
                if process.terminationStatus == 0 {
                    continuation.resume(returning: output)
                } else {
                    continuation.resume(throwing: SharpServiceError.processingFailed(output))
                }
            } catch {
                continuation.resume(throwing: SharpServiceError.pythonNotFound)
            }
        }
    }
    
    private func findExecutable(_ name: String) async -> String {
        // 1. Check current directory (if running in dev)
        let cwd = FileManager.default.currentDirectoryPath
        let possiblePaths = [
            "\(cwd)/ml-sharp/venv/bin/\(name)",
            "\(cwd)/venv/bin/\(name)",
            "/usr/local/bin/\(name)",
            "/opt/homebrew/bin/\(name)"
        ]
        
        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        
        // 2. Fallback to /usr/bin/env to find it in system PATH
        return "/usr/bin/env"
    }
    
    private func cameraPositionForStyle(_ style: ParallaxStyle, at t: Double) -> CameraPosition {
        // t goes from 0 to 1
        let angle = t * 2 * .pi
        
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
    
    private func renderPointCloud(_ splats: GaussianSplatData, camera: CameraPosition, device: MTLDevice) throws -> NSImage {
        // Simplified point cloud renderer (placeholder for full Gaussian splatting)
        // A production implementation would use compute shaders for proper 3DGS rendering
        
        let width = 800
        let height = 600
        
        // Create bitmap context
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw SharpServiceError.renderingFailed
        }
        
        // Clear to black
        context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Simple projection matrix
        let fov: Float = 60 * .pi / 180
        let aspect = Float(width) / Float(height)
        let near: Float = 0.1
        // let far: Float = 100.0
        
        // Project and render each Gaussian as a point/circle
        for i in 0..<splats.pointCount {
            var pos = splats.positions[i]
            
            // Apply camera transform
            pos.x -= Float(camera.x)
            pos.y -= Float(camera.y)
            pos.z -= Float(camera.z)
            
            // Apply rotation (simplified)
            let cosY = cos(Float(camera.rotationY))
            let sinY = sin(Float(camera.rotationY))
            let rotX = pos.x * cosY - pos.z * sinY
            let rotZ = pos.x * sinY + pos.z * cosY
            pos.x = rotX
            pos.z = rotZ
            
            // Skip if behind camera
            if pos.z <= near { continue }
            
            // Perspective projection
            let scale = 1.0 / tan(fov / 2)
            let x2d = (pos.x * scale / pos.z) / aspect
            let y2d = pos.y * scale / pos.z
            
            // Convert to screen coordinates
            let screenX = Int((x2d + 1) * 0.5 * Float(width))
            let screenY = Int((1 - y2d) * 0.5 * Float(height))
            
            // Skip if off screen
            if screenX < 0 || screenX >= width || screenY < 0 || screenY >= height { continue }
            
            // Draw point
            let color = splats.colors[i]
            let opacity = splats.opacities[i]
            let size = max(1, Int(5 / pos.z))  // Perspective size
            
            context.setFillColor(CGColor(
                red: CGFloat(color.x),
                green: CGFloat(color.y),
                blue: CGFloat(color.z),
                alpha: CGFloat(opacity)
            ))
            context.fillEllipse(in: CGRect(
                x: screenX - size/2,
                y: screenY - size/2,
                width: size,
                height: size
            ))
        }
        
        guard let cgImage = context.makeImage() else {
            throw SharpServiceError.renderingFailed
        }
        
        return NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
    }
}
