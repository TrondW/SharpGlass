import Foundation
import AppKit
import Metal
import MetalKit
import Vision
import AVFoundation
import ImageIO

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
public protocol SharpServiceProtocol {
    /// Checks if the backend executable exists and is executable.
    func isAvailable() async -> Bool
    
    /// Generates 3D Gaussian Splats from a single 2D image.
    /// - Parameters:
    ///   - image: The source NSImage.
    ///   - originalURL: Optional URL for fallback if background removal fails.
    ///   - cleanBackground: Whether to attempt background removal (defaults to false).
    /// - Returns: Parsed `GaussianSplatData` ready for rendering.
    func generateGaussians(from image: NSImage, originalURL: URL?, cleanBackground: Bool) async throws -> GaussianSplatData
    
    /// Renders a novel view implementation of the scene.
    /// - Parameters:
    ///   - splats: The Gaussian data.
    ///   - cameraPosition: The desired virtual camera pose.
    /// - Returns: The rendered frame as an NSImage.
    func renderNovelView(_ splats: GaussianSplatData, cameraPosition: CameraPosition) async throws -> NSImage
    
    
    /// Downloads and installs the ml-sharp backend environment.
    /// - Parameter progress: Closure reporting (Stage Name, 0.0-1.0 progress).
    func setupBackend(progress: @escaping (String, Double) -> Void) async throws
    
    /// Cleans up temporary files.
    func cleanup()
}

// MARK: - Data Types

public struct CameraPosition: Sendable {
    public var x: Double = 0
    public var y: Double = 0
    public var z: Double = 0
    public var rotationX: Double = 0
    public var rotationY: Double = 0
    
    public var target: SIMD3<Double>? = nil
    
    public static let center = CameraPosition()
    
    public init(x: Double = 0, y: Double = 0, z: Double = 0, rotationX: Double = 0, rotationY: Double = 0, target: SIMD3<Double>? = nil) {
        self.x = x
        self.y = y
        self.z = z
        self.rotationX = rotationX
        self.rotationY = rotationY
        self.target = target
    }
    
    public func viewMatrix() -> matrix_float4x4 {
        let eye = vector_float3(Float(self.x), Float(self.y), Float(self.z))
        let targetPos: vector_float3
        if let t = self.target {
            targetPos = vector_float3(Float(t.x), Float(t.y), Float(t.z))
        } else {
            targetPos = eye + vector_float3(0, 0, -1)
        }
        let up = vector_float3(0, 1, 0)
        
        let z = normalize(eye - targetPos)
        let x = normalize(simd_cross(up, z))
        let y = cross(z, x)
        
        return matrix_float4x4(columns: (
            vector_float4(x.x, y.x, z.x, 0),
            vector_float4(x.y, y.y, z.y, 0),
            vector_float4(x.z, y.z, z.z, 0),
            vector_float4(-dot(x, eye), -dot(y, eye), -dot(z, eye), 1)
        ))
    }
}


/// Represents 3D Gaussian Splat data
public struct GaussianSplatData: Identifiable, Sendable {
    public let id = UUID()
    public var plyData: Data? // Raw PLY file data (can be evicted to save memory)
    public let positions: [SIMD3<Float>]  // Gaussian centers
    public let colors: [SIMD3<Float>]  // RGB colors
    public let opacities: [Float]  // Alpha values
    public let scales: [SIMD3<Float>]  // Log-scales (converted to exp)
    public let rotations: [SIMD4<Float>] // Quaternions
    public let shs: [Float] // Spherical Harmonics coefficients (45 per splat)
    public let pointCount: Int
    
    public init(plyPath: URL) throws {
        guard let data = try? Data(contentsOf: plyPath) else {
            throw SharpServiceError.failedToLoadPLY
        }
        self.plyData = data
        
        // Parse PLY header and data
        // Parse PLY header and data
        (self.positions, self.colors, self.opacities, self.scales, self.rotations, self.shs) = try GaussianSplatData.parsePLY(data)
        self.pointCount = positions.count
    }
    
    /// Create PLY file data from Gaussian splat attributes.
    /// - Parameters:
    ///   - positions: XYZ positions
    ///   - colors: RGB colors (will be converted to SH DC)
    ///   - opacities: Log-space opacity (or alpha, depending on pipeline)
    ///   - scales: Log-space scales
    ///   - rotations: Quaternions (r, x, y, z)
    /// - Returns: Binary PLY data
    public static func createPLYData(
        positions: [SIMD3<Float>],
        colors: [SIMD3<Float>],
        opacities: [Float],
        scales: [SIMD3<Float>],
        rotations: [SIMD4<Float>],
        shs: [Float] = []
    ) throws -> Data {
        let pointCount = positions.count
        
        // Use an array of strings to build header safely, avoiding multiline string newline pitfalls
        let headerLines = [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex \(pointCount)",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header"
        ]
        
        let headerString = headerLines.joined(separator: "\n") + "\n"
        
        guard var data = headerString.data(using: .utf8) else {
            throw SharpServiceError.processingFailed("Failed to create PLY header")
        }
        
        // Write binary data for each point
        for i in 0..<pointCount {
            // Position
            data.append(contentsOf: withUnsafeBytes(of: positions[i].x) { Data($0) })
            data.append(contentsOf: withUnsafeBytes(of: positions[i].y) { Data($0) })
            data.append(contentsOf: withUnsafeBytes(of: positions[i].z) { Data($0) })
            
            // Normal (zeros for now)
            let zero: Float = 0.0
            data.append(contentsOf: withUnsafeBytes(of: zero) { Data($0) })
            data.append(contentsOf: withUnsafeBytes(of: zero) { Data($0) })
            data.append(contentsOf: withUnsafeBytes(of: zero) { Data($0) })
            
            // Color: Convert RGB to SH DC coefficients (f_dc_0, f_dc_1, f_dc_2)
            // Inverse of: rgb = 0.5 + 0.28209 * f_dc
            // Therefore: f_dc = (rgb - 0.5) / 0.28209
            let shC: Float = 0.28209479177387814
            let dc0 = (colors[i].x - 0.5) / shC
            let dc1 = (colors[i].y - 0.5) / shC
            let dc2 = (colors[i].z - 0.5) / shC
            data.append(contentsOf: withUnsafeBytes(of: dc0) { Data($0) })
            data.append(contentsOf: withUnsafeBytes(of: dc1) { Data($0) })
            data.append(contentsOf: withUnsafeBytes(of: dc2) { Data($0) })

            // Opacity (MUST match header order)
            data.append(contentsOf: withUnsafeBytes(of: opacities[i]) { Data($0) })
            
            // Scale
            data.append(contentsOf: withUnsafeBytes(of: scales[i].x) { Data($0) })
            data.append(contentsOf: withUnsafeBytes(of: scales[i].y) { Data($0) })
            data.append(contentsOf: withUnsafeBytes(of: scales[i].z) { Data($0) })
            
            // Rotation (quaternion)
            data.append(contentsOf: withUnsafeBytes(of: rotations[i].x) { Data($0) })
            data.append(contentsOf: withUnsafeBytes(of: rotations[i].y) { Data($0) })
            data.append(contentsOf: withUnsafeBytes(of: rotations[i].z) { Data($0) })
            data.append(contentsOf: withUnsafeBytes(of: rotations[i].w) { Data($0) })
        }
        
        return data
    }
    public init(data: Data) throws {
        self.plyData = data
        
        // Parse PLY header and data
        (self.positions, self.colors, self.opacities, self.scales, self.rotations, self.shs) = try GaussianSplatData.parsePLY(data)
        self.pointCount = positions.count
    }
    
    /// Evicts the raw PLY data from memory to save space. 
    /// Should be called after the splat is loaded into GPU/Mental buffers.
    public mutating func evictRawPLYData() {
        self.plyData = nil
    }
    
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
        
        // RUTHLESS VALIDATION: Ensure critical properties exist
        guard xIdx != -1, yIdx != -1, zIdx != -1 else {
            print("Sharp Error: Missing critical XYZ properties")
            throw SharpServiceError.invalidPLYFormat
        }
        
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
            print("Sharp Error: Data size mismatch. Expected at least \(expectedSize), got \(data.count)")
            throw SharpServiceError.invalidPLYFormat
        }

        // RUTHLESS VALIDATION: Ensure critical offsets were found in packed layout
        let criticalOffsets = [xIdx: "x", yIdx: "y", zIdx: "z"]
        for (idx, name) in criticalOffsets {
            if idx != -1 && (idx >= propertyOffsets.count || propertyOffsets[idx] == -1) {
                print("Sharp Error: Critical property '\(name)' was excluded or missing in packed layout")
                throw SharpServiceError.invalidPLYFormat
            }
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
    
    // MARK: - Pruning
    
    /// Intelligently prune splats to a maximum count based on quality metrics
    /// Preserves the highest-quality, most visually important splats
    /// Intelligently prune splats to a maximum count based on quality metrics
    /// Preserves the highest-quality, most visually important splats
    public func pruned(maxCount: Int) -> GaussianSplatData {
        guard pointCount > maxCount else {
            return self // Already under limit
        }
        
        print("Sharp: Pruning splat from \(pointCount) to \(maxCount) points...")
        
        // Calculate scene center for spatial scoring
        var center = SIMD3<Float>(0, 0, 0)
        for pos in positions {
            center += pos
        }
        center /= Float(pointCount)
        
        // Calculate quality score for each splat
        struct SplatScore {
            let index: Int
            let score: Float
        }
        
        var scores: [SplatScore] = []
        scores.reserveCapacity(pointCount)
        
        for i in 0..<pointCount {
            let opacity = opacities[i]
            let scale = scales[i]
            let pos = positions[i]
            
            // Opacity score (0-1): Higher opacity = more visible
            let opacityScore = opacity
            
            // Scale score (0-1): Moderate scales are good, extremes are artifacts
            // Average scale across 3 dimensions
            let avgScale = (scale.x + scale.y + scale.z) / 3.0
            // Ideal range: 0.01 to 1.0, penalize outside this range
            let scaleScore: Float
            if avgScale < 0.001 {
                scaleScore = 0.1 // Too small, likely noise
            } else if avgScale > 10.0 {
                scaleScore = 0.2 // Too large, likely artifact
            } else if avgScale >= 0.01 && avgScale <= 1.0 {
                scaleScore = 1.0 // Ideal range
            } else {
                scaleScore = 0.5 // Acceptable but not ideal
            }
            
            // Spatial score (0-1): Closer to center = more important
            let distance = simd_distance(pos, center)
            let maxDistance: Float = 50.0 // Assume scene is within 50 units
            let spatialScore = max(0, 1.0 - (distance / maxDistance))
            
            // Combined score with weights
            // Opacity is most important (50%), scale quality (30%), spatial (20%)
            let totalScore = opacityScore * 0.5 + scaleScore * 0.3 + spatialScore * 0.2
            
            scores.append(SplatScore(index: i, score: totalScore))
        }
        
        // Sort by score descending and take top N
        scores.sort { $0.score > $1.score }
        let topScores = Array(scores.prefix(maxCount))
        
        // Build pruned arrays in quality order (highest quality first)
        var prunedPositions: [SIMD3<Float>] = []
        var prunedColors: [SIMD3<Float>] = []
        var prunedOpacities: [Float] = []
        var prunedScales: [SIMD3<Float>] = []
        var prunedRotations: [SIMD4<Float>] = []
        var prunedShs: [Float] = []
        
        prunedPositions.reserveCapacity(maxCount)
        prunedColors.reserveCapacity(maxCount)
        prunedOpacities.reserveCapacity(maxCount)
        prunedScales.reserveCapacity(maxCount)
        prunedRotations.reserveCapacity(maxCount)
        
        let hasSH = !shs.isEmpty
        if hasSH {
            prunedShs.reserveCapacity(maxCount * 45)
        }
        
        // Iterate through sorted scores to maintain quality order
        for scoreEntry in topScores {
            let i = scoreEntry.index
            prunedPositions.append(positions[i])
            prunedColors.append(colors[i])
            prunedOpacities.append(opacities[i])
            prunedScales.append(scales[i])
            prunedRotations.append(rotations[i])
            
            if hasSH {
                let shStart = i * 45
                for j in 0..<45 {
                    prunedShs.append(shs[shStart + j])
                }
            }
        }
        
        let removedPercent = Float(pointCount - maxCount) / Float(pointCount) * 100.0
        print("Sharp: ✅ Pruned splat (removed \(String(format: "%.1f", removedPercent))% low-quality splats)")
        
        // Create new GaussianSplatData with pruned arrays
        // We now generate plyData so the pruned splat can be saved/previewed
        let prunedData = try? GaussianSplatData.createPLYData(
            positions: prunedPositions,
            colors: prunedColors,
            opacities: prunedOpacities,
            scales: prunedScales,
            rotations: prunedRotations,
            shs: prunedShs
        )
        
        return GaussianSplatData(
            positions: prunedPositions,
            colors: prunedColors,
            opacities: prunedOpacities,
            scales: prunedScales,
            rotations: prunedRotations,
            shs: prunedShs,
            plyData: prunedData
        )
    }
    
    // Private initializer for pruned data
    private init(positions: [SIMD3<Float>], colors: [SIMD3<Float>], opacities: [Float], 
                  scales: [SIMD3<Float>], rotations: [SIMD4<Float>], shs: [Float], plyData: Data?) {
        self.plyData = plyData
        self.positions = positions
        self.colors = colors
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations
        self.shs = shs
        self.pointCount = positions.count
    }
}


// MARK: - Errors

public enum SharpServiceError: Error, LocalizedError {
    case pythonNotFound
    case sharpNotInstalled
    case modelNotFound
    case processingFailed(String)
    case failedToLoadPLY
    case invalidPLYFormat
    case renderingFailed
    case runtimeError(String)
    
    public var errorDescription: String? {
        switch self {
        case .pythonNotFound: return "Python 3.13+ not found"
        case .sharpNotInstalled: return "ml-sharp not installed. Run: pip install -r requirements.txt"
        case .modelNotFound: return "Sharp model checkpoint not found"
        case .processingFailed(let msg): return "Processing failed: \(msg)"
        case .failedToLoadPLY: return "Failed to load PLY file"
        case .invalidPLYFormat: return "Invalid PLY format"
        case .renderingFailed: return "Rendering failed"
        case .runtimeError(let msg): return msg
        }
    }
}

// MARK: - Implementation

@MainActor
public class SharpService: SharpServiceProtocol {
    
    public let tempDirectory: URL
    private var cachedGaussians: GaussianSplatData?
    
    /// Flag to disable buggy stereo reconstruction (spatial image merge) for now.
    /// When disabled, spatial photos will be processed as single images.
    private let isStereoReconstructionEnabled = false
    
    public init() {
        self.tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AgentGlass-Sharp-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }
    
    deinit {
        // Final cleanup
        try? FileManager.default.removeItem(at: tempDirectory)
    }
    
    /// Clean up intermediate files in the temporary directory (inputs, depths, partial PLYs)
    /// but keeps the main temp directory structure.
    public func cleanup() {
        print("Sharp: Cleaning up intermediate processing files...")
        do {
            let files = try FileManager.default.contentsOfDirectory(at: tempDirectory, includingPropertiesForKeys: nil)
            for file in files {
                // Don't remove the directories themselves during intermediate cleanup
                // unless we are finishing entirely.
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: file.path, isDirectory: &isDir), !isDir.boolValue {
                    try FileManager.default.removeItem(at: file)
                }
            }
        } catch {
            print("Sharp Warning: Cleanup failed: \(error)")
        }
    }
    
    /// Check if ml-sharp is available
    public func isAvailable() async -> Bool {
        // Fast check: if we know the path exists from previous runs
        let setupPath = await getAppSupportPath()
        let localPath = setupPath + "/venv/bin/sharp"
        if FileManager.default.fileExists(atPath: localPath) {
            print("Sharp: ✅ Found local backend at \(localPath)")
            return true
        }
    
        do {
            // Debug: Check where it's resolving to
            let resolved = await findExecutable("sharp")
            print("Sharp: 🔎 Checking availability... Resolved 'sharp' to: \(resolved)")
            
            let result = try await runCommand("sharp", arguments: ["--help"])
            let valid = result.contains("sharp")
            if valid {
                print("Sharp: ✅ Backend confirmed available via: \(resolved)")
            } else {
                print("Sharp: ❌ 'sharp' command found but --help validation failed.")
            }
            return valid
        } catch {
            print("Sharp: ❌ Backend not found. Error: \(error)")
            return false
        }
    }
    
    private func getAppSupportPath() async -> String {
        guard let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return "/tmp"
        }
        let path = appSupport.appendingPathComponent("com.trond.SharpGlass").path
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        return path
    }
    
    /// Setup the backend by creating a venv and installing dependencies
    public func setupBackend(progress: @escaping (String, Double) -> Void) async throws {
        let appSupport = await getAppSupportPath()
        
        // 1. Check for Python 3.13
        progress("Checking for Python 3.13...", 0.1)
        do {
            // Try to find python3.13 in path
            let _ = try await runCommand("/usr/bin/env", arguments: ["python3.13", "--version"])
        } catch {
            throw SharpServiceError.runtimeError("Python 3.13 not found. Please install it with 'brew install python@3.13' or from python.org")
        }
        
        // 2. Locate ml-sharp sources
        // In a bundled app, this should be in Resources/ml-sharp.
        // In dev, it might be a sibling directory.
        progress("Locating ml-sharp sources...", 0.2)
        
        var sourcePath: String?
        
        // Check Bundle
        if let bundlePath = Bundle.main.path(forResource: "ml-sharp", ofType: nil) {
            sourcePath = bundlePath
        } 
        // Check Dev Sibling (Code/SharpGlass/ml-sharp)
        else {
            let devPath = FileManager.default.currentDirectoryPath + "/ml-sharp"
            if FileManager.default.fileExists(atPath: devPath) {
                sourcePath = devPath
            }
        }
        
        guard let mlSharpPath = sourcePath else {
            throw SharpServiceError.runtimeError("Could not find ml-sharp sources. Please ensure 'ml-sharp' folder is inside the app bundle or project directory.")
        }
        
        // 3. Create venv
        progress("Creating virtual environment...", 0.3)
        let venvPath = appSupport + "/venv"
        if !FileManager.default.fileExists(atPath: venvPath) {
            _ = try await runCommand("/usr/bin/env", arguments: ["python3.13", "-m", "venv", venvPath])
        }
        
        // 4. Update pip
        progress("Updating pip...", 0.4)
        let pipPath = venvPath + "/bin/pip"
        _ = try await runCommand(pipPath, arguments: ["install", "--upgrade", "pip"])
        
        // 5. Install Dependencies
        progress("Installing dependencies (this may take a while)...", 0.5)
        // We install from the ml-sharp source reference
        // Note: We use -e . if in dev, but for user install just install regular
        _ = try await runCommand(pipPath, arguments: ["install", mlSharpPath])
        
        // 6. Verify
        progress("Verifying installation...", 0.9)
        let sharpPath = venvPath + "/bin/sharp"
        if !FileManager.default.fileExists(atPath: sharpPath) {
             throw SharpServiceError.runtimeError("Installation completed but 'sharp' binary is missing.")
        }
        
        progress("Ready!", 1.0)
    }
    
    /// Generate 3D Gaussian splats from a single image
    public func generateGaussians(from image: NSImage, originalURL: URL? = nil, cleanBackground: Bool = false) async throws -> GaussianSplatData {
        // Save input image
        let inputPath = tempDirectory.appendingPathComponent("input.jpg")
        let outputPath = tempDirectory.appendingPathComponent("output")
        
        try? FileManager.default.createDirectory(at: outputPath, withIntermediateDirectories: true)
        
        // If background removal is requested, we process the image first
        if cleanBackground {
            print("Sharp: Removing background using Vision AI...")
            do {
                if let maskedImage = try await removeBackground(from: image) {
                    print("Sharp: Background removed successfully.")
                    let jpegData = autoreleasepool { () -> Data? in
                        guard let tiffData = maskedImage.tiffRepresentation,
                              let bitmap = NSBitmapImageRep(data: tiffData) else { return nil }
                        return bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.9])
                    }
                    guard let data = jpegData else {
                        throw SharpServiceError.processingFailed("Failed to convert masked image")
                    }
                    try data.write(to: inputPath)
                } else {
                    print("Sharp Warning: Background removal found no subject. Falling back to original.")
                    try saveFallbackImage(image: image, originalURL: originalURL, to: inputPath)
                }
            } catch {
                print("Sharp Error: Background removal failed with error: \(error.localizedDescription). Falling back to original.")
                try saveFallbackImage(image: image, originalURL: originalURL, to: inputPath)
            }
        } else {
            // Always convert to JPEG format (original might be HEIC or other format PIL can't read)
            print("Sharp: Converting image to JPEG format...")
            let jpegData = autoreleasepool { () -> Data? in
                guard let tiffData = image.tiffRepresentation,
                      let bitmap = NSBitmapImageRep(data: tiffData) else { return nil }
                return bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.9])
            }
            guard let data = jpegData else {
                throw SharpServiceError.processingFailed("Failed to convert image to JPEG")
            }
            try data.write(to: inputPath)
            print("Sharp: Image converted and saved to \(inputPath.path)")
        }
        
        // Check for spatial photo depth data (iPhone spatial photos)
        if let url = originalURL {
            if let depthData = extractDepthData(from: url) {
                print("Sharp: ✨ Spatial photo detected! Extracting depth map...")
                let depthPath = tempDirectory.appendingPathComponent("depth.png")
                do {
                    try saveDepthMap(depthData: depthData, to: depthPath)
                    print("Sharp: ✅ Depth map saved to \(depthPath.path)")
                    // TODO: Pass depth map to ml-sharp when depth input is supported
                } catch {
                    print("Sharp Warning: Failed to save depth map: \(error.localizedDescription)")
                }
            }
            
            
            // Check for stereo pair (iPhone spatial photos with left/right images)
            if isStereoReconstructionEnabled, let (leftImage, rightImage) = extractStereoPair(from: url) {
                print("Sharp: 🎬 Stereo pair detected! Using stereo reconstruction...")
                print("Sharp: Processing left image...")
                
                // Generate splat from left image
                let leftInputPath = tempDirectory.appendingPathComponent("input_left.jpg")
                guard let leftTiffData = leftImage.tiffRepresentation,
                      let leftBitmap = NSBitmapImageRep(data: leftTiffData),
                      let leftJpegData = leftBitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.9]) else {
                    throw SharpServiceError.processingFailed("Failed to convert left image to JPEG")
                }
                try leftJpegData.write(to: leftInputPath)
                
                let leftOutputPath = tempDirectory.appendingPathComponent("output_left")
                try? FileManager.default.createDirectory(at: leftOutputPath, withIntermediateDirectories: true)
                
                let leftResult = try await runCommand("sharp", arguments: [
                    "predict",
                    "-i", leftInputPath.path,
                    "-o", leftOutputPath.path
                ])
                
                let leftPlyFiles = try FileManager.default.contentsOfDirectory(at: leftOutputPath, includingPropertiesForKeys: nil)
                    .filter { $0.pathExtension == "ply" }
                guard let leftPlyPath = leftPlyFiles.first else {
                    throw SharpServiceError.processingFailed("No PLY output for left image: \(leftResult)")
                }
                
                print("Sharp: Processing right image...")
                
                // Generate splat from right image
                let rightInputPath = tempDirectory.appendingPathComponent("input_right.jpg")
                guard let rightTiffData = rightImage.tiffRepresentation,
                      let rightBitmap = NSBitmapImageRep(data: rightTiffData),
                      let rightJpegData = rightBitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.9]) else {
                    throw SharpServiceError.processingFailed("Failed to convert right image to JPEG")
                }
                try rightJpegData.write(to: rightInputPath)
                
                let rightOutputPath = tempDirectory.appendingPathComponent("output_right")
                try? FileManager.default.createDirectory(at: rightOutputPath, withIntermediateDirectories: true)
                
                let rightResult = try await runCommand("sharp", arguments: [
                    "predict",
                    "-i", rightInputPath.path,
                    "-o", rightOutputPath.path
                ])
                
                let rightPlyFiles = try FileManager.default.contentsOfDirectory(at: rightOutputPath, includingPropertiesForKeys: nil)
                    .filter { $0.pathExtension == "ply" }
                guard let rightPlyPath = rightPlyFiles.first else {
                    throw SharpServiceError.processingFailed("No PLY output for right image: \(rightResult)")
                }
                
                
                // Validate PLY files exist and have content
                guard FileManager.default.fileExists(atPath: leftPlyPath.path) else {
                    throw SharpServiceError.processingFailed("Left PLY file not created")
                }
                guard FileManager.default.fileExists(atPath: rightPlyPath.path) else {
                    throw SharpServiceError.processingFailed("Right PLY file not created")
                }
                
                // Load both splats with detailed logging
                print("Sharp: Loading left splat from \(leftPlyPath.path)...")
                let leftSplat = try await MainActor.run { try GaussianSplatData(plyPath: leftPlyPath) }
                print("Sharp: ✅ Left splat loaded (\(leftSplat.pointCount) points)")
                
                print("Sharp: Loading right splat from \(rightPlyPath.path)...")
                let rightSplat = try await MainActor.run { try GaussianSplatData(plyPath: rightPlyPath) }
                print("Sharp: ✅ Right splat loaded (\(rightSplat.pointCount) points)")
                
                // Merge splats with baseline transformation
                let mergedSplat = try await mergeSplats(left: leftSplat, right: rightSplat, baseline: 0.065)
                
                self.cachedGaussians = mergedSplat
                return mergedSplat
            }
        }
        
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
    public func renderNovelView(_ splats: GaussianSplatData, cameraPosition: CameraPosition) async throws -> NSImage {
        // Use Metal to render the Gaussian splats
        // This is a simplified splatting renderer
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SharpServiceError.renderingFailed
        }
        
        // For now, use a software-based point cloud renderer
        // A full implementation would use Metal compute shaders for Gaussian splatting
        return try renderPointCloud(splats, camera: cameraPosition, device: device)
    }
    

    
    
    private func saveFallbackImage(image: NSImage, originalURL: URL?, to path: URL) throws {
        // Always convert to JPEG format (original might be HEIC or other format PIL can't read)
        print("Sharp: Converting fallback image to JPEG format...")
        guard let tiffData = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let jpegData = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.9]) else {
            throw SharpServiceError.processingFailed("Failed to convert fallback image to JPEG")
        }
        try jpegData.write(to: path)
        print("Sharp: Fallback image converted and saved to \(path.path)")
    }
    
    // MARK: - Spatial Photo Support
    
    /// Extract depth data from iPhone spatial photos (HEIC format)
    /// Returns AVDepthData if the image contains depth/disparity information
    private func extractDepthData(from url: URL) -> AVDepthData? {
        guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            return nil
        }
        
        // Try disparity first (more common in spatial photos)
        if let disparityData = CGImageSourceCopyAuxiliaryDataInfoAtIndex(
            imageSource, 0, kCGImageAuxiliaryDataTypeDisparity) as? [AnyHashable: Any] {
            do {
                return try AVDepthData(fromDictionaryRepresentation: disparityData)
            } catch {
                print("Sharp Warning: Failed to create AVDepthData from disparity: \(error.localizedDescription)")
            }
        }
        
        // Fall back to depth
        if let depthData = CGImageSourceCopyAuxiliaryDataInfoAtIndex(
            imageSource, 0, kCGImageAuxiliaryDataTypeDepth) as? [AnyHashable: Any] {
            do {
                return try AVDepthData(fromDictionaryRepresentation: depthData)
            } catch {
                print("Sharp Warning: Failed to create AVDepthData from depth: \(error.localizedDescription)")
            }
        }
        
        return nil
    }
    
    /// Save depth map from AVDepthData as a PNG image
    /// Converts depth/disparity data to a grayscale image for visualization and future use
    private func saveDepthMap(depthData: AVDepthData, to path: URL) throws {
        // Convert to depth (meters) if it's disparity
        let convertedDepthData = depthData.depthDataType == kCVPixelFormatType_DisparityFloat32 
            ? depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
            : depthData
        
        let depthMap = convertedDepthData.depthDataMap
        
        // Convert CVPixelBuffer to CIImage
        let ciImage = CIImage(cvPixelBuffer: depthMap)
        
        // Create CGImage from CIImage
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw SharpServiceError.processingFailed("Failed to create depth map CGImage")
        }
        
        let nsImage = NSImage(cgImage: cgImage, size: NSSize(
            width: cgImage.width, 
            height: cgImage.height
        ))
        
        // Save as PNG to preserve depth precision
        guard let tiffData = nsImage.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let pngData = bitmap.representation(using: .png, properties: [:]) else {
            throw SharpServiceError.processingFailed("Failed to convert depth map to PNG")
        }
        
        try pngData.write(to: path)
    }
    
    /// Extract stereo pair (left and right images) from iPhone spatial photos
    /// Returns tuple of (left, right) images if spatial photo, nil otherwise
    private func extractStereoPair(from url: URL) -> (left: NSImage, right: NSImage)? {
        guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            return nil
        }
        
        // Check if we have multiple images (spatial photos have 2)
        let imageCount = CGImageSourceGetCount(imageSource)
        guard imageCount >= 2 else {
            return nil
        }
        
        // Extract left image (index 0)
        guard let leftCGImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            return nil
        }
        
        // Extract right image (index 1)
        guard let rightCGImage = CGImageSourceCreateImageAtIndex(imageSource, 1, nil) else {
            return nil
        }
        
        let leftSize = NSSize(width: leftCGImage.width, height: leftCGImage.height)
        let rightSize = NSSize(width: rightCGImage.width, height: rightCGImage.height)
        
        let leftImage = NSImage(cgImage: leftCGImage, size: leftSize)
        let rightImage = NSImage(cgImage: rightCGImage, size: rightSize)
        
        return (leftImage, rightImage)
    }
    
    /// Merge two Gaussian splat point clouds with a baseline transformation
    /// Used for stereo reconstruction from spatial photos
    private func mergeSplats(left: GaussianSplatData, right: GaussianSplatData, baseline: Float = 0.065) async throws -> GaussianSplatData {
        print("Sharp: Merging stereo splats (left: \(left.pointCount) points, right: \(right.pointCount) points)")
        
        do {
            // Validate inputs
            guard left.pointCount > 0 && right.pointCount > 0 else {
                throw SharpServiceError.processingFailed("Cannot merge empty splats")
            }
            
            // Transform right splat positions by baseline (camera separation)
            // iPhone spatial photos have ~65mm baseline
            var transformedRightPositions = right.positions
            for i in 0..<transformedRightPositions.count {
                // Translate right camera view by baseline along X axis
                transformedRightPositions[i].x += baseline
            }
            
            // Combine all attributes from both splats
            let mergedPositions = left.positions + transformedRightPositions
            let mergedColors = left.colors + right.colors
            let mergedOpacities = left.opacities + right.opacities
            let mergedScales = left.scales + right.scales
            let mergedRotations = left.rotations + right.rotations
            let mergedSHs = left.shs + right.shs
            
            print("Sharp: Creating merged PLY data...")
            
            // Create merged PLY data
            let mergedPLYData = try GaussianSplatData.createPLYData(
                positions: mergedPositions,
                colors: mergedColors,
                opacities: mergedOpacities,
                scales: mergedScales,
                rotations: mergedRotations,
                shs: mergedSHs
            )
            
            print("Sharp: Merged PLY data created (\(mergedPLYData.count) bytes)")
            
            // Save merged PLY
            let mergedPath = tempDirectory.appendingPathComponent("merged.ply")
            try mergedPLYData.write(to: mergedPath)
            print("Sharp: ✅ Merged PLY saved to \(mergedPath.path)")
            
            // Load merged splat
            print("Sharp: Loading merged splat...")
            let mergedSplat = try await MainActor.run {
                try GaussianSplatData(plyPath: mergedPath)
            }
            print("Sharp: ✅ Merged splat loaded (\(mergedSplat.pointCount) total points)")
            
            // Auto-prune if exceeds GPU memory limits
            let maxSplatCount = 2_000_000 // 2M splats - individual 1.18M loads work fine
            if mergedSplat.pointCount > maxSplatCount {
                print("Sharp: ⚠️ Merged splat exceeds safe limit (\(mergedSplat.pointCount) > \(maxSplatCount))")
                let prunedSplat = await MainActor.run {
                    mergedSplat.pruned(maxCount: maxSplatCount)
                }
                return prunedSplat
            }
            
            return mergedSplat
        } catch {
            print("Sharp Error: Merge failed: \(error.localizedDescription)")
            throw SharpServiceError.processingFailed("Stereo merge failed: \(error.localizedDescription)")
        }
    }

    
    // REMOVED: createPLYData refactored into GaussianSplatData
    
    /// Native macOS Background Removal using Vision
    private func removeBackground(from image: NSImage) async throws -> NSImage? {
        guard let tiffData = image.tiffRepresentation,
              let ciImage = CIImage(data: tiffData) else {
            print("Sharp Error: Failed to create CIImage for background removal")
            return nil 
        }
        
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(ciImage: ciImage)
        
        do {
            try handler.perform([request])
        } catch {
            print("Sharp Warning: Vision request failed: \(error.localizedDescription)")
            return nil
        }
        
        // Check if we got any results
        guard let results = request.results, !results.isEmpty else {
            print("Sharp Warning: Vision request succeeded but returned no mask results (no subject detected)")
            return nil
        }
        
        guard let result = results.first else {
            print("Sharp Warning: Vision result is empty")
            return nil
        }
        
        // Generate scaled mask for the image
        guard let maskPixelBuffer = try? result.generateScaledMaskForImage(forInstances: result.allInstances, from: handler) else {
            print("Sharp Warning: Failed to generate mask from Vision result")
            return nil
        }
        
        // Convert mask to CIImage
        let ciMask = CIImage(cvPixelBuffer: maskPixelBuffer)
        
        // Resize mask to match original image
        let originalExtent = ciImage.extent
        let scaleX = originalExtent.width / ciMask.extent.width
        let scaleY = originalExtent.height / ciMask.extent.height
        let scaledMask = ciMask.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        // Apply mask: Background becomes solid White (Standard 3DGS training convention)
        let whiteBg = CIImage(color: CIColor.white).cropped(to: originalExtent)
        
        let composition = CIFilter(name: "CIBlendWithMask")!
        composition.setValue(ciImage, forKey: kCIInputImageKey)
        composition.setValue(whiteBg, forKey: kCIInputBackgroundImageKey)
        composition.setValue(scaledMask, forKey: kCIInputMaskImageKey)
        
        guard let outCI = composition.outputImage else { 
            print("Sharp Error: CIFilter composition failed")
            return nil 
        }
        
        let context = CIContext()
        guard let outCG = context.createCGImage(outCI, from: originalExtent) else { 
            print("Sharp Error: Failed to create CGImage from CI result")
            return nil 
        }
        
        return NSImage(cgImage: outCG, size: image.size)
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
        // 0. Check if it's already an absolute path
        if name.hasPrefix("/") && FileManager.default.isExecutableFile(atPath: name) {
            return name
        }

        // 1. Check for executable in the app bundle (for distributed apps)
        if let bundlePath = Bundle.main.path(forResource: name, ofType: nil, inDirectory: "Resources/bin") {
             if FileManager.default.isExecutableFile(atPath: bundlePath) {
                 return bundlePath
             }
        }

        // 2. Check Application Support (for user-installed backend)
        if let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let appSupportPath = appSupport.appendingPathComponent("com.trond.SharpGlass/venv/bin/\(name)").path
            if FileManager.default.isExecutableFile(atPath: appSupportPath) {
                return appSupportPath
            }
        }

        // 3. Check current directory (if running in dev)
        let cwd = FileManager.default.currentDirectoryPath
        let localVenvPath = cwd + "/venv/bin/\(name)"
        if FileManager.default.isExecutableFile(atPath: localVenvPath) {
             return localVenvPath
        }
        
        // 4. Fallback to /usr/bin/env to find it in system PATH
        return "/usr/bin/env"
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
