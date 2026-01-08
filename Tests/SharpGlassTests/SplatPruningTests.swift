import Testing
import Foundation
@testable import SharpGlassLibrary

@Suite("Splat Pruning Tests")
struct SplatPruningTests {
    
    @Test("Pruning preserves highest quality splats")
    @MainActor
    func pruningPreservesHighQuality() throws {
        // Create test dataset with varying quality
        let positions = [
            SIMD3<Float>(0, 0, 0),    // High quality (center, high opacity)
            SIMD3<Float>(1, 1, 1),    // Medium quality
            SIMD3<Float>(100, 100, 100), // Low quality (far outlier)
            SIMD3<Float>(0.5, 0.5, 0.5), // High quality (center, high opacity)
        ]
        
        let colors = Array(repeating: SIMD3<Float>(0.5, 0.5, 0.5), count: 4)
        
        let opacities: [Float] = [
            0.9,  // High
            0.5,  // Medium
            0.1,  // Low
            0.95, // Very high
        ]
        
        let scales = [
            SIMD3<Float>(0.1, 0.1, 0.1),   // Good scale
            SIMD3<Float>(0.5, 0.5, 0.5),   // Good scale
            SIMD3<Float>(20.0, 20.0, 20.0), // Artifact (too large)
            SIMD3<Float>(0.05, 0.05, 0.05), // Good scale
        ]
        
        let rotations = Array(repeating: SIMD4<Float>(1, 0, 0, 0), count: 4)
        let _ : [Float] = []
        
        // Create splat data using private initializer via reflection
        // Since we can't access private init directly, we'll create via PLY
        let plyData = try createTestPLY(
            positions: positions,
            colors: colors,
            opacities: opacities,
            scales: scales,
            rotations: rotations
        )
        
        let splat = try GaussianSplatData(data: plyData)
        
        // Prune to 2 splats
        let pruned = splat.pruned(maxCount: 2)
        
        #expect(pruned.pointCount == 2)
        
        // The two highest quality splats should be indices 0 and 3
        // (high opacity, good scale, near center)
        // We can't directly check indices, but we can verify properties
        let prunedOpacities = pruned.opacities
        #expect(prunedOpacities.allSatisfy { $0 >= 0.5 })
    }
    
    @Test("Pruning below limit returns same splat")
    @MainActor
    func pruningBelowLimitReturnsOriginal() throws {
        let positions = [SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 1, 1)]
        let colors = Array(repeating: SIMD3<Float>(0.5, 0.5, 0.5), count: 2)
        let opacities: [Float] = [0.8, 0.9]
        let scales = Array(repeating: SIMD3<Float>(0.1, 0.1, 0.1), count: 2)
        let rotations = Array(repeating: SIMD4<Float>(1, 0, 0, 0), count: 2)
        
        let plyData = try createTestPLY(
            positions: positions,
            colors: colors,
            opacities: opacities,
            scales: scales,
            rotations: rotations
        )
        
        let splat = try GaussianSplatData(data: plyData)
        let pruned = splat.pruned(maxCount: 10)
        
        // Should return same splat (same ID)
        #expect(pruned.id == splat.id)
        #expect(pruned.pointCount == 2)
    }
    
    @Test("Pruning removes low opacity splats")
    @MainActor
    func pruningRemovesLowOpacity() throws {
        let positions = Array(repeating: SIMD3<Float>(0, 0, 0), count: 5)
        let colors = Array(repeating: SIMD3<Float>(0.5, 0.5, 0.5), count: 5)
        let opacities: [Float] = [0.95, 0.9, 0.5, 0.2, 0.1]
        let scales = Array(repeating: SIMD3<Float>(0.1, 0.1, 0.1), count: 5)
        let rotations = Array(repeating: SIMD4<Float>(1, 0, 0, 0), count: 5)
        
        let plyData = try createTestPLY(
            positions: positions,
            colors: colors,
            opacities: opacities,
            scales: scales,
            rotations: rotations
        )
        
        let splat = try GaussianSplatData(data: plyData)
        let pruned = splat.pruned(maxCount: 3)
        
        #expect(pruned.pointCount == 3)
        
        // All remaining splats should have relatively high opacity
        let avgOpacity = pruned.opacities.reduce(0, +) / Float(pruned.pointCount)
        #expect(avgOpacity > 0.5)
    }
    
    @Test("Pruning removes oversized splats")
    @MainActor
    func pruningRemovesOversizedSplats() throws {
        let positions = Array(repeating: SIMD3<Float>(0, 0, 0), count: 4)
        let colors = Array(repeating: SIMD3<Float>(0.5, 0.5, 0.5), count: 4)
        let opacities = Array(repeating: Float(0.8), count: 4)
        
        let scales = [
            SIMD3<Float>(0.1, 0.1, 0.1),    // Good
            SIMD3<Float>(0.5, 0.5, 0.5),    // Good
            SIMD3<Float>(50.0, 50.0, 50.0), // Artifact
            SIMD3<Float>(0.05, 0.05, 0.05), // Good
        ]
        
        let rotations = Array(repeating: SIMD4<Float>(1, 0, 0, 0), count: 4)
        
        let plyData = try createTestPLY(
            positions: positions,
            colors: colors,
            opacities: opacities,
            scales: scales,
            rotations: rotations
        )
        
        let splat = try GaussianSplatData(data: plyData)
        let pruned = splat.pruned(maxCount: 3)
        
        #expect(pruned.pointCount == 3)
        
        // All remaining splats should have reasonable scales
        for scale in pruned.scales {
            let avgScale = (scale.x + scale.y + scale.z) / 3.0
            #expect(avgScale < 10.0)
        }
    }
    
    // Helper function to create test PLY data
    private func createTestPLY(
        positions: [SIMD3<Float>],
        colors: [SIMD3<Float>],
        opacities: [Float],
        scales: [SIMD3<Float>],
        rotations: [SIMD4<Float>]
    ) throws -> Data {
        let count = positions.count
        
        let header = """
        ply
        format binary_little_endian 1.0
        element vertex \(count)
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        property float f_dc_0
        property float f_dc_1
        property float f_dc_2
        property float opacity
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        end_header
        
        """
        
        var data = header.data(using: .utf8)!
        
        // Convert colors to SH DC coefficients
        let shC: Float = 0.28209479177387814
        
        for i in 0..<count {
            // Position
            withUnsafeBytes(of: positions[i].x) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: positions[i].y) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: positions[i].z) { data.append(contentsOf: $0) }
            
            // Normals (dummy)
            let zero: Float = 0
            withUnsafeBytes(of: zero) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: zero) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: zero) { data.append(contentsOf: $0) }
            
            // Colors as SH DC
            let dc0 = (colors[i].x - 0.5) / shC
            let dc1 = (colors[i].y - 0.5) / shC
            let dc2 = (colors[i].z - 0.5) / shC
            withUnsafeBytes(of: dc0) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: dc1) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: dc2) { data.append(contentsOf: $0) }
            
            // Opacity (inverse sigmoid)
            let logitOpacity = -log((1.0 / opacities[i]) - 1.0)
            withUnsafeBytes(of: logitOpacity) { data.append(contentsOf: $0) }
            
            // Scales (log space)
            let logScale0 = log(scales[i].x)
            let logScale1 = log(scales[i].y)
            let logScale2 = log(scales[i].z)
            withUnsafeBytes(of: logScale0) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: logScale1) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: logScale2) { data.append(contentsOf: $0) }
            
            // Rotations
            withUnsafeBytes(of: rotations[i].x) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: rotations[i].y) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: rotations[i].z) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: rotations[i].w) { data.append(contentsOf: $0) }
        }
        
        return data
    }
}
