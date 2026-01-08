import Foundation
import simd

/// Pure functional matrix utilities for 3D rendering.
/// Encapsulates coordinate system conversions and projection logic for unit testing.
struct MathUtils {
    
    /// creating a right-handed LookAt matrix.
    /// - Parameters:
    ///   - eye: Camera position
    ///   - target: Look-at target
    ///   - up: Up vector
    /// - Returns: 4x4 View Matrix
    static func matrix_look_at_right_hand(eye: vector_float3, target: vector_float3, up: vector_float3) -> matrix_float4x4 {
        let z = normalize(eye - target) // Forward is -Z, so Eye - Target = Positive Z axis (Backwards)
        let x = normalize(simd_cross(up, z)) // Right
        let y = cross(z, x) // Up
        
        // Standard LookAt
        return matrix_float4x4(columns: (
            vector_float4(x.x, y.x, z.x, 0),
            vector_float4(x.y, y.y, z.y, 0),
            vector_float4(x.z, y.z, z.z, 0),
            vector_float4(-dot(x, eye), -dot(y, eye), -dot(z, eye), 1)
        ))
    }
    
    /// Creates a standard perspective projection matrix for Metal's NDC.
    /// Metal NDC: X[-1,1], Y[-1,1], Z[0,1]
    static func makePerspectiveMatrix(fovRadians: Float, aspect: Float, near: Float, far: Float) -> matrix_float4x4 {
        let ys = 1 / tanf(fovRadians * 0.5)
        let xs = ys / aspect
        let zs = far / (near - far)
        
        return matrix_float4x4(columns: (
            vector_float4(xs, 0, 0, 0),
            vector_float4(0, ys, 0, 0),
            vector_float4(0, 0, zs, -1),
            vector_float4(0, 0, zs * near, 0)
        ))
    }
    
    /// Creates an asymmetric (off-axis) perspective projection matrix.
    /// This is used for "holographic" or eye-tracking based views.
    /// - Parameters:
    ///   - left, right, bottom, top: The coordinates of the frustum's near plane.
    ///   - near: Near clipping plane.
    ///   - far: Far clipping plane.
    /// - Returns: 4x4 Off-Axis Projection Matrix.
    static func makeOffAxisFrustumMatrix(left: Float, right: Float, bottom: Float, top: Float, near: Float, far: Float) -> matrix_float4x4 {
        let x = (2.0 * near) / (right - left)
        let y = (2.0 * near) / (top - bottom)
        let a = (right + left) / (right - left)
        let b = (top + bottom) / (top - bottom)
        let c = far / (near - far)
        let d = (far * near) / (near - far)
        
        return matrix_float4x4(columns: (
            vector_float4(x, 0, 0, 0),
            vector_float4(0, y, 0, 0),
            vector_float4(a, b, c, -1),
            vector_float4(0, 0, d, 0)
        ))
    }

    static func degreesToRadians(_ degrees: Float) -> Float {
        return degrees * .pi / 180
    }
}
