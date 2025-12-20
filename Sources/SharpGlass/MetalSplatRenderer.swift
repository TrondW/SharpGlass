import Foundation
import Metal
import MetalKit
import simd

// MARK: - Bridged Structs (Must match Shaders.metal)

struct SplatData {
    var position: SIMD4<Float>   // 16
    var color: SIMD4<Float>      // 32
    var scale: SIMD4<Float>      // 48 (includes opacity in w)
    var quaternion: SIMD4<Float> // 64
}

struct MetalUniforms {
    var viewMatrix: matrix_float4x4    // 64
    var projectionMatrix: matrix_float4x4 // 128
    var viewportSize: SIMD4<Float>     // 144 (xy=size, z=hasSH)
    var styleParams: SIMD4<Float>      // 160 (x=exposure, y=gamma, z=vignette)
    var cameraPosition: SIMD4<Float>   // 176
}

// MARK: - Renderer

@MainActor
class MetalSplatRenderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var pipelineState: MTLRenderPipelineState?
    private var depthStencilState: MTLDepthStencilState?
    
    // Geometry buffers
    // Geometry buffers
    private var splatBuffer: MTLBuffer?
    private var shBuffer: MTLBuffer?
    private var emptyBuffer: MTLBuffer?
    private var splatCount: Int = 0
    private var currentSplatID: UUID?
    
    // Camera state
    var cameraPosition: CameraPosition = .center
    var aspectRatio: Float = 1.0
    var viewportSize: SIMD2<Float> = SIMD2<Float>(1024, 1024)
    
    init(metalKitView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported")
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        super.init()
        
        metalKitView.device = device
        metalKitView.delegate = self
        metalKitView.colorPixelFormat = .bgra8Unorm
        metalKitView.depthStencilPixelFormat = .depth32Float
        
        self.emptyBuffer = device.makeBuffer(length: 4, options: .storageModeShared)
        
        buildPipeline(view: metalKitView)
    }
    
    private func buildPipeline(view: MTKView) {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct VertexOut {
            float4 position [[position]];
            float4 color;
            float2 uv;
            float3 conic;
        };

        struct Uniforms {
            float4x4 viewMatrix;
            float4x4 projectionMatrix;
            float4 viewportSize;  // xy=size, z=hasSH
            float4 styleParams;   // x=exposure, y=gamma, z=vignette
            float4 cameraPosition; // xyz=pos
        };

        struct SplatData {
            float4 position;
            float4 color;
            float4 scale;     // opacity in .w
            float4 quaternion;
        };

        // SH Basis Constants
        constant float SH_C1 = 0.4886025119f;
        constant float SH_C2[] = { 1.0925484306f, -1.0925484306f, 0.31539156525f, -1.0925484306f, 0.5462742153f };
        constant float SH_C3[] = { -0.5900435899f, 2.8906114426f, -0.4570457995f, 0.3731763326f, -0.4570457995f, 1.4453057213f, -0.5900435899f };

        float3 computeSH(float3 dir, constant float* sh, int idx) {
            float x = dir.x;
            float y = dir.y;
            float z = dir.z;

            // Band 1
            float3 result = 0;
            int offset = idx * 45;
            
            // Basis 1
            float3 d1 = float3(sh[offset+0], sh[offset+1], sh[offset+2]);
            result += SH_C1 * (-y) * d1; // Y1,-1
            
            float3 d2 = float3(sh[offset+3], sh[offset+4], sh[offset+5]);
            result += SH_C1 * (z) * d2;  // Y1,0
            
            float3 d3 = float3(sh[offset+6], sh[offset+7], sh[offset+8]);
            result += SH_C1 * (-x) * d3; // Y1,1
            
            // Band 2
            float xx = x*x, yy = y*y, zz = z*z;
            float xy = x*y, yz = y*z, xz = x*z;
            
            result += SH_C2[0] * (xy) * float3(sh[offset+9], sh[offset+10], sh[offset+11]);
            result += SH_C2[1] * (yz) * float3(sh[offset+12], sh[offset+13], sh[offset+14]);
            result += SH_C2[2] * (2.0f * zz - xx - yy) * float3(sh[offset+15], sh[offset+16], sh[offset+17]);
            result += SH_C2[3] * (xz) * float3(sh[offset+18], sh[offset+19], sh[offset+20]);
            result += SH_C2[4] * (xx - yy) * float3(sh[offset+21], sh[offset+22], sh[offset+23]);
            
            // Band 3
            result += SH_C3[0] * (3 * xx - yy) * y * float3(sh[offset+24], sh[offset+25], sh[offset+26]);
            result += SH_C3[1] * (x * z * y) * float3(sh[offset+27], sh[offset+28], sh[offset+29]);
            result += SH_C3[2] * y * (4 * zz - xx - yy) * float3(sh[offset+30], sh[offset+31], sh[offset+32]);
            result += SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * float3(sh[offset+33], sh[offset+34], sh[offset+35]);
            result += SH_C3[4] * x * (4 * zz - xx - yy) * float3(sh[offset+36], sh[offset+37], sh[offset+38]);
            result += SH_C3[5] * (xx - yy) * z * float3(sh[offset+39], sh[offset+40], sh[offset+41]);
            result += SH_C3[6] * x * (xx - 3 * yy) * float3(sh[offset+42], sh[offset+43], sh[offset+44]);
            
            return result;
        }

        float3x3 quadToMat(float4 q) {
            float r = q.x; float x = q.y; float y = q.z; float z = q.w;
            return float3x3(
                1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
                2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
            );
        }

        vertex VertexOut splatVertex(uint vertexID [[vertex_id]],
                                     uint instanceID [[instance_id]],
                                     const device SplatData* splats [[buffer(0)]],
                                     constant Uniforms& uniforms [[buffer(1)]],
                                     constant uint* sortOrders [[buffer(2)]],
                                     constant float* shs [[buffer(3)]]) {
            VertexOut out;
            uint sortedIndex = sortOrders[instanceID];
            SplatData splat = splats[sortedIndex];
            
            float4 p_view = uniforms.viewMatrix * float4(splat.position.xyz, 1.0);
            
            float3 finalColor = splat.color.rgb;
            
            // View Direction (Object Center to Camera)
            float3 dir = normalize(splat.position.xyz - uniforms.cameraPosition.xyz);
            
            if (uniforms.viewportSize.z > 0.5f) {
                 finalColor += computeSH(dir, shs, sortedIndex);
            }
            
            out.color = float4(max(0.0f, min(1.0f, finalColor)), splat.scale.w);

            float3x3 R = quadToMat(splat.quaternion);
            float3x3 S = float3x3(0);
            S[0][0] = splat.scale.x; S[1][1] = splat.scale.y; S[2][2] = splat.scale.z;
            float3x3 M = R * S;
            float3x3 Sigma = M * transpose(M);

            float f = uniforms.projectionMatrix[1][1];
            float x = p_view.x; float y = p_view.y; float z = p_view.z;
            
            float3x3 J = float3x3(
                f / z,   0,       -(f * x) / (z * z),
                0,       f / z,   -(f * y) / (z * z),
                0,       0,       0
            );
            
            float3x3 W = float3x3(uniforms.viewMatrix[0].xyz, uniforms.viewMatrix[1].xyz, uniforms.viewMatrix[2].xyz);
            float3x3 T = J * W;
            float3x3 cov2D = T * Sigma * transpose(T);
            
            cov2D[0][0] += 0.3f;
            cov2D[1][1] += 0.3f;

            float det = cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[0][1];
            if (det <= 0.0f) { out.position = 0; return out; }
            float mid = 0.5f * (cov2D[0][0] + cov2D[1][1]);
            float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
            float radius = ceil(3.0f * sqrt(lambda1));

            float2 localQuad[] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
            float2 offset = localQuad[vertexID] * radius;
            
            float4 p_clip = uniforms.projectionMatrix * p_view;
            float2 p_ndc = p_clip.xy / p_clip.w;
            float2 ndc_offset = offset * (2.0f / uniforms.viewportSize.xy);
            
            out.position = float4(p_ndc + ndc_offset, p_clip.z / p_clip.w, 1.0);
            out.uv = offset;
            
            float inv_det = 1.0f / det;
            out.conic = float3(cov2D[1][1] * inv_det, -cov2D[0][1] * inv_det, cov2D[0][0] * inv_det);
            
            return out;
        }

        // ACES Tone Mapping
        float3 aces_tonemap(float3 x) {
            float a = 2.51f; float b = 0.03f; float c = 2.43f; float d = 0.59f; float e = 0.14f;
            return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
        }

        fragment float4 splatFragment(VertexOut in [[stage_in]], constant Uniforms& uniforms [[buffer(1)]]) {
            float2 d = in.uv;
            float power = -0.5f * (in.conic.x * d.x * d.x + 2.0f * in.conic.y * d.x * d.y + in.conic.z * d.y * d.y);
            if (power > 0.0f) discard_fragment();
            float alpha = min(0.99f, in.color.a * exp(power));
            if (alpha < 1.0f/255.0f) discard_fragment();
            
            float3 color = in.color.rgb;
            
            // Apply Grading
            float exposure = uniforms.styleParams.x;
            float gamma = uniforms.styleParams.y;
            float vignetteStrength = uniforms.styleParams.z;
            
            color *= pow(2.0f, exposure);
            float dist = length(d);
            float vignette = 1.0f - smoothstep(0.5f, 1.5f, dist) * vignetteStrength;
            color *= vignette;
            
            color = aces_tonemap(color);
            color = pow(color, float3(1.0f / max(0.1f, gamma)));
            
            return float4(color, alpha);
        }
        """
        
        do {
            // Load from Source string (Reliable fallback)
            let library = try device.makeLibrary(source: shaderSource, options: nil)
            
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.label = "Splat Pipeline"
            descriptor.vertexFunction = library.makeFunction(name: "splatVertex")
            descriptor.fragmentFunction = library.makeFunction(name: "splatFragment")
            descriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
            descriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat
            
            descriptor.colorAttachments[0].isBlendingEnabled = true
            descriptor.colorAttachments[0].rgbBlendOperation = .add
            descriptor.colorAttachments[0].alphaBlendOperation = .add
            
            // Accumulated Alpha (Standard Over Operator pre-multiplied)
            // But we have non-premultiplied color in shader?
            // Shader returns (color.rgb, alpha).
            // Destination is standard alpha blending: SrcAlpha + (1-SrcAlpha)*Dst
            // RGB: Src.RGB * Src.A + Dst.RGB * (1-Src.A)
            
            descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            descriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
            descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            
            pipelineState = try device.makeRenderPipelineState(descriptor: descriptor)
            
            let depthDescriptor = MTLDepthStencilDescriptor()
            depthDescriptor.depthCompareFunction = .lessEqual
            depthDescriptor.isDepthWriteEnabled = true
            depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)
            
            view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        } catch {
            print("Metal Error: \(error)")
        }
    }
    
    // Sort State
    private var sortBuffer: MTLBuffer?
    private var splatPositions: [SIMD3<Float>] = [] // CPU Copy for Depth Calc
    
    func load(gaussians: GaussianSplatData) {
        if gaussians.id == currentSplatID { return }
        
        var splats: [SplatData] = []
        // Keep a separate position array for fast sorting without striding full struct
        splatPositions.removeAll(keepingCapacity: true)
        
        for i in 0..<gaussians.pointCount {
            splats.append(SplatData(
                position: SIMD4<Float>(gaussians.positions[i].x, gaussians.positions[i].y, gaussians.positions[i].z, 1.0),
                color: SIMD4<Float>(gaussians.colors[i].x, gaussians.colors[i].y, gaussians.colors[i].z, 1.0),
                scale: SIMD4<Float>(gaussians.scales[i].x, gaussians.scales[i].y, gaussians.scales[i].z, gaussians.opacities[i]),
                quaternion: gaussians.rotations[i]
            ))
            splatPositions.append(gaussians.positions[i])
        }
        
        self.splatCount = splats.count
        let size = splats.count * MemoryLayout<SplatData>.stride
        self.splatBuffer = device.makeBuffer(bytes: splats, length: size, options: .storageModeShared)
        
        // Initialize indices [0, 1, 2...]
        let indices = Array(0..<UInt32(splatCount))
        self.sortBuffer = device.makeBuffer(bytes: indices, length: splatCount * MemoryLayout<UInt32>.stride, options: .storageModeShared)
        
        // SH Buffer
        if !gaussians.shs.isEmpty {
            self.shBuffer = device.makeBuffer(bytes: gaussians.shs, length: gaussians.shs.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        } else {
            self.shBuffer = nil
        }
        
        self.currentSplatID = gaussians.id
        
        print("Metal: Loaded \(splatCount) splats. SH Data: \(self.shBuffer != nil)")
    }
    
    // Sorting State
    private var isSorting = false
    private let sortQueue = DispatchQueue(label: "com.sharpglass.sort", qos: .userInteractive)
    
    // Style State
    var exposure: Float = 0
    var gamma: Float = 2.2
    var vignetteStrength: Float = 0.5
    
    private func depthSort() {
        guard splatCount > 0, !isSorting else { return }
        
        // Capture state for background thread
        let eye = SIMD3<Float>(Float(cameraPosition.x), Float(cameraPosition.y), Float(cameraPosition.z))
        let target: SIMD3<Float>
        if let t = cameraPosition.target {
            target = SIMD3<Float>(Float(t.x), Float(t.y), Float(t.z))
        } else {
            target = eye + SIMD3<Float>(0, 0, -1)
        }
        let forward = normalize(target - eye)
        
        // Capture immutable copies of data needed for sort
        // Note: splatPositions is constant once loaded, so safe to read if we don't reload mid-sort.
        // But to be safe vs load(), load() should cancel or block. 
        // For now, assuming load() happens rarely.
        let count = self.splatCount
        let positions = self.splatPositions 
        
        isSorting = true
        
        sortQueue.async { [weak self] in
            guard let self = self else { return }
            
            // 1. Calculate Depths
            // Create a temporary array of (index, depth)
            // Using a simple struct to hold index+depth
            struct DepthItem {
                var index: UInt32
                var depth: Float
            }
            
            var depths = [DepthItem]()
            depths.reserveCapacity(count)
            
            // Optimized Loop (UnsafeBufferPointer for speed)
            positions.withUnsafeBufferPointer { buffer in
                guard let ptr = buffer.baseAddress else { return }
                for i in 0..<count {
                    let p = ptr[i]
                    // Dot product: (p - eye) . forward
                    let dx = p.x - eye.x
                    let dy = p.y - eye.y
                    let dz = p.z - eye.z
                    let depth = dx * forward.x + dy * forward.y + dz * forward.z
                    depths.append(DepthItem(index: UInt32(i), depth: depth))
                }
            }
            
            // 2. Sort (Back-to-Front)
            // Descending depth
            depths.sort { $0.depth > $1.depth }
            
            // 3. Extract Indices
            let sortedIndices = depths.map { $0.index }
            
            // 4. Upload to GPU (Main Thread)
            DispatchQueue.main.async {
                // Check if splat count changed (reloaded)
                if self.splatCount == count {
                    // Update GPU buffer
                    // using replaceRegion or making a new buffer? 
                    // contents().copyMemory is fastest for shared buffer.
                    if let sortBuffer = self.sortBuffer {
                        let contents = sortBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
                        sortedIndices.withUnsafeBufferPointer { srcBuf in
                            if let src = srcBuf.baseAddress {
                                contents.update(from: src, count: count)
                            }
                        }
                        
                        // Notify GPU buffer modified
                        sortBuffer.didModifyRange(0..<count * MemoryLayout<UInt32>.stride)
                    }
                }
                self.isSorting = false
            }
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        aspectRatio = Float(size.width / size.height)
        viewportSize = SIMD2<Float>(Float(size.width), Float(size.height))
    }
    
    func draw(in view: MTKView) {
        // ... (existing implementation) ...
        guard let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor,
              let pipeline = pipelineState,
              let buffer = splatBuffer,
              let sortBuf = sortBuffer else { return }
        
        depthSort()
        
        // ... setup matrices ...
        let viewMatrix = makeViewMatrix()
        let projMatrix = makePerspectiveMatrix(fovRadians: degreesToRadians(60), aspect: aspectRatio, near: 0.1, far: 1000)
        var uniforms = MetalUniforms(
            viewMatrix: viewMatrix,
            projectionMatrix: projMatrix,
            viewportSize: SIMD4<Float>(Float(viewportSize.x), Float(viewportSize.y), shBuffer != nil ? 1.0 : 0.0, 0.0),
            styleParams: SIMD4<Float>(exposure, gamma, vignetteStrength, 0.0),
            cameraPosition: SIMD4<Float>(Float(cameraPosition.x), Float(cameraPosition.y), Float(cameraPosition.z), 1.0)
        )
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        
        encoder.setRenderPipelineState(pipeline)
        if let ds = depthStencilState { encoder.setDepthStencilState(ds) }
        
        encoder.setVertexBuffer(buffer, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<MetalUniforms>.stride, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<MetalUniforms>.stride, index: 1)
        encoder.setVertexBuffer(sortBuf, offset: 0, index: 2)
        
        if let shBuf = shBuffer {
            encoder.setVertexBuffer(shBuf, offset: 0, index: 3)
        } else {
            encoder.setVertexBuffer(emptyBuffer, offset: 0, index: 3)
        }
        
        if splatCount > 0 {
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: splatCount)
        }
        
        encoder.endEncoding()
        
        // IF capturing, we need to blit or just rely on drawable if feasible? 
        // For simple offline render, we can read back from drawable.texture
        // But we need to ensure command buffer completion
        
        if isCapturingFrame {
            let capturedTexture = drawable.texture
            commandBuffer.addCompletedHandler { [weak self] _ in
                self?.lastCapturedTexture = capturedTexture
            }
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    // Capture State
    var isCapturingFrame = false
    var lastCapturedTexture: MTLTexture?
    
    // Synchronous Snapshot for Offline Render Loop
    // This blocks Main Thread but that's what we want for "Offline" perfect rendering
    func snapshot(at size: CGSize) -> CVPixelBuffer? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil } 
        // We reuse the existing device actually
        
        // 1. Create a transient texture for offscreen rendering if not using the view
        // But we are rendering to the view.
        // For offline render, we might want to force a specific resolution (1920x1080) regardless of window size.
        // Let's assume we render to the view and capture its drawable for now to keep it simple.
        // Or better: Create a separate Texture to render to?
        
        // SIMPLIFICATION:
        // We will just read the last captured texture from the draw loop.
        // The render loop needs to run once.
        
        guard let texture = lastCapturedTexture else { return nil }
        
        let width = texture.width
        let height = texture.height
        
        var cvPixelBuffer: CVPixelBuffer?
        let attributes = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferMetalCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attributes, &cvPixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = cvPixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        
        let region = MTLRegionMake2D(0, 0, width, height)
        texture.getBytes(baseAddress, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
        
        return buffer
    }
    
    // MARK: - Camera Math
    
    private func makeViewMatrix() -> matrix_float4x4 {
        // Standard LookAt Matrix
        let eye = vector_float3(Float(cameraPosition.x), Float(cameraPosition.y), Float(cameraPosition.z))
        
        let target: vector_float3
        if let t = cameraPosition.target {
            target = vector_float3(Float(t.x), Float(t.y), Float(t.z))
        } else {
            // Fallback for legacy calls (should generally not be hit if VM is correct)
            // Use rotation fields as Euler angles (Orbit default center at zero?)
            // Just look forward -Z from Eye
            target = eye + vector_float3(0, 0, -1)
        }
        
        let up = vector_float3(0, 1, 0) // World Up is +Y (Metal standard for camera basis)
        
        return matrix_look_at_right_hand(eye: eye, target: target, up: up)
    }
    
    private func makePerspectiveMatrix(fovRadians: Float, aspect: Float, near: Float, far: Float) -> matrix_float4x4 {
        // OpenCV Projection:
        // x_ndc = x_view / z_view
        // y_ndc = y_view / z_view (OpenCV Y is Down, Screen Y is Down... wait)
        
        // Metal NDC:
        // x: [-1, 1] Right
        // y: [-1, 1] Up (Bottom is -1, Top is +1)
        // z: [0, 1] Into screen
        
        // Input View Space (OpenCV):
        // +X: Right
        // +Y: Down
        // +Z: Forward
        
        // Projection needs:
        // x_clip = x_view (Right maps to Right)
        // y_clip = -y_view (Down maps to Down, which is Negative in Metal NDC)
        // z_clip = map z_view from [near, far] to [0, 1]
        
        let ys = 1 / tanf(fovRadians * 0.5)
        let xs = ys / aspect
        let zs = far / (near - far)
        
        return matrix_float4x4(columns: (
            vector_float4(-xs, 0, 0, 0),   // x scale (flipped to fix mirroring)
            vector_float4(0, -ys, 0, 0),   // y scale (negated to flip Y-down to Y-up)
            vector_float4(0, 0, zs, -1),   // z remapping (standard RH projection)
            vector_float4(0, 0, zs * near, 0)
        ))
    }
    
    // Matrix Helpers
    
    private func matrix_look_at_right_hand(eye: vector_float3, target: vector_float3, up: vector_float3) -> matrix_float4x4 {
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
    
    private func degreesToRadians(_ degrees: Float) -> Float { degrees * .pi / 180 }
}

