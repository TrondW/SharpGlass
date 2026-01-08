import Foundation
@preconcurrency import Metal
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
    var invViewMatrix: matrix_float4x4    // 192
    var invProjectionMatrix: matrix_float4x4 // 256
    var viewportSize: SIMD4<Float>     // 272 (xy=size, z=hasSH)
    var styleParams: SIMD4<Float>      // 288 (x=exposure, y=gamma, z=vignette, w=splatScale)
    var cameraPosition: SIMD4<Float>   // 304 (xyz=pos, w=colorMode)
    var colorParams: SIMD4<Float>      // 320 (x=saturation, y, z, w)
    var affordanceParams: SIMD4<Float> // 336 (xyz = orbitTarget, w = isNavigating)
    var headPosition: SIMD4<Float>     // 352 (xyz = pos in cm, w = isHolographic)
}

// MARK: - Renderer

/// A high-performance Metal renderer for 3D Gaussian Splats.
///
/// **Pipeline Overview:**
/// 1. **Sort**: GPU-based Radix Sort orders splats back-to-front based on view depth.
/// 2. **Vertex**: Projects 3D Gaussians to 2D screen space (NDC) and computes covariance.
/// 3. **Fragment**: Renders the Gaussian splat equations, handling alpha blending and spherical harmonics.
///
/// **Coordinate System:**
/// - World: Right-handed, Y-Up (standard 3D).
/// - ml-sharp Data: Often Y-Down, so we adjust the Up vector in `makeViewMatrix`.
/// - NDC (Metal): X[-1, 1], Y[-1, 1] (Bottom-Up), Z[0, 1].
@MainActor
class MetalSplatRenderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var pipelineState: MTLRenderPipelineState?
    private var gridPipelineState: MTLRenderPipelineState?
    private var depthStencilState: MTLDepthStencilState?
    
    // Compute Pipeline States for Sorting
    private var calcDepthsPipelineState: MTLComputePipelineState?
    private var histogramPipelineState: MTLComputePipelineState?
    private var prefixSumPipelineState: MTLComputePipelineState?
    private var scatterPipelineState: MTLComputePipelineState?
    private var clearBufferPipelineState: MTLComputePipelineState?
    
    // Sorting Buffers (Ping-Pong)
    private var sortKeysBufferA: MTLBuffer?
    private var sortKeysBufferB: MTLBuffer?
    private var sortIndicesBufferA: MTLBuffer?
    private var sortIndicesBufferB: MTLBuffer?
    private var histogramBuffer: MTLBuffer?
    private var prefixSumBuffer: MTLBuffer?
    
    private var splatBuffer: MTLBuffer?
    private var shBuffer: MTLBuffer?
    private var emptyBuffer: MTLBuffer?
    private var splatCount: Int = 0
    private var currentSplatID: UUID?
    
    // Safety limit to prevent GPU memory crashes
    private static let MAX_SPLAT_COUNT = 2_000_000
    
    // Camera state
    var cameraPosition: CameraPosition = .center
    var orbitTarget: SIMD3<Double> = SIMD3(0, 0, 0)
    var isNavigating: Bool = false
    var isHolographic: Bool = false
    var headPosition: SIMD3<Float> = SIMD3<Float>(0, 0, 60)
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
        // Use bgra10_xr for HDR/EDR support on modern Macs
        metalKitView.colorPixelFormat = .bgra10_xr
        metalKitView.depthStencilPixelFormat = .depth32Float
        
        if let layer = metalKitView.layer as? CAMetalLayer {
            layer.wantsExtendedDynamicRangeContent = true
            layer.colorspace = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)
        }
        
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
            float4x4 invViewMatrix;
            float4x4 invProjectionMatrix;
            float4 viewportSize;  // xy=size, z=hasSH
            float4 styleParams;   // x=exposure, y=gamma, z=vignette, w=splatScale
            float4 cameraPosition; // xyz=pos, w=colorMode
            float4 colorParams;    // x=saturation
            float4 affordanceParams; // xyz=orbitTarget, w=isNavigating
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

        // Grid & Pivot Affordances
        vertex VertexOut gridVertex(uint vid [[vertex_id]], constant Uniforms &uniforms [[buffer(1)]]) {
            // Full screen quad
            float2 positions[4] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1) };
            float2 uv = positions[vid];
            
            VertexOut out;
            out.position = float4(uv, 1.0, 1.0);
            out.uv = uv;
            return out;
        }

        /// Renders an infinite ground grid on the Y=0 plane using raycasting.
        /// Also projects and renders the yellow orbit pivot point during navigation.
        fragment float4 gridFragment(VertexOut in [[stage_in]], constant Uniforms &uniforms [[buffer(1)]]) {
            // Raycast Ground Grid
            float4x4 invView = uniforms.invViewMatrix;
            float4x4 invProj = uniforms.invProjectionMatrix;
            
            float4 ndc = float4(in.uv, 0.0, 1.0); // Near plane
            float4 nearWorld = invView * invProj * ndc;
            nearWorld /= nearWorld.w;
            
            float4 camPos = invView * float4(0,0,0,1);
            float3 rayDir = normalize(nearWorld.xyz - camPos.xyz);
            
            // Plane intersection (Y=+2.0, below splats in ml-sharp Y-down coords)
            float t = (2.0 - camPos.y) / rayDir.y;
            if (t < 0) discard_fragment();
            
            float3 worldPos = camPos.xyz + rayDir * t;
            
            // Fading
            float dist = length(worldPos.xz);
            float alpha = 1.0 - smoothstep(5.0, 20.0, dist);
            if (alpha <= 0) discard_fragment();
            
            // Grid lines
            float2 grid = abs(fract(worldPos.xz - 0.5) - 0.5) / (fwidth(worldPos.xz) + 0.001);
            float line = min(grid.x, grid.y);
            float gridVal = 1.0 - min(line, 1.0);
            
            // Axes
            float axisX = 1.0 - min(abs(worldPos.z) / (fwidth(worldPos.z) + 0.001), 1.0);
            float axisZ = 1.0 - min(abs(worldPos.x) / (fwidth(worldPos.x) + 0.001), 1.0);
            
            float4 color = float4(0.3, 0.3, 0.3, 0.2 * alpha);
            if (gridVal > 0) color = float4(0.5, 0.5, 0.5, 0.3 * alpha);
            if (axisX > 0) color = float4(1.0, 0.2, 0.2, 0.5 * alpha); // X axis
            if (axisZ > 0) color = float4(0.2, 0.2, 1.0, 0.5 * alpha); // Z axis
            
            // Pivot Point Overlay
            float3 orbitTarget = uniforms.affordanceParams.xyz;
            bool isNavigating = uniforms.affordanceParams.w > 0.5;
            
            if (isNavigating) {
                // Project orbit target
                float4 targetClip = uniforms.projectionMatrix * uniforms.viewMatrix * float4(orbitTarget, 1.0);
                float2 targetNDC = targetClip.xy / targetClip.w;
                float d = length(in.uv - targetNDC);
                float pivot = 1.0 - smoothstep(0.005, 0.007, d);
                if (pivot > 0) color = mix(color, float4(1.0, 1.0, 0.0, 0.8), pivot);
            }
            
            return color;
        }

        /// Main splat vertex shader. Projects 3D Gaussians into 2D screenspace.
        /// Handles SH color computation and covariance-to-ellipse math.
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
            float s = uniforms.styleParams.w; // splatScale
            S[0][0] = splat.scale.x * s; S[1][1] = splat.scale.y * s; S[2][2] = splat.scale.z * s;
            float3x3 M = R * S;
            float3x3 Sigma = M * transpose(M);

            float focal_y = uniforms.projectionMatrix[1][1] * uniforms.viewportSize.y / 2.0f;
            float focal_x = uniforms.projectionMatrix[0][0] * uniforms.viewportSize.x / 2.0f;
            
            float x = p_view.x; float y = p_view.y; float z = p_view.z;
            
            float3x3 J = float3x3(
                focal_x / z,   0,       -(focal_x * x) / (z * z),
                focal_y / z,   0,       -(focal_y * y) / (z * z), // Swapped row/column order for float3x3? 
                0,             0,       0
            );
            // Metal float3x3 is column-major. 
            // We want:
            // [ fx/z, 0, -fx*x/z^2 ]
            // [ 0, fy/z, -fy*y/z^2 ]
            // [ 0, 0, 0 ]
            
            J = float3x3(
                float3(focal_x / z, 0, 0),
                float3(0, focal_y / z, 0),
                float3(-(focal_x * x) / (z * z), -(focal_y * y) / (z * z), 0)
            );
            
            float3x3 W = float3x3(uniforms.viewMatrix[0].xyz, uniforms.viewMatrix[1].xyz, uniforms.viewMatrix[2].xyz);
            float3x3 T = J * W;
            float3x3 cov2D = T * Sigma * transpose(T);
            
            // Mip-Splatting: 0.3 pixel anti-aliasing filter
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
            if (p_clip.w < 0.05f) { out.position = float4(0,0,0,1); return out; }
            float2 p_ndc = p_clip.xy / p_clip.w;
            float2 ndc_offset = offset * (2.0f / uniforms.viewportSize.xy);
            
            // Critical: We use a small Z offset for splats to distinguish them from the grid 
            // if we were using depth testing, but for over-blending we just need to preserve 
            // the relative order from sorting.
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
        
            // --- TONE MAPPING ---
            float exposure = uniforms.styleParams.x;
            float gamma = uniforms.styleParams.y;
            float vignetteStrength = uniforms.styleParams.z;
            float saturationVal = uniforms.colorParams.x;
            int colorMode = (int)uniforms.cameraPosition.w;
            
            color *= exp2(exposure);
            float dist = length(d);
            float vignette = 1.0f - smoothstep(0.5f, 1.5f, dist) * vignetteStrength;
            color *= vignette;
            
            // Saturation adjustment in Linear space
            float luma = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
            color = mix(float3(luma), color, saturationVal);
            
            if (colorMode == 1) { // FILMIC (ACES)
                // Refined ACES Filmic Curve (Narkowicz 2015)
                const float a = 2.51f;
                const float b = 0.03f;
                const float c = 2.43f;
                const float d = 0.59f;
                const float e = 0.14f;
                color = (color * (a * color + b)) / (color * (c * color + d) + e);
            }
            
            // Note: Since we use bgra10_xr (Extended Linear sRGB), 
            // the OS handles display gamma. We output values that represent 
            // the intended color in that space.
            // If the input was sRGB and we want to match, we usually want 
            // a gamma close to 1.0 if our input data is already sRGB-ish, 
            // but for a truly linear pipeline, we might need a custom curve.
            color = pow(max(0.0001f, color), max(0.01f, gamma));
            
            return float4(color, alpha);
        }

        // --- GPU SORT KERNELS (RADIX SORT) ---
        
        kernel void calculateDepths(uint id [[thread_position_in_grid]],
                                   device const SplatData* splats [[buffer(0)]],
                                   device uint* keys [[buffer(1)]],
                                   device uint* indices [[buffer(2)]],
                                   constant Uniforms& uniforms [[buffer(3)]]) {
            if (id >= uint(uniforms.viewportSize.w)) return;
            
            float4 p_view = uniforms.viewMatrix * float4(splats[id].position.xyz, 1.0);
            float depth = p_view.z;
            
            // BACK-TO-FRONT SORTING (Most distant first)
            // Range of Z is [Near, Far] where Near is e.g. -0.1 and Far is e.g. -100.
            // We want to sort Ascending: -100.0, -99.9, ..., -0.1.
            // Standard float-to-uint mapping for monotonic sorting:
            uint u = as_type<uint>(depth);
            uint mask = (u >> 31) ? ~u : u ^ 0x80000000;
            keys[id] = mask;
            indices[id] = id;
        }

        kernel void histogram(uint id [[thread_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint bid [[threadgroup_position_in_grid]],
                             device const uint* keys [[buffer(0)]],
                             device uint* histograms [[buffer(1)]],
                             constant uint& bitOffset [[buffer(2)]],
                             constant uint& count [[buffer(3)]]) {
            threadgroup atomic_uint localHist[256];
            if (tid < 256) atomic_store_explicit(&localHist[tid], 0, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            if (id < count) {
                uint key = keys[id];
                uint bucket = (key >> bitOffset) & 0xFF;
                atomic_fetch_add_explicit(&localHist[bucket], 1, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            if (tid < 256) {
                histograms[bid * 256 + tid] = atomic_load_explicit(&localHist[tid], memory_order_relaxed);
            }
        }

        kernel void prefixSum(uint id [[thread_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               device uint* histograms [[buffer(0)]],
                               constant uint& threadgroupCount [[buffer(1)]]) {
            // id is 0..255 (one thread per bucket)
            if (tid >= 256) return;
            
            threadgroup uint bucketTotals[256];
            
            // 1. Calculate the total for this bucket across all threadgroups
            uint total = 0;
            for (uint j = 0; j < threadgroupCount; j++) {
                total += histograms[j * 256 + tid];
            }
            bucketTotals[tid] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // 2. Prefix sum the bucket totals (serial for 256 elements is fine)
            if (tid == 0) {
                uint sum = 0;
                for (uint i = 0; i < 256; i++) {
                    uint val = bucketTotals[i];
                    bucketTotals[i] = sum;
                    sum += val;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // 3. Write out the final offsets for each threadgroup in this bucket
            uint bucketStart = bucketTotals[tid];
            uint runningSum = bucketStart;
            for (uint j = 0; j < threadgroupCount; j++) {
                uint val = histograms[j * 256 + tid];
                histograms[j * 256 + tid] = runningSum;
                runningSum += val;
            }
        }

        kernel void scatter(uint id [[thread_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint bid [[threadgroup_position_in_grid]],
                            device const uint* srcKeys [[buffer(0)]],
                            device const uint* srcIndices [[buffer(1)]],
                            device uint* dstKeys [[buffer(2)]],
                            device uint* dstIndices [[buffer(3)]],
                            device uint* histograms [[buffer(4)]],
                            constant uint& bitOffset [[buffer(5)]],
                            constant uint& count [[buffer(6)]]) {
            threadgroup atomic_uint localOffsets[256];
            if (tid < 256) {
                atomic_store_explicit(&localOffsets[tid], histograms[bid * 256 + tid], memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            if (id < count) {
                uint key = srcKeys[id];
                uint index = srcIndices[id];
                uint bucket = (key >> bitOffset) & 0xFF;
                
                uint slot = atomic_fetch_add_explicit(&localOffsets[bucket], 1, memory_order_relaxed);
                dstKeys[slot] = key;
                dstIndices[slot] = index;
            }
        }

        kernel void clearBuffer(uint id [[thread_position_in_grid]],
                               device uint* buffer [[buffer(0)]],
                               constant uint& count [[buffer(1)]]) {
            if (id < count) buffer[id] = 0;
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
            
            // Grid Pipeline
            let gridVertexFunc = library.makeFunction(name: "gridVertex")!
            let gridFragmentFunc = library.makeFunction(name: "gridFragment")!
            descriptor.vertexFunction = gridVertexFunc
            descriptor.fragmentFunction = gridFragmentFunc
            gridPipelineState = try device.makeRenderPipelineState(descriptor: descriptor)
            
            // --- Compute Pipelines ---
            let calcDepthsFunc = library.makeFunction(name: "calculateDepths")!
            let histogramFunc = library.makeFunction(name: "histogram")!
            let prefixSumFunc = library.makeFunction(name: "prefixSum")!
            let scatterFunc = library.makeFunction(name: "scatter")!
            
            calcDepthsPipelineState = try device.makeComputePipelineState(function: calcDepthsFunc)
            histogramPipelineState = try device.makeComputePipelineState(function: histogramFunc)
            prefixSumPipelineState = try device.makeComputePipelineState(function: prefixSumFunc)
            scatterPipelineState = try device.makeComputePipelineState(function: scatterFunc)
            
            if let clearFunc = library.makeFunction(name: "clearBuffer") {
                clearBufferPipelineState = try device.makeComputePipelineState(function: clearFunc)
            }
            
            let depthDescriptor = MTLDepthStencilDescriptor()
            depthDescriptor.depthCompareFunction = .lessEqual
            depthDescriptor.isDepthWriteEnabled = false // HIGH QUALITY: Disable depth write for Gaussian blending
            depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)
            
            view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        } catch {
            print("Metal Error: \(error)")
        }
    }
    
    // Sorting State (GPU)
    func load(gaussians: GaussianSplatData) {
        if gaussians.id == currentSplatID { return }
        
        // Safety check: Prevent GPU memory overflow
        if gaussians.pointCount > Self.MAX_SPLAT_COUNT {
            print("Metal Error: Splat count (\(gaussians.pointCount)) exceeds safe limit (\(Self.MAX_SPLAT_COUNT))")
            print("Metal Error: This would cause GPU memory overflow and crash the system.")
            print("Metal Error: Please prune the splat dataset before loading.")
            return
        }
        
        var splats: [SplatData] = []

        for i in 0..<gaussians.pointCount {
            splats.append(SplatData(
                position: SIMD4<Float>(gaussians.positions[i].x, gaussians.positions[i].y, gaussians.positions[i].z, 1.0),
                color: SIMD4<Float>(gaussians.colors[i].x, gaussians.colors[i].y, gaussians.colors[i].z, 1.0),
                scale: SIMD4<Float>(gaussians.scales[i].x, gaussians.scales[i].y, gaussians.scales[i].z, gaussians.opacities[i]),
                quaternion: gaussians.rotations[i]
            ))
        }
        
        self.splatCount = splats.count
        let size = splats.count * MemoryLayout<SplatData>.stride
        
        // Explicitly nil out old buffers before allocating new ones to help ARC/Metal
        self.splatBuffer = nil
        self.sortKeysBufferA = nil
        self.sortKeysBufferB = nil
        self.sortIndicesBufferA = nil
        self.sortIndicesBufferB = nil
        self.histogramBuffer = nil
        self.shBuffer = nil
        
        self.splatBuffer = device.makeBuffer(bytes: splats, length: size, options: .storageModeShared)
        
        // --- Initialize GPU Sorting Buffers ---
        let elementCount = splatCount
        let uint32Size = MemoryLayout<UInt32>.stride
        
        self.sortKeysBufferA = device.makeBuffer(length: elementCount * uint32Size, options: .storageModePrivate)
        self.sortKeysBufferB = device.makeBuffer(length: elementCount * uint32Size, options: .storageModePrivate)
        self.sortIndicesBufferA = device.makeBuffer(length: elementCount * uint32Size, options: .storageModePrivate)
        self.sortIndicesBufferB = device.makeBuffer(length: elementCount * uint32Size, options: .storageModePrivate)
        
        // Fill Indices A with [0, 1, 2, ...]
        let initialIndices = (0..<UInt32(elementCount)).map { $0 }
        let tempIndicesBuffer = device.makeBuffer(bytes: initialIndices, length: elementCount * uint32Size, options: .storageModeShared)
        
        // Copy to Private Buffer A
        if let commandBuffer = commandQueue.makeCommandBuffer(),
           let encoder = commandBuffer.makeBlitCommandEncoder() {
            encoder.copy(from: tempIndicesBuffer!, sourceOffset: 0, to: sortIndicesBufferA!, destinationOffset: 0, size: elementCount * uint32Size)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        self.finalSortedIndicesBuffer = sortIndicesBufferA
        
        // Radix Sort needs Histograms and Prefix Sums
        // For 8-bit radix, we have 256 bins.
        // Parallel radix sort often uses tiered histograms (per block).
        // To keep it simple but GPU-efficient, we'll use a 16-bit radix (65536 bins) 
        // or multiple 8-bit passes. Let's do 4 passes of 8 bits.
        // We need 256 * (number of threadgroups) bins.
        let threadgroupCount = (elementCount + 255) / 256
        self.histogramBuffer = device.makeBuffer(length: 256 * threadgroupCount * uint32Size, options: .storageModePrivate)
        
        // SH Buffer
        if !gaussians.shs.isEmpty {
            self.shBuffer = device.makeBuffer(bytes: gaussians.shs, length: gaussians.shs.count * MemoryLayout<Float>.stride, options: .storageModeShared)
        } else {
            self.shBuffer = nil
        }
        
        self.currentSplatID = gaussians.id
        
        print("Metal: Loaded \(splatCount) splats. GPU Sort Buffers allocated. SH Data: \(self.shBuffer != nil)")
    }
    
    // Sorting State
    private var isSorting = false // No longer used for CPU sort, but kept for logic if needed
    // sortQueue is no longer needed as everything is on GPU Command Queue
    
    // Style State
    var exposure: Float = 0
    var gamma: Float = 1.0
    var vignetteStrength: Float = 0.5
    var splatScale: Float = 1.0
    var colorMode: Int = 0 
    var saturation: Float = 1.0
    
    private var finalSortedIndicesBuffer: MTLBuffer?

    /// Dispatches the GPU-based parallel radix sort.
    /// This is the core performance bottleneck, moved to GPU to maintain 60FPS.
    /// 1. Calculate Depths in view space.
    /// 2. Perform 4-pass Radix Sort (8-bits per pass).
    private func dispatchGPUPointsSort(commandBuffer: MTLCommandBuffer, viewMatrix: matrix_float4x4) {
        guard let calcDepths = calcDepthsPipelineState,
              let histPipe = histogramPipelineState,
              let sumPipe = prefixSumPipelineState,
              let scatterPipe = scatterPipelineState,
              let clearPipe = clearBufferPipelineState,
              let splats = splatBuffer,
              let keysA = sortKeysBufferA,
              let keysB = sortKeysBufferB,
              let indicesA = sortIndicesBufferA,
              let indicesB = sortIndicesBufferB,
              let histBuf = histogramBuffer,
              splatCount > 0 else { return }
        
        let threadgroupSize = 256
        let threadgroupCount = (splatCount + threadgroupSize - 1) / threadgroupSize
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "GPU Sort Encoder"
        
        // 1. Calculate Depths
        encoder.setComputePipelineState(calcDepths)
        encoder.setBuffer(splats, offset: 0, index: 0)
        encoder.setBuffer(keysA, offset: 0, index: 1)
        encoder.setBuffer(indicesA, offset: 0, index: 2)
        
        var uniforms = MetalUniforms(
            viewMatrix: viewMatrix,
            projectionMatrix: matrix_float4x4(0),
            invViewMatrix: viewMatrix.inverse,
            invProjectionMatrix: matrix_float4x4(0),
            viewportSize: SIMD4<Float>(0, 0, 0, Float(splatCount)),
            styleParams: SIMD4<Float>(0, 0, 0, 0),
            cameraPosition: SIMD4<Float>(0, 0, 0, 0),
            colorParams: SIMD4<Float>(0, 0, 0, 0),
            affordanceParams: SIMD4<Float>(0, 0, 0, 0),
            headPosition: SIMD4<Float>(0, 0, 0, 0)
        )
        encoder.setBytes(&uniforms, length: MemoryLayout<MetalUniforms>.stride, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: threadgroupCount, height: 1, depth: 1), 
                                  threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        
        encoder.memoryBarrier(scope: .buffers)
        
        // 2. Radix Sort (4 passes of 8 bits)
        var srcKeys = keysA
        var dstKeys = keysB
        var srcIndices = indicesA
        var dstIndices = indicesB
        
        for i in 0..<4 {
            var bitOffset = uint(i * 8)
            var uCount = uint(splatCount)
            var uTGCount = uint(threadgroupCount)
            var hCount = uint(256 * threadgroupCount)
            
            // a. Clear Histogram (Using compute instead of blit to stay in same encoder)
            encoder.setComputePipelineState(clearPipe)
            encoder.setBuffer(histBuf, offset: 0, index: 0)
            encoder.setBytes(&hCount, length: 4, index: 1)
            let hTGCount = (Int(hCount) + 255) / 256
            encoder.dispatchThreadgroups(MTLSize(width: hTGCount, height: 1, depth: 1), 
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            
            // b. Histogram
            encoder.setComputePipelineState(histPipe)
            encoder.setBuffer(srcKeys, offset: 0, index: 0)
            encoder.setBuffer(histBuf, offset: 0, index: 1)
            encoder.setBytes(&bitOffset, length: 4, index: 2)
            encoder.setBytes(&uCount, length: 4, index: 3)
            encoder.dispatchThreadgroups(MTLSize(width: threadgroupCount, height: 1, depth: 1), 
                                      threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            
            encoder.memoryBarrier(scope: .buffers)
            
            // c. Prefix Sum
            encoder.memoryBarrier(scope: .buffers)
            encoder.setComputePipelineState(sumPipe)
            encoder.setBuffer(histBuf, offset: 0, index: 0)
            encoder.setBytes(&uTGCount, length: 4, index: 1)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), 
                                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            
            // d. Scatter
            encoder.memoryBarrier(scope: .buffers)
            encoder.setComputePipelineState(scatterPipe)
            encoder.setBuffer(srcKeys, offset: 0, index: 0)
            encoder.setBuffer(srcIndices, offset: 0, index: 1)
            encoder.setBuffer(dstKeys, offset: 0, index: 2)
            encoder.setBuffer(dstIndices, offset: 0, index: 3)
            encoder.setBuffer(histBuf, offset: 0, index: 4)
            encoder.setBytes(&bitOffset, length: 4, index: 5)
            encoder.setBytes(&uCount, length: 4, index: 6)
            encoder.dispatchThreadgroups(MTLSize(width: threadgroupCount, height: 1, depth: 1), 
                                      threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            
            // Swap ping-pong
            let tempK = srcKeys; srcKeys = dstKeys; dstKeys = tempK
            let tempI = srcIndices; srcIndices = dstIndices; dstIndices = tempI
        }
        
        encoder.endEncoding()
        self.finalSortedIndicesBuffer = srcIndices
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        aspectRatio = Float(size.width / size.height)
        viewportSize = SIMD2<Float>(Float(size.width), Float(size.height))
    }
    
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor,
              let pipeline = pipelineState,
              let buffer = splatBuffer else { return }
        
        let viewMatrix = makeViewMatrix()
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        // 1. GPU Sort
        dispatchGPUPointsSort(commandBuffer: commandBuffer, viewMatrix: viewMatrix)
        
        // 2. Render
        let projMatrix: matrix_float4x4
        if isHolographic {
            // --- Holographic "Window" Projection ---
            // scale = (Visible Frustum Width at Orbit Distance) / (Physical Screen Width)
            // This ensures the object fills the same % of the screen as in standard mode.
            let worldScale = calculateWorldScale()
            
            let viewAspect = viewportSize.x / viewportSize.y
            // Scale physical dimensions into Virtual World Units
            let sw: Float = 30.0 * worldScale 
            let sh: Float = sw / viewAspect
            
            let hx = headPosition.x * worldScale
            let hy = headPosition.y * worldScale
            let hz = headPosition.z * worldScale
            
            // Proximity damping: ensure we don't clip as the user approaches the screen
            let d = hz 
            let near: Float = 0.1
            let far: Float = 1000.0
            
            // Construct the asymmetric portal bounds
            let left   = (-sw/2 - hx) * near / d
            let right  = (sw/2 - hx) * near / d
            let top    = (sh/2 - hy) * near / d   // Y-down alignment
            let bottom = (-sh/2 - hy) * near / d
            
            projMatrix = MathUtils.makeOffAxisFrustumMatrix(left: left, right: right, bottom: bottom, top: top, near: near, far: far)
        } else {
            projMatrix = MathUtils.makePerspectiveMatrix(fovRadians: MathUtils.degreesToRadians(60), aspect: aspectRatio, near: 0.1, far: 1000)
        }

        var uniforms = MetalUniforms(
            viewMatrix: viewMatrix,
            projectionMatrix: projMatrix,
            invViewMatrix: viewMatrix.inverse,
            invProjectionMatrix: projMatrix.inverse,
            viewportSize: SIMD4<Float>(viewportSize.x, viewportSize.y, shBuffer != nil ? 1.0 : 0.0, Float(splatCount)),
            styleParams: SIMD4<Float>(exposure, gamma, vignetteStrength, splatScale),
            cameraPosition: SIMD4<Float>(Float(cameraPosition.x), Float(cameraPosition.y), Float(cameraPosition.z), Float(colorMode)),
            colorParams: SIMD4<Float>(saturation, 0, 0, 0),
            affordanceParams: SIMD4<Float>(Float(orbitTarget.x), Float(orbitTarget.y), Float(orbitTarget.z), isNavigating ? 1.0 : 0.0),
            headPosition: SIMD4<Float>(headPosition.x, headPosition.y, headPosition.z, isHolographic ? 1.0 : 0.0)
        )
        
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        
        encoder.setRenderPipelineState(pipeline)
        if let ds = depthStencilState { encoder.setDepthStencilState(ds) }
        
        // --- GRID PASS (DISABLED - was confusing users) ---
        /*
        if let gridPSO = gridPipelineState {
            encoder.setRenderPipelineState(gridPSO)
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<MetalUniforms>.stride, index: 1)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<MetalUniforms>.stride, index: 1)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.setRenderPipelineState(pipeline)
        }
        */
        
        encoder.setVertexBuffer(buffer, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<MetalUniforms>.stride, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<MetalUniforms>.stride, index: 1)
        
        // Final sorted indices are in finalSortedIndicesBuffer
        if let sortBuf = finalSortedIndicesBuffer {
            encoder.setVertexBuffer(sortBuf, offset: 0, index: 2)
        }
        
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
                DispatchQueue.main.async {
                    self?.lastCapturedTexture = capturedTexture
                }
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
        guard let _ = MTLCreateSystemDefaultDevice() else { return nil } 
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
    
    // MARK: - Camera Math
    
    // Helper to match Holographic Scale to Standard Camera FOV
    private func calculateWorldScale() -> Float {
        // Standard Camera FOV is 60 degrees.
        // We want the Visible Frustum Width at 'orbitDistance' to map to the Physical Screen Width (30cm).
        
        let eye = vector_float3(Float(cameraPosition.x), Float(cameraPosition.y), Float(cameraPosition.z))
        let target: vector_float3
        if let t = cameraPosition.target {
            target = vector_float3(Float(t.x), Float(t.y), Float(t.z))
        } else {
            target = eye + vector_float3(0, 0, -1)
        }
        
        let dist = distance(eye, target)
        
        // Width = 2 * dist * tan(fov/2) * aspect
        // FOV = 60 deg -> tan(30) = 0.577
        let fovRad = MathUtils.degreesToRadians(60)
        let visibleHeight = 2.0 * dist * tan(fovRad / 2.0)
        let visibleWidth = visibleHeight * aspectRatio
        
        // We want visibleWidth (virtual units) to map to 30.0 cm (physical units)
        // So: virtual = physical * scale
        // scale = virtual / physical
        
        let physicalWidthCM: Float = 30.0
        return visibleWidth / physicalWidthCM
    }
    
    private func makeViewMatrix() -> matrix_float4x4 {
        let eye = vector_float3(Float(cameraPosition.x), Float(cameraPosition.y), Float(cameraPosition.z))
        
        let target: vector_float3
        if let t = cameraPosition.target {
            target = vector_float3(Float(t.x), Float(t.y), Float(t.z))
        } else {
            target = eye + vector_float3(0, 0, -1)
        }
        
        let up = vector_float3(0, -1, 0) // ml-sharp uses Y-down
        
        // 1. Base Orbit/Fly Matrix
        let baseView = MathUtils.matrix_look_at_right_hand(eye: eye, target: target, up: up)
        
        if isHolographic {
            // --- Holographic World Scaling ---
            // To achieve 1:1 motion parallax, physical movement (cm) must map
            // to virtual units proportional to the current manual zoom level.
            let distToSplat = distance(eye, target)
            let worldScale = calculateWorldScale()
            
            let hx = headPosition.x * worldScale
            let hy = headPosition.y * worldScale
            let hz = headPosition.z * worldScale 
            
            // We want the 'camera' to be at (hx, hy, hz) relative to the target (0,0,0).
            // But 'baseView' already places 'target' at (0, 0, -distToSplat).
            // So we first shift target to (0,0,0): Translate(0, 0, distToSplat)
            // Then translate to Eye: Translate(-hx, -hy, -hz)
            // Combined: Translate(-hx, -hy, -hz + distToSplat)
            
            let headTranslate = matrix_float4x4(columns: (
                vector_float4(1, 0, 0, 0),
                vector_float4(0, 1, 0, 0),
                vector_float4(0, 0, 1, 0),
                vector_float4(-hx, -hy, -hz + distToSplat, 1)
            ))
            
            return headTranslate * baseView
        }
        
        return baseView
    }
}

