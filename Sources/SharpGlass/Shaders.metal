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
    float2 viewportSize;
    float exposure;
    float gamma;
    float vignetteStrength;
    float _pad1;
    float _pad2;
    float _pad3;
    float3 cameraPosition;
    float hasSH;
    float3 _padEnd;
};

struct SplatData {
    float3 position;
    float3 color;
    float opacity;
    float3 scale;
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
    
    float4 p_view = uniforms.viewMatrix * float4(splat.position, 1.0);
    // if (p_view.z > -0.2) { out.position = 0; return out; } 
    // Simplified near clipping to avoid disappearing objects if origin is bad

    float3 finalColor = splat.color;
    
    // View Direction (Object Center to Camera)
    float3 dir = normalize(splat.position - uniforms.cameraPosition);
    
    if (uniforms.hasSH > 0.5f) {
         finalColor += computeSH(dir, shs, sortedIndex);
    }
    
    out.color = float4(max(0.0f, min(1.0f, finalColor)), splat.opacity);

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
    float2 ndc_offset = offset * (2.0f / uniforms.viewportSize);
    
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
    color *= pow(2.0f, uniforms.exposure);
    float dist = length(d);
    float vignette = 1.0f - smoothstep(0.5f, 1.5f, dist) * uniforms.vignetteStrength;
    color *= vignette;
    color = aces_tonemap(color);
    color = pow(color, float3(1.0f / uniforms.gamma));
    
    return float4(color, alpha);
}
