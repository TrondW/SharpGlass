import SwiftUI
import MetalKit

struct MetalSplatView: NSViewRepresentable {
    let gaussians: GaussianSplatData?
    let camera: CameraPosition
    @ObservedObject var viewModel: SharpViewModel // Add ViewModel reference
    
    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()
        
        mtkView.layer?.isOpaque = false
        
        // Initialize renderer
        let renderer = MetalSplatRenderer(metalKitView: mtkView)
        context.coordinator.renderer = renderer
        
        // Inject renderer into ViewModel asynchronously ensures UI loop is ready
        DispatchQueue.main.async {
            self.viewModel.renderer = renderer
        }
        
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        guard let renderer = context.coordinator.renderer else { return }
        
        // Update camera
        renderer.cameraPosition = camera
        
        // Update data if present (renderer handles deduplication via ID)
        if let g = gaussians {
            renderer.load(gaussians: g)
        }
        
        // Update Style
        renderer.exposure = Float(viewModel.exposure)
        renderer.gamma = Float(viewModel.gamma)
        renderer.vignetteStrength = Float(viewModel.vignetteStrength)
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var renderer: MetalSplatRenderer?
    }
}
