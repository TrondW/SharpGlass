import AVFoundation
import CoreImage
import AppKit

final class VideoExporter: @unchecked Sendable {
    private var assetWriter: AVAssetWriter?
    private var assetWriterInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private let size: CGSize
    private let queue = DispatchQueue(label: "com.sharpglass.videoexport")
    
    init(url: URL, size: CGSize) throws {
        self.size = size
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }
    
    func start(url: URL) throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: size.width,
            AVVideoHeightKey: size.height
        ]
        
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
                kCVPixelBufferWidthKey as String: size.width,
                kCVPixelBufferHeightKey as String: size.height
            ]
        )
        
        if writer.canAdd(input) {
            writer.add(input)
        }
        
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        
        self.assetWriter = writer
        self.assetWriterInput = input
        self.pixelBufferAdaptor = adaptor
    }
    
    func append(pixelBuffer: CVPixelBuffer, at time: Double) {
        queue.sync {
            guard let input = self.assetWriterInput,
                  let adaptor = self.pixelBufferAdaptor,
                  input.isReadyForMoreMediaData else { return }
            
            let cmTime = CMTime(seconds: time, preferredTimescale: 600)
            adaptor.append(pixelBuffer, withPresentationTime: cmTime)
        }
    }
    
    func finish() async {
        return await withCheckedContinuation { continuation in
            queue.async {
                guard let writer = self.assetWriter else {
                    continuation.resume()
                    return
                }
                
                self.assetWriterInput?.markAsFinished()
                writer.finishWriting {
                    continuation.resume()
                }
            }
        }
    }
}
