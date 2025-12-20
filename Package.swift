// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SharpGlass",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .executable(name: "SharpGlass", targets: ["SharpGlass"])
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "SharpGlass",
            dependencies: [],
            path: "Sources/SharpGlass",
            resources: [
                .process("Shaders.metal")
            ]
        ),
        .testTarget(
            name: "SharpGlassTests",
            dependencies: ["SharpGlass"],
            path: "Tests/SharpGlassTests"
        )
    ]
)
