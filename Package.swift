// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SharpGlass",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .executable(name: "SharpGlass", targets: ["SharpGlassApp"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "SharpGlassLibrary",
            dependencies: [],
            path: "Sources/SharpGlass",
            exclude: ["Shaders.metal"]
        ),
        .executableTarget(
            name: "SharpGlassApp",
            dependencies: ["SharpGlassLibrary"],
            path: "Sources/Main",
            exclude: ["Info.plist"],
            resources: [
                .process("Assets.xcassets")
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Sources/Main/Info.plist"
                ])
            ]
        ),
        .testTarget(
            name: "SharpGlassTests",
            dependencies: ["SharpGlassLibrary"],
            path: "Tests/SharpGlassTests"
        )
    ]
)
