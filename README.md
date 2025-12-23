# SharpGlass

SharpGlass is a macOS application for generating and rendering 3D Gaussian Splats from single images using Apple's `ml-sharp`. It allows you to create high-quality 3D views from everyday photos, with built-in support for iPhone spatial photos.

## Features

- **3D Generation**: Convert single JPEG/PNG images into 3D Gaussian Splats.
- **Spatial Photo Support**: Extract depth and stereo pairs from HEIC spatial photos for enhanced reconstruction.
- **Real-time Rendering**: interactive 3D navigation using a high-performance Metal-based splatting renderer.
- **Optimized Resource Management**: Automatically prunes large datasets and manages memory efficiently for smooth performance.
- **Smart Onboarding**: Automatically handles backend setup (Python venv, dependencies) for end-users on first launch.
- **Modern UI**: Full-bleed design with glassmorphic overlays and intuitive drag-and-drop file support.

## Prerequisites

- **macOS 15.0+**
- **High Performance GPU** (M-Series Pro/Max/Ultra recommended for large datasets)
- **Python 3.13+** (Required for `ml-sharp`)
- **Xcode 16.0+** (For building from source)

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TrondW/SharpGlass.git
   cd SharpGlass
   ```

2. **Run the setup script**:
   This script creates a Python virtual environment, installs `ml-sharp` and its dependencies, and builds the Swift project.
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Run the application**:
   ```bash
   swift run SharpGlassApp
   ```

## Installation & Setup Details

### Python Environment
SharpGlass uses a Python-based backend (`ml-sharp`) for 3D inference.

- **For Users**: The app features **Smart Onboarding** which automatically installs the backend environment into `~/Library/Application Support/` on the first run. No manual terminal commands required.
- **For Developers**: The `setup.sh` script prepares your local development environment.

## Building for Distribution
To create a standalone `.app` bundle:
```bash
./build_distribution.sh
```
See [DISTRIBUTION.md](DISTRIBUTION.md) for full details on signing, notarization, and architecture.

### ML Models
On the first run, the application will download the necessary model weights (approx. 1.2GB) required for 3D generation.

## Usage

- **Import**: Drag and drop an image or use the "Open" menu.
- **Navigation**:
  - **Orbit**: Left-click + Drag
  - **Pan**: Right-click + Drag (or Shift + Drag)
  - **Zoom**: Scroll or Option + Click + Drag
  - **Focus**: Press 'F' to center the view on the splat.
- **Export**: Use the "Export Video" button to generate a 3D parallax animation.

## Contributing

Contributions are welcome! Please feel free to submit Pull Requests or open issues.

## License

[MIT License](LICENSE)
