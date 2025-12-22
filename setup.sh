#!/bin/bash

# SharpGlass Setup Script
# This script sets up the Python environment and dependencies for SharpGlass.

set -e

echo "🚀 Setting up SharpGlass..."

# 1. Check for Python 3.13+
if ! command -v python3.13 &> /dev/null; then
    echo "❌ Python 3.13 not found. Please install it (e.g., brew install python@3.13)."
    exit 1
fi

# 2. Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3.13 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# 3. Activate venv and install dependencies
echo "🛠 Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip

# Check if ml-sharp exists
if [ ! -d "ml-sharp" ]; then
    echo "⚠️  ml-sharp directory not found. Cloning from repository..."
    # Replace with actual URL if public, otherwise instructions
    echo "   Cloning ml-sharp from apple/ml-sharp..."
    git clone https://github.com/apple/ml-sharp.git ml-sharp
fi

if [ -d "ml-sharp" ]; then
    cd ml-sharp
    pip install -e .
    cd ..
else
    echo "❌ ml-sharp is required for 3D generation. Process aborted."
    exit 1
fi

echo "✅ Python environment ready."

# 4. Build Swift Project
echo "🏗 Building SharpGlass..."
swift build

echo "✨ Setup complete! You can now run the app with 'swift run SharpGlassApp'"
