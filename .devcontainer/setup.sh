#!/bin/bash

echo "🧹 Cleaning up previous build artifacts..."
# Remove any existing egg-info directories
find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true

echo "📦 Installing Flare FL in editable mode..."
pip install -e .

echo "🔧 Installing development dependencies..."
pip install -r requirements-dev.txt

echo "✅ Setup complete!"
