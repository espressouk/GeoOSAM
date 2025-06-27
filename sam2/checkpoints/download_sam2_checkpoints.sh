#!/bin/bash
# Improved download_sam2_checkpoints.sh

set -e  # Exit on any error

echo "🚀 GeoOSAM SAM2 Checkpoint Downloader"
echo "====================================="

# Check if we're in the right directory
if [[ ! "$(basename $(pwd))" == "checkpoints" ]]; then
    echo "❌ Please run this script from the checkpoints directory"
    exit 1
fi

# Check if checkpoint already exists
if [[ -f "sam2_hiera_tiny.pt" ]]; then
    echo "✅ Checkpoint already exists: $(ls -lh sam2_hiera_tiny.pt)"
    exit 0
fi

# Detect download tool
if command -v wget &> /dev/null; then
    DL_CMD="wget --progress=bar:force"
elif command -v curl &> /dev/null; then
    DL_CMD="curl -L -O --progress-bar"
else
    echo "❌ Please install wget or curl to download the checkpoint."
    exit 1
fi

echo "📥 Downloading SAM2 tiny checkpoint (~38MB)..."
echo "🌐 URL: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"

$DL_CMD "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"

# Verify download
if [[ -f "sam2_hiera_tiny.pt" ]] && [[ $(stat -f%z "sam2_hiera_tiny.pt" 2>/dev/null || stat -c%s "sam2_hiera_tiny.pt") -gt 1000000 ]]; then
    echo "✅ Download successful: $(ls -lh sam2_hiera_tiny.pt)"
else
    echo "❌ Download failed or file too small"
    exit 1
fi
