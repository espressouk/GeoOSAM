#!/usr/bin/env bash
# download_sam2.1_checkpoint.sh
# --------------------------------------------
# Downloads the SAM 2.1 *tiny* checkpoint once,
# verifies size, and exits cleanly.
# --------------------------------------------

set -e  # Exit immediately on any error

echo "🚀 GeoOSAM · SAM 2.1 Checkpoint Downloader"
echo "=========================================="

# Ensure we’re inside the checkpoints directory
if [[ ! "$(basename "$(pwd)")" == "checkpoints" ]]; then
  echo "❌ Please run this script from your plugin’s checkpoints directory."
  echo "   (current dir: $(pwd))"
  exit 1
fi

CKPT_FILE="sam2.1_hiera_tiny.pt"
CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/${CKPT_FILE}"

# If the checkpoint is already present, skip download
if [[ -f "${CKPT_FILE}" ]]; then
  echo "✅ Checkpoint already exists: $(ls -lh "${CKPT_FILE}")"
  exit 0
fi

# Detect an available downloader
if command -v wget &> /dev/null; then
  DL_CMD="wget --progress=bar:force -O ${CKPT_FILE}"
elif command -v curl &> /dev/null; then
  DL_CMD="curl -L -o ${CKPT_FILE} --progress-bar"
else
  echo "❌ Neither wget nor curl found. Please install one of them."
  exit 1
fi

echo "📥 Downloading SAM 2.1 tiny checkpoint (~160 MB)…"
echo "🌐 ${CKPT_URL}"
eval "${DL_CMD} \"${CKPT_URL}\""

# Simple size check (>1 MB) to catch truncated downloads
FILE_SIZE=$(stat -c%s "${CKPT_FILE}" 2>/dev/null || stat -f%z "${CKPT_FILE}")
if [[ "${FILE_SIZE}" -gt 1000000 ]]; then
  echo "✅ Download successful: $(ls -lh "${CKPT_FILE}")"
else
  echo "❌ Download failed or file too small. Remove the file and retry."
  exit 1
fi
