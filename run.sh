#!/usr/bin/env bash
set -e

echo "============================================================"
echo "   EarlyBird Semantic Architecture Workbench (Linux/CachyOS)"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Qdrant container is up if docker compose exists
if command -v docker &> /dev/null; then
    echo "Ensuring Qdrant vector database is running..."
    docker compose up -d > /dev/null 2>&1 || true
fi

cd earlybird-arch

# If running on AMD GPU (ROCm) with CachyOS, enable ROCm / Vulkan acceleration
if [ -d "/opt/rocm" ]; then
    echo "Detected ROCm installation at /opt/rocm - AMD Radeon RX acceleration enabled."
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
fi

echo "Starting Workbench on http://localhost:3030 ..."
cargo run --release -- serve --port 3030
