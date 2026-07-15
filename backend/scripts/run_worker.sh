#!/bin/bash
set -euo pipefail

IMAGE="${MV_CONTAINER_IMAGE:-nvcr.io/nvidia/deepstream:9.0-triton-multiarch}"

# Resolve monorepo root from backend/scripts/run_worker.sh
HOST_WORKSPACE="$(cd "$(dirname "$0")/../.." && pwd)"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <container-input-path> <container-output-dir> [host_gpu_id]"
    exit 1
fi

INPUT_PATH="$1"
OUTPUT_DIR="$2"
HOST_GPU_ID="${3:-0}"

mkdir -p "$HOST_WORKSPACE/$OUTPUT_DIR"

docker run --rm \
    --gpus "device=$HOST_GPU_ID" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e GST_PLUGIN_PATH="/app/backend/native/build" \
    -e MV_CONTAINER_IMAGE="$IMAGE" \
    -v "$HOST_WORKSPACE:/app" \
    -w /app \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -c "set -o pipefail; timeout 900 /app/backend/native/build/deepstream_face_worker '$INPUT_PATH' '$OUTPUT_DIR' 0 2>&1 | tee /app/$OUTPUT_DIR/run.log"

echo "run_worker_exit=$?"
