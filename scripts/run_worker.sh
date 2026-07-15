#!/bin/bash
set -euo pipefail

IMAGE="${MV_CONTAINER_IMAGE:-nvcr.io/nvidia/deepstream:9.0-triton-multiarch}"
DIGEST="${MV_DEEPSTREAM_IMAGE_DIGEST:-sha256:2e45070ad134b9ab2caa4a97ba4d52fa8744a4f0db30900bd92828d51425a69a}"
HOST_WORKSPACE="/home/user/Workspace/MergenVisionPhase2"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <container-input-path> <container-output-dir> [gpu_id]"
    exit 1
fi

INPUT_PATH="$1"
OUTPUT_DIR="$2"
GPU_ID="${3:-0}"

mkdir -p "$HOST_WORKSPACE/$OUTPUT_DIR"

docker run --rm \
    --gpus "device=$GPU_ID" \
    -e CUDA_VISIBLE_DEVICES="$GPU_ID" \
    -e GST_PLUGIN_PATH="/app/native/build" \
    -e MV_CONTAINER_IMAGE="$IMAGE" \
    -e MV_DEEPSTREAM_IMAGE_DIGEST="$DIGEST" \
    -v "$HOST_WORKSPACE:/app" \
    -w /app \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -c "set -o pipefail; timeout 900 /app/native/build/deepstream_face_worker '$INPUT_PATH' '$OUTPUT_DIR' '$GPU_ID' 2>&1 | tee /app/$OUTPUT_DIR/run.log"

echo "run_worker_exit=$?"
