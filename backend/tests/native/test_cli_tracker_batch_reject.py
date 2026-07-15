#!/usr/bin/env python3
"""CLI gate: tracker + batch-size > 1 must be rejected before pipeline starts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
WORKER = "/app/backend/native/build/deepstream_face_worker"
VIDEO = "/app/backend/artifacts/videos/friendsshort_50f.mp4"


def main() -> int:
    run_name = "cli_tracker_batch_reject"
    out_dir = REPO / "backend" / "out" / run_name
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    container_out = f"/app/backend/out/{run_name}"
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "device=0",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-e", "GST_PLUGIN_PATH=/app/backend/native/build/gst-plugins",
        "-v", f"{REPO}:/app",
        "-w", "/app",
        "--entrypoint", WORKER,
        CONTAINER,
        VIDEO,
        container_out,
        "0",
        "--batch-size", "8",
        "--tracker", "/app/backend/native/config/nvdcf.yml",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout + result.stderr
    if result.returncode == 0:
        print("FAIL: worker accepted tracker + batch-size 8", file=sys.stderr)
        return 1
    if "NvMOT contract violation" not in out:
        print("FAIL: expected NvMOT contract violation error, got:\n" + out, file=sys.stderr)
        return 1
    print("CLI tracker+batch reject PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
