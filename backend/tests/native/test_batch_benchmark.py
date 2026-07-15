#!/usr/bin/env python3
"""Batch-N performance gate.

Runs the worker on the long video with batch=1 and batch=16, then checks that
batch=16 provides a measurable speed-up. Tracker is off.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
WORKER = "/app/backend/native/build/deepstream_face_worker"
VIDEO = "/app/backend/artifacts/videos/Friends.mp4"


def _run(batch_size: int, run_name: str) -> dict:
    out_dir = REPO / "backend" / "out" / run_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    container_out = f"/app/backend/out/{run_name}"
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "device=0",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-e", "GST_PLUGIN_PATH=/app/backend/native/build/gst-plugins",
        "-e", "USE_NEW_NVSTREAMMUX=0",
        "-v", f"{REPO}:/app",
        "-w", "/app",
        "--entrypoint", WORKER,
        CONTAINER,
        VIDEO,
        container_out,
        "0",
        "--batch-size", str(batch_size),
        "--tracker", "off",
    ]
    # Long video may take up to ~120s with batch=1; batch=16 finishes in ~15s.
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"FAIL: worker failed for batch={batch_size}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(1)
    manifest_path = out_dir / "run_manifest.json"
    with manifest_path.open() as f:
        return json.load(f)


def main() -> int:
    b1 = _run(1, "batch_benchmark_b1")
    b16 = _run(16, "batch_benchmark_b16")

    for m, name in [(b1, "batch=1"), (b16, "batch=16")]:
        if not m.get("completed") or m.get("exit_code", -1) != 0:
            print(f"FAIL: {name} did not complete cleanly: {m}", file=sys.stderr)
            return 1

    def fps(m: dict) -> float:
        return m["frames_processed"] / m["wall_time_sec"] if m["wall_time_sec"] else 0.0

    fps1 = fps(b1)
    fps16 = fps(b16)
    speedup = fps16 / fps1 if fps1 else 0.0

    print(f"batch=1:  {fps1:.1f} FPS ({b1['frames_processed']} frames, {b1['wall_time_sec']:.2f}s)")
    print(f"batch=16: {fps16:.1f} FPS ({b16['frames_processed']} frames, {b16['wall_time_sec']:.2f}s)")
    print(f"speedup:  {speedup:.2f}x")

    if speedup < 1.5:
        print("FAIL: batch=16 speedup < 1.5x", file=sys.stderr)
        return 1
    if fps16 < 300.0:
        print("FAIL: batch=16 FPS < 300", file=sys.stderr)
        return 1
    print("Batch benchmark PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
