#!/usr/bin/env python3
"""Annotated render parity gate.

Runs the worker with the GPU OSD/render path enabled and checks that the
output MP4 has the same frame count and duration as the source.
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
VIDEO = "/app/backend/artifacts/videos/friendsshort_50f.mp4"
BATCH_SIZE = 4
FFPROBE = shutil.which("ffprobe") or "/usr/bin/ffprobe"


def _run() -> Path:
    run_name = "render_parity_smoke"
    out_dir = REPO / "backend" / "out" / run_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    container_out = f"/app/backend/out/{run_name}"
    annotated = f"{container_out}/annotated.mp4"
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
        "--batch-size", str(BATCH_SIZE),
        "--tracker", "off",
        "--render",
        "--annotated-output", annotated,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print("FAIL: worker render run failed", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(1)
    mp4_path = out_dir / "annotated.mp4"
    if not mp4_path.exists():
        print(f"FAIL: annotated output missing at {mp4_path}", file=sys.stderr)
        raise SystemExit(1)
    return out_dir


def _probe(path: Path) -> dict:
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=nb_read_packets,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: ffprobe failed for {path}: {result.stderr}", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(result.stdout)


def _frame_count(path: Path) -> int | None:
    # Prefer packet count when available; fall back to duration * fps.
    info = _probe(path)
    stream = (info.get("streams") or [{}])[0]
    packets = stream.get("nb_read_packets")
    if packets is not None:
        return int(packets)
    fps_str = stream.get("r_frame_rate", "")
    duration = stream.get("duration")
    if duration is None:
        duration = info.get("format", {}).get("duration")
    if fps_str and duration:
        num, den = fps_str.split("/")
        fps = int(num) / int(den)
        return int(round(float(duration) * fps))
    return None


def main() -> int:
    out_dir = _run()
    source_path = REPO / VIDEO.lstrip("/app/")
    source_frames = _frame_count(source_path)
    render_frames = _frame_count(out_dir / "annotated.mp4")

    if source_frames is None or render_frames is None:
        print("FAIL: could not determine frame count", file=sys.stderr)
        return 1
    if source_frames != render_frames:
        print(
            f"FAIL: render frame count mismatch: source={source_frames} render={render_frames}",
            file=sys.stderr,
        )
        return 1

    source_info = _probe(source_path)
    render_info = _probe(out_dir / "annotated.mp4")
    source_duration = float(
        (source_info.get("streams") or [{}])[0].get("duration")
        or source_info.get("format", {}).get("duration", 0)
    )
    render_duration = float(
        (render_info.get("streams") or [{}])[0].get("duration")
        or render_info.get("format", {}).get("duration", 0)
    )
    if abs(source_duration - render_duration) > 0.1:
        print(
            f"FAIL: render duration mismatch: source={source_duration:.3f}s render={render_duration:.3f}s",
            file=sys.stderr,
        )
        return 1

    print(
        f"Render parity PASSED (frames={render_frames}, duration={render_duration:.3f}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
