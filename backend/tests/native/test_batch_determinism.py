#!/usr/bin/env python3
"""Batch-N determinism gate.

Runs the native worker multiple times with a fixed batch size > 1 and verifies
that detections are identical frame-by-frame. Tracker is off because tracker
IDs are not deterministic across process lifetimes.
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
BATCH_SIZE = 8
RUNS = 5


def _run(run_dir_name: str) -> Path:
    out_dir = REPO / "backend" / "out" / run_dir_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    container_out = f"/app/backend/out/{run_dir_name}"
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
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"FAIL: worker failed for {run_dir_name}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(1)
    return out_dir


def _normalize(jsonl_path: Path) -> dict[int, list[tuple]]:
    frames: dict[int, list[tuple]] = {}
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            dets = []
            for d in rec["detections"]:
                tup = (
                    round(float(d["x1"]), 3),
                    round(float(d["y1"]), 3),
                    round(float(d["x2"]), 3),
                    round(float(d["y2"]), 3),
                    round(float(d["score"]), 4),
                    tuple(round(float(d["landmarks"][k]), 3) for k in range(10)),
                )
                dets.append(tup)
            dets.sort()
            frames[int(rec["frame"])] = dets
    return frames


def main() -> int:
    per_run = [_normalize(_run(f"batch_determinism_run_{i}") / "detections.jsonl") for i in range(RUNS)]

    frame_sets = [set(run.keys()) for run in per_run]
    if not all(fs == frame_sets[0] for fs in frame_sets):
        print("FAIL: frame sets differ between runs", file=sys.stderr)
        return 1

    failures = 0
    for frame_idx in sorted(frame_sets[0]):
        expected = per_run[0][frame_idx]
        for i in range(1, RUNS):
            if per_run[i][frame_idx] != expected:
                print(
                    f"FAIL: frame {frame_idx} differs between run 0 and run {i}",
                    file=sys.stderr,
                )
                failures += 1
                break

    if failures:
        print(f"\nBatch determinism FAILED ({failures} mismatch(es))", file=sys.stderr)
        return 1
    print(f"Batch determinism PASSED ({len(frame_sets[0])} frames across {RUNS} runs, batch={BATCH_SIZE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
