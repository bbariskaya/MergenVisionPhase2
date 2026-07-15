#!/usr/bin/env python3
"""Detector determinism test.

Runs the native DeepStream RetinaFace worker multiple times on the same short
clip and checks that per-frame detection geometry/score/landmarks are identical
across runs. Raw tracker IDs are intentionally ignored because NvDCF may
assign different internal IDs across independent process lifetimes.
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
# Sprint 03 Foundation Closure requires 20 independent runs for determinism gate.
RUNS = 20


def _run_worker(run_dir_name: str) -> Path:
    out_dir = REPO / "backend" / "out" / run_dir_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    container_out = f"/app/backend/out/{run_dir_name}"
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "device=0",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-e", "GST_PLUGIN_PATH=/app/backend/native/build",
        "-v", f"{REPO}:/app",
        "-w", "/app",
        "--entrypoint", WORKER,
        CONTAINER,
        VIDEO,
        container_out,
        "0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: worker run failed for {run_dir_name}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(1)
    return out_dir


def _normalize_detections(jsonl_path: Path) -> dict[int, list[tuple]]:
    frames: dict[int, list[tuple]] = {}
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            frame_idx = int(rec["frame"])
            normalized = []
            for d in rec["detections"]:
                landmarks = tuple(round(float(d["landmarks"][k]), 3) for k in range(10))
                tup = (
                    round(float(d["x1"]), 3),
                    round(float(d["y1"]), 3),
                    round(float(d["x2"]), 3),
                    round(float(d["y2"]), 3),
                    round(float(d["score"]), 4),
                    landmarks,
                )
                normalized.append(tup)
            normalized.sort()
            frames[frame_idx] = normalized
    return frames


def main() -> int:
    run_dirs: list[Path] = []
    for i in range(RUNS):
        run_dir = _run_worker(f"detector_determinism_run_{i}")
        run_dirs.append(run_dir)

    per_run = [_normalize_detections(rd / "detections.jsonl") for rd in run_dirs]

    # All runs must process the same set of frames.
    frame_sets = [set(run.keys()) for run in per_run]
    if not all(fs == frame_sets[0] for fs in frame_sets):
        print("FAIL: frame sets differ between runs", file=sys.stderr)
        for i, fs in enumerate(frame_sets):
            print(f"  run {i}: {sorted(fs)}", file=sys.stderr)
        return 1

    failures = 0
    for frame_idx in sorted(frame_sets[0]):
        expected = per_run[0][frame_idx]
        for i in range(1, RUNS):
            actual = per_run[i][frame_idx]
            if actual != expected:
                print(
                    f"FAIL: frame {frame_idx} differs between run 0 and run {i}",
                    file=sys.stderr,
                )
                print(f"  expected: {expected}", file=sys.stderr)
                print(f"  actual:   {actual}", file=sys.stderr)
                failures += 1

    if failures:
        print(f"\nDetector determinism FAILED ({failures} frame mismatch(es))", file=sys.stderr)
        return 1

    print(f"Detector determinism PASSED ({len(frame_sets[0])} frames across {RUNS} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
