#!/usr/bin/env python3
"""Batch detection parity gate.

Runs the native worker with batch=1 and batch=N on the same short clip and
verifies that per-frame detections are equivalent. The tracker must be off
because batch>1 with tracker is rejected by the NvMOT contract.
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
BATCH_SIZES = [1, 2, 8]
IOU_THRESHOLD = 0.95
SCORE_TOLERANCE = 0.01
LANDMARK_TOLERANCE = 1.0  # pixels


def _iou(a: dict, b: dict) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _run(batch_size: int, run_dir_name: str) -> Path:
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
        "--batch-size", str(batch_size),
        "--tracker", "off",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"FAIL: worker failed for batch={batch_size}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(1)
    return out_dir


def _load_frames(jsonl_path: Path) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = {}
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            detections = sorted(rec["detections"], key=lambda d: (d["x1"], d["y1"], d["score"]))
            frames[int(rec["frame"])] = detections
    return frames


def _match_detections(expected: list[dict], actual: list[dict]) -> list[str]:
    errors: list[str] = []
    if len(expected) != len(actual):
        errors.append(f"detection count differs: {len(expected)} vs {len(actual)}")
        return errors
    for exp, act in zip(expected, actual):
        iou = _iou(exp, act)
        if iou < IOU_THRESHOLD:
            errors.append(f"IoU {iou:.3f} < {IOU_THRESHOLD}: {exp} vs {act}")
        if abs(exp["score"] - act["score"]) > SCORE_TOLERANCE:
            errors.append(f"score diff {abs(exp['score'] - act['score']):.4f} > {SCORE_TOLERANCE}")
        for k in range(10):
            if abs(exp["landmarks"][k] - act["landmarks"][k]) > LANDMARK_TOLERANCE:
                errors.append(f"landmark[{k}] diff too large")
                break
    return errors


def main() -> int:
    results: dict[int, dict[int, list[dict]]] = {}
    for bs in BATCH_SIZES:
        out_dir = _run(bs, f"batch_parity_b{bs}")
        results[bs] = _load_frames(out_dir / "detections.jsonl")

    base = results[1]
    failures = 0
    for bs in BATCH_SIZES[1:]:
        other = results[bs]
        if set(base.keys()) != set(other.keys()):
            print(f"FAIL: frame sets differ between batch=1 and batch={bs}", file=sys.stderr)
            failures += 1
            continue
        for frame_idx in sorted(base.keys()):
            errs = _match_detections(base[frame_idx], other[frame_idx])
            if errs:
                print(f"FAIL batch={bs} frame={frame_idx}:", file=sys.stderr)
                for e in errs[:3]:
                    print(f"  {e}", file=sys.stderr)
                failures += 1

    if failures:
        print(f"\nBatch detection parity FAILED ({failures} mismatch(es))", file=sys.stderr)
        return 1
    print(f"Batch detection parity PASSED ({len(base)} frames, batches {BATCH_SIZES})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
