#!/usr/bin/env python3
"""Compare tracker output for batch=1 vs batch>1 to validate temporal correctness.

Does not judge identity accuracy (no ground truth); checks structural invariants:
- same frame count
- same detection count per frame
- no duplicate track IDs within a single frame
- similar number of unique track IDs
- track lifespan ranges
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/user/Workspace/MergenVisionPhase2")
WORKER = "/app/backend/native/build/deepstream_face_worker"
CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
GST_PLUGIN_PATH = "/app/backend/native/build/gst-plugins"
VIDEO = REPO / "backend" / "artifacts" / "videos" / "Friends.mp4"
GPU_ID = 0


def host_to_container(path: Path) -> str:
    return str(Path("/app") / path.relative_to(REPO))


def run_worker(batch_size: int, run_name: str) -> Path:
    out_dir = REPO / "backend" / "out" / run_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    envs = [
        "-e", f"CUDA_VISIBLE_DEVICES={GPU_ID}",
        "-e", f"GST_PLUGIN_PATH={GST_PLUGIN_PATH}",
        "-e", "USE_NEW_NVSTREAMMUX=0",
    ]
    if batch_size > 1:
        envs += ["-e", "MV_ALLOW_TRACKER_BATCH=1"]
    cmd = [
        "docker", "run", "--rm", "--gpus", f"device={GPU_ID}",
        *envs,
        "-v", f"{REPO}:/app",
        "-w", "/app",
        "--entrypoint", WORKER,
        CONTAINER,
        host_to_container(VIDEO),
        host_to_container(out_dir),
        str(GPU_ID),
        "--batch-size", str(batch_size),
        "--tracker", "/app/backend/native/configs/tracker_NvDCF_mergen.yml",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"FAIL: batch={batch_size} worker failed", file=sys.stderr)
        print(result.stdout[-2000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(1)
    return out_dir


def load_detections(jsonl_path: Path) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = {}
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            frames[int(rec["frame"])] = rec["detections"]
    return frames


def analyze(frames: dict[int, list[dict]], label: str) -> dict:
    track_frames: dict[int, set[int]] = defaultdict(set)
    duplicate_frames = 0
    total_detections = 0
    for frame_idx, dets in frames.items():
        total_detections += len(dets)
        seen_ids = set()
        for d in dets:
            tid = d["track_id"]
            if tid in seen_ids:
                duplicate_frames += 1
                break
            seen_ids.add(tid)
            track_frames[tid].add(frame_idx)
    if not track_frames:
        print(f"  {label}: no detections", file=sys.stderr)
        return {}
    lifespans = [max(v) - min(v) + 1 for v in track_frames.values()]
    return {
        "unique_tracks": len(track_frames),
        "total_detections": total_detections,
        "duplicate_id_frames": duplicate_frames,
        "min_lifespan": min(lifespans),
        "max_lifespan": max(lifespans),
        "median_lifespan": sorted(lifespans)[len(lifespans) // 2],
    }


def main() -> int:
    print("Running batch=1 tracker-on baseline...", flush=True)
    b1_dir = run_worker(1, "tracker_correctness_b1")
    print("Running batch=8 tracker-on (override)...", flush=True)
    b8_dir = run_worker(8, "tracker_correctness_b8")

    b1_frames = load_detections(b1_dir / "detections.jsonl")
    b8_frames = load_detections(b8_dir / "detections.jsonl")

    if set(b1_frames.keys()) != set(b8_frames.keys()):
        print("FAIL: frame sets differ", file=sys.stderr)
        return 1

    det_count_mismatch = 0
    for frame_idx in sorted(b1_frames.keys()):
        if len(b1_frames[frame_idx]) != len(b8_frames[frame_idx]):
            det_count_mismatch += 1
    if det_count_mismatch:
        print(f"FAIL: detection count mismatch in {det_count_mismatch} frames", file=sys.stderr)
        return 1

    b1_stats = analyze(b1_frames, "batch=1")
    b8_stats = analyze(b8_frames, "batch=8")

    print("\n=== Tracker batch correctness comparison ===")
    print(f"Frames: {len(b1_frames)}")
    print(f"{'metric':<25} {'batch=1':>12} {'batch=8':>12}")
    for k, v1 in b1_stats.items():
        print(f"{k:<25} {v1:>12} {b8_stats.get(k, 'n/a'):>12}")

    if b8_stats.get("duplicate_id_frames", 0) > 0:
        print("FAIL: batch=8 has duplicate track IDs within a frame", file=sys.stderr)
        return 1

    # Tolerance: batch=8 should not produce wildly more/fewer tracks.
    ratio = b8_stats["unique_tracks"] / b1_stats["unique_tracks"] if b1_stats["unique_tracks"] else 1.0
    print(f"\nunique_tracks ratio (b8/b1): {ratio:.2f}")
    if not (0.5 <= ratio <= 2.0):
        print("FAIL: unique track count differs >2x between batch=1 and batch=8", file=sys.stderr)
        return 1

    print("\nTracker batch correctness structural check PASSED")
    print("Note: this does not prove identity accuracy; only structural consistency.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
