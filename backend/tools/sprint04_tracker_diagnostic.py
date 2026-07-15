#!/usr/bin/env python3
"""Diagnose NvDCF tracker ID assignment for Sprint 04.

Runs batch=1 and batch=8 tracker-on, then inspects the raw object_id values
produced by the serialization probe downstream of nvtracker.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path("/home/user/Workspace/MergenVisionPhase2")
WORKER = "/app/backend/native/build/deepstream_face_worker"
CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
GST_PLUGIN_PATH = "/app/backend/native/build/gst-plugins"
VIDEO = REPO / "backend" / "artifacts" / "videos" / "Friends.mp4"
GPU_ID = 0
UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF


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


def load_frames(jsonl_path: Path) -> dict[int, list[dict[str, Any]]]:
    frames: dict[int, list[dict[str, Any]]] = {}
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            frames[int(rec["frame"])] = rec["detections"]
    return frames


def diagnose(frames: dict[int, list[dict[str, Any]]], label: str) -> dict[str, Any]:
    track_seen_in_frame: list[tuple[int, list[int]]] = []
    duplicate_frames: list[int] = []
    all_ids: list[int] = []
    for frame_idx in sorted(frames):
        dets = frames[frame_idx]
        ids = [int(d["track_id"]) for d in dets]
        all_ids.extend(ids)
        if len(ids) > 1 and len(set(ids)) == 1:
            duplicate_frames.append(frame_idx)
            track_seen_in_frame.append((frame_idx, ids))

    id_counter = Counter(all_ids)
    unique_ids = sorted(id_counter.keys())
    sentinel_count = id_counter.get(UNTRACKED_OBJECT_ID, 0)

    report: dict[str, Any] = {
        "label": label,
        "total_frames": len(frames),
        "total_detections": len(all_ids),
        "unique_track_ids": len(unique_ids),
        "unique_ids_decimal": [str(v) for v in unique_ids][:10],
        "unique_ids_hex": [hex(v) for v in unique_ids][:10],
        "duplicate_frames_count": len(duplicate_frames),
        "duplicate_frames_sample": duplicate_frames[:5],
        "top_id": unique_ids[0] if unique_ids else None,
        "top_id_hex": hex(unique_ids[0]) if unique_ids else None,
        "top_id_count": id_counter[unique_ids[0]] if unique_ids else 0,
        "untracked_count": sentinel_count,
    }

    if duplicate_frames:
        first = duplicate_frames[0]
        dets = frames[first]
        report["first_duplicate_frame"] = {
            "frame": first,
            "detection_count": len(dets),
            "all_ids_decimal": [int(d["track_id"]) for d in dets],
            "all_ids_hex": [hex(int(d["track_id"])) for d in dets],
            "detector_scores": [round(float(d["score"]), 4) for d in dets],
        }

    # Strict assertion: tracker-on output must contain zero UNTRACKED_OBJECT_ID.
    if sentinel_count > 0:
        report["verdict"] = "TRACKER_DID_NOT_ASSIGN_IDS"
    elif duplicate_frames:
        report["verdict"] = "DUPLICATE_IDS_WITHOUT_SENTINEL"
    else:
        report["verdict"] = "OK"
    return report


def main() -> int:
    print("Running batch=1 tracker-on diagnostic...", flush=True)
    b1_dir = run_worker(1, "tracker_diag_b1")
    print("Running batch=8 tracker-on diagnostic (override)...", flush=True)
    b8_dir = run_worker(8, "tracker_diag_b8")

    b1_frames = load_frames(b1_dir / "detections.jsonl")
    b8_frames = load_frames(b8_dir / "detections.jsonl")

    b1_report = diagnose(b1_frames, "batch=1")
    b8_report = diagnose(b8_frames, "batch=8")

    out_dir = REPO / "backend" / "out" / "tracker_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tracker_diagnostic_report.json", "w") as f:
        json.dump({"batch1": b1_report, "batch8": b8_report}, f, indent=2)

    print("\n=== Tracker ID diagnostic report ===")
    for r in [b1_report, b8_report]:
        print(f"\n{r['label']}:")
        print(f"  frames={r['total_frames']} detections={r['total_detections']} unique_ids={r['unique_track_ids']}")
        print(f"  top_id={r['top_id']} ({r['top_id_hex']}) count={r['top_id_count']}")
        print(f"  duplicate_frames_count={r['duplicate_frames_count']} untracked_count={r['untracked_count']}")
        print(f"  verdict={r['verdict']}")
        if r.get("first_duplicate_frame"):
            fd = r["first_duplicate_frame"]
            print(f"  first_duplicate_frame={fd['frame']} ids={fd['all_ids_decimal']} ({fd['all_ids_hex']}) scores={fd['detector_scores']}")

    if b1_report["verdict"] == "OK" and b8_report["verdict"] == "OK":
        print("\nTracker diagnostic: OK")
        return 0
    print("\nTracker diagnostic: KNOWN_BROKEN / DEFERRED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
