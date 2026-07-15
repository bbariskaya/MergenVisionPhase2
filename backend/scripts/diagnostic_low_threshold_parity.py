#!/usr/bin/env python3
"""Diagnostic: run native pipeline with a lowered confidence threshold.

Production threshold stays at 0.5; this tool is only for root-cause evidence
around threshold-boundary detections (e.g. frame 41 and 42).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend" / "tests" / "fixtures" / "cpu_oracle"))
from cpu_oracle_retinaface import detect_image

CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
WORKER = "/app/backend/native/build/deepstream_face_worker"
VIDEO = "/app/backend/artifacts/videos/friendsshort_50f.mp4"


def _run_native(threshold: float) -> Path:
    out_dir = REPO / "backend" / "out" / f"diag_parity_thresh_{threshold}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    container_out = f"/app/backend/out/diag_parity_thresh_{threshold}"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "device=0",
        "-e",
        "CUDA_VISIBLE_DEVICES=0",
        "-e",
        f"MV_DIAG_CONF_THRESHOLD={threshold}",
        "-e",
        "GST_PLUGIN_PATH=/app/backend/native/build/gst-plugins",
        "-v",
        f"{REPO}:/app",
        "-w",
        "/app",
        "--entrypoint",
        WORKER,
        CONTAINER,
        VIDEO,
        container_out,
        "0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"native worker failed at threshold {threshold}")
    return out_dir


def _native_dets_for_frame(jsonl: Path, frame_idx: int) -> list[dict]:
    with jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            if int(rec["frame"]) == frame_idx:
                return rec.get("detections", [])
    return []


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(
        0.0, b[2] - b[0]
    ) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    out_dir = _run_native(0.45)
    jsonl = out_dir / "detections.jsonl"
    video = REPO / "backend" / "artifacts" / "videos" / "friendsshort_50f.mp4"
    cap = cv2.VideoCapture(str(video))
    for frame_idx in (41, 42):
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cpu_boxes, cpu_scores, cpu_landmarks = detect_image(frame)
        native_dets = _native_dets_for_frame(jsonl, frame_idx)
        print(f"\n=== frame {frame_idx} native threshold=0.45 ===")
        print(f"CPU detections: {len(cpu_boxes)}; native detections: {len(native_dets)}")
        for i, (b, s, l) in enumerate(zip(cpu_boxes, cpu_scores, cpu_landmarks)):
            best_iou = 0.0
            best_nd = None
            best_j = -1
            for j, nd in enumerate(native_dets):
                nb = np.array([nd["x1"], nd["y1"], nd["x2"], nd["y2"]])
                iou = _iou(b, nb)
                if iou > best_iou:
                    best_iou = iou
                    best_nd = nd
                    best_j = j
            print(
                f"  CPU[{i}] score={s:.4f} box={b.tolist()} -> "
                f"best native[{best_j}] iou={best_iou:.4f} "
                f"native_score={best_nd['score'] if best_nd else None}"
            )
            if best_nd:
                nat_l = np.array(best_nd["landmarks"], dtype=np.float32).reshape(5, 2)
                lm_err = float(np.max(np.linalg.norm(l - nat_l, axis=1)))
                print(f"    landmark max err={lm_err:.2f}px")
    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
