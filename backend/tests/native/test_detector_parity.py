#!/usr/bin/env python3
"""Detector parity test: native GPU worker vs CPU ONNX Runtime oracle."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "backend" / "tests" / "fixtures" / "cpu_oracle"))

try:
    from cpu_oracle_retinaface import detect_image
except ImportError as exc:
    print(f"FATAL: cannot import CPU oracle: {exc}", file=sys.stderr)
    sys.exit(1)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_native_frame_detections(jsonl_path: Path, frame_idx: int) -> list[dict]:
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if int(rec["frame"]) == frame_idx:
                return rec["detections"]
    return []


def test_frame_parity(video_path: Path, jsonl_path: Path, frame_idx: int = 0) -> None:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    assert ok, "failed to read first frame from video"
    for _ in range(frame_idx):
        ok, frame = cap.read()
        assert ok, f"failed to read frame {frame_idx}"
    cap.release()

    cpu_boxes, cpu_scores, cpu_landmarks = detect_image(frame)
    native_dets = _load_native_frame_detections(jsonl_path, frame_idx)

    assert len(cpu_boxes) > 0, "CPU oracle found no faces in test frame"
    print(f"CPU detections: {len(cpu_boxes)}, native detections: {len(native_dets)}")

    # Match CPU oracle detections to native detections by IoU.
    matched = 0
    for cpu_box, cpu_score in zip(cpu_boxes, cpu_scores):
        best_iou = 0.0
        best_native = None
        for nd in native_dets:
            nb = np.array([nd["x1"], nd["y1"], nd["x2"], nd["y2"]], dtype=np.float32)
            iou = _iou(cpu_box, nb)
            if iou > best_iou:
                best_iou = iou
                best_native = nd

        assert best_iou >= 0.80, (
            f"CPU box {cpu_box} best IoU to native = {best_iou:.3f}; no match"
        )
        assert best_native is not None
        score_diff = abs(float(best_native["score"]) - float(cpu_score))
        assert score_diff <= 0.1, (
            f"score mismatch: cpu={cpu_score:.4f} native={best_native['score']:.4f}"
        )
        matched += 1

    # Allow native NMS to remove slightly fewer/more due to floating differences,
    # but every CPU oracle face must be recoverable.
    assert matched == len(cpu_boxes), (
        f"matched {matched}/{len(cpu_boxes)} CPU detections"
    )
    print(f"Frame {frame_idx} parity OK ({matched} matches)")


def main() -> int:
    video = REPO / "backend" / "artifacts" / "videos" / "friendsshort_50f.mp4"
    jsonl = REPO / "backend" / "out" / "sprint01_50f_acceptance" / "detections.jsonl"
    if not video.exists():
        print(f"SKIP: video not found: {video}", file=sys.stderr)
        return 0
    if not jsonl.exists():
        # Fall back to last smoke output if acceptance run has not populated it.
        jsonl = REPO / "backend" / "out" / "sprint01_50f_v3" / "detections.jsonl"
    if not jsonl.exists():
        print(f"SKIP: native detections not found: {jsonl}", file=sys.stderr)
        return 0

    test_frame_parity(video, jsonl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
