#!/usr/bin/env python3
"""Render detections.jsonl onto the source video for human review.

This is a bounded debugging/visualization tool, not a production pipeline stage.
It overlays bounding boxes, confidence scores, placeholder track/identity labels,
and landmarks at original resolution.
"""
import argparse
import json
import math
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit("OpenCV is required: python3-opencv") from exc


COLOR_BOX = (0, 255, 0)          # green
COLOR_LANDMARK = (0, 165, 255)   # orange
COLOR_TEXT = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2


def load_detections(path: Path) -> dict:
    det_by_frame = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            det_by_frame[int(rec["frame"])] = rec.get("detections", [])
    return det_by_frame


def draw_detections(frame, detections, track_label: str, face_id_label: str):
    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        score = det.get("score", 0.0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, THICKNESS)

        label = f"{face_id_label} {track_label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLOR_BOX, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), FONT, FONT_SCALE, (0, 0, 0), THICKNESS)

        landmarks = det.get("landmarks", [])
        for i in range(0, len(landmarks), 2):
            lx, ly = int(round(landmarks[i])), int(round(landmarks[i + 1]))
            cv2.circle(frame, (lx, ly), 3, COLOR_LANDMARK, -1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Render detections over a video.")
    parser.add_argument("input_video", type=Path)
    parser.add_argument("detections_jsonl", type=Path)
    parser.add_argument("output_video", type=Path)
    parser.add_argument("--track-label", default="UNTRACKED")
    parser.add_argument("--face-id-label", default="unknown")
    args = parser.parse_args()

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    det_by_frame = load_detections(args.detections_jsonl)

    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open {args.input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(args.output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open video writer for {args.output_video}")

    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = det_by_frame.get(frame_idx, [])
        if detections:
            draw_detections(frame, detections, args.track_label, args.face_id_label)
        writer.write(frame)
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Wrote {written} frames to {args.output_video}")


if __name__ == "__main__":
    main()
