"""Side-by-side YOLO11 COCO detection vs RetinaFace face detection on Phoebe images.

Output: out/phoebe_yolo11_vs_retinaface/<stem>_compare.jpg

YOLO11 is a general 80-class COCO detector; guitar is *not* one of those classes,
but the script will draw every COCO class it actually detects (person, cup, chair,
etc.) for comparison with RetinaFace face boxes.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
DATASET_DIR = ARTIFACTS_DIR / "gallery" / "Phoebe"
OUT_DIR = REPO_ROOT / "out" / "phoebe_yolo11_vs_retinaface"

RETINA_ONNX = MODELS_DIR / "retinaface_r50_dynamic.onnx"
RETINA_INPUT_SIZE = 640
RETINA_MEAN = np.array([104.0, 117.0, 123.0], dtype=np.float32)
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
_VARIANCE = np.array([0.1, 0.2], dtype=np.float32)


def _build_priors(image_size: int = 640) -> np.ndarray:
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    anchors = []
    for k, step in enumerate(steps):
        f_h = int(np.ceil(image_size / step))
        f_w = int(np.ceil(image_size / step))
        for i, j in product(range(f_h), range(f_w)):
            for min_size in min_sizes[k]:
                cx = (j + 0.5) * step / image_size
                cy = (i + 0.5) * step / image_size
                anchors += [cx, cy, min_size / image_size, min_size / image_size]
    return np.array(anchors, dtype=np.float32).reshape(-1, 4)


_PRIORS = _build_priors(RETINA_INPUT_SIZE)


def _nms_cpu(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[1:][iou <= threshold]
    return keep


def decode_retinaface(
    loc: np.ndarray, conf: np.ndarray, landms: np.ndarray, orig_wh: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = conf[:, 1]
    valid = scores >= CONF_THRESHOLD
    if valid.sum() == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 5, 2))
    loc = loc[valid]
    landms = landms[valid]
    scores = scores[valid]
    priors = _PRIORS[valid]

    cx = priors[:, 0] + loc[:, 0] * _VARIANCE[0] * priors[:, 2]
    cy = priors[:, 1] + loc[:, 1] * _VARIANCE[0] * priors[:, 3]
    w = priors[:, 2] * np.exp(loc[:, 2] * _VARIANCE[1])
    h = priors[:, 3] * np.exp(loc[:, 3] * _VARIANCE[1])
    boxes_640 = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    landmarks_640 = np.zeros((loc.shape[0], 5, 2), dtype=np.float32)
    for k in range(5):
        landmarks_640[:, k, 0] = priors[:, 0] + landms[:, k * 2] * _VARIANCE[0] * priors[:, 2]
        landmarks_640[:, k, 1] = priors[:, 1] + landms[:, k * 2 + 1] * _VARIANCE[0] * priors[:, 3]

    keep = _nms_cpu(boxes_640, scores, NMS_THRESHOLD)
    boxes_640, landmarks_640, scores = boxes_640[keep], landmarks_640[keep], scores[keep]

    scale_x, scale_y = float(orig_wh[0]), float(orig_wh[1])
    boxes = boxes_640.copy()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    landmarks = landmarks_640.copy()
    landmarks[:, :, 0] *= scale_x
    landmarks[:, :, 1] *= scale_y
    return boxes, scores, landmarks


def preprocess_detector(image_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image_bgr, (RETINA_INPUT_SIZE, RETINA_INPUT_SIZE)).astype(np.float32)
    resized -= RETINA_MEAN
    return np.ascontiguousarray(resized.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)


def run_yolo(model: Any, image_bgr: np.ndarray) -> list[tuple[np.ndarray, int, float, str]]:
    results = model(image_bgr, device="cpu", verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    names = model.names
    detections = []
    for box, c, s in zip(xyxy, cls, conf):
        if s < 0.25:
            continue
        detections.append((box, int(c), float(s), names.get(int(c), str(int(c)))))
    return detections


def run_retinaface(session: ort.InferenceSession, image_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    h, w = image_bgr.shape[:2]
    inp = preprocess_detector(image_bgr)
    name = session.get_inputs()[0].name
    loc, conf, landms = session.run(None, {name: inp})
    boxes, scores, landmarks = decode_retinaface(loc[0], conf[0], landms[0], (w, h))
    if len(boxes) == 0:
        return None, None, 0.0
    best = int(np.argmax(scores))
    return boxes[best].astype(int), landmarks[best], float(scores[best])


def draw_panel(image: np.ndarray, yolo_dets: list[tuple], face_box: np.ndarray | None, landmarks: np.ndarray | None, face_score: float) -> np.ndarray:
    canvas = image.copy()
    for i, (box, cls_id, score, label) in enumerate(yolo_dets):
        x1, y1, x2, y2 = map(int, box)
        # stagger text vertically to avoid overlap
        text_y = max(y1 - 10 - (i * 18), 20)
        color = (255, 0, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"YOLO {label} {score:.2f}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if face_box is not None:
        x1, y1, x2, y2 = map(int, face_box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, f"RetinaFace score={face_score:.3f}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if landmarks is not None:
        for idx, (lx, ly) in enumerate(landmarks):
            color = (0, 255, 0) if idx < 2 else (0, 255, 255) if idx == 2 else (255, 0, 0)
            cv2.circle(canvas, (int(lx), int(ly)), 3, color, -1)
    return canvas


def main() -> int:
    from ultralytics import YOLO

    if not RETINA_ONNX.exists():
        print(f"FATAL: {RETINA_ONNX} not found", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yolo_model = YOLO("yolo11n.pt")
    retina_session = ort.InferenceSession(str(RETINA_ONNX), providers=["CPUExecutionProvider"])

    paths = sorted(DATASET_DIR.glob("*.jpg")) + sorted(DATASET_DIR.glob("*.png"))
    count = 0
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        yolo_dets = run_yolo(yolo_model, img)
        face_box, landmarks, face_score = run_retinaface(retina_session, img)
        panel = draw_panel(img, yolo_dets, face_box, landmarks, face_score)
        out_path = OUT_DIR / f"{p.stem}_compare.jpg"
        cv2.imwrite(str(out_path), panel)
        count += 1

    print(f"Wrote {count} comparison images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
