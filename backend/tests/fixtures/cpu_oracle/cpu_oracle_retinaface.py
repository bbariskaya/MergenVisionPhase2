"""CPU-only RetinaFace oracle for detector parity tests.

Uses OpenCV decode/resize and ONNX Runtime inference. This is intentionally not
production code; it provides a reference baseline for the native GPU pipeline.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[4]
RETINA_ONNX = REPO_ROOT / "backend" / "artifacts" / "models" / "retinaface_r50_dynamic.onnx"

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
RETINA_INPUT_SIZE = 640
RETINA_MEAN = np.array([104.0, 117.0, 123.0], dtype=np.float32)
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
                s_kx = min_size / image_size
                s_ky = min_size / image_size
                cx = (j + 0.5) * step / image_size
                cy = (i + 0.5) * step / image_size
                anchors += [cx, cy, s_kx, s_ky]
    return np.array(anchors, dtype=np.float32).reshape(-1, 4)


_PRIORS = _build_priors(RETINA_INPUT_SIZE)


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image_bgr, (RETINA_INPUT_SIZE, RETINA_INPUT_SIZE)).astype(np.float32)
    resized -= RETINA_MEAN
    tensor = resized.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(tensor, dtype=np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float, top_k: int = 2000) -> list[int]:
    order = np.argsort(-scores, kind="stable")[:top_k]
    keep: list[int] = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= threshold]
    return keep


def decode(
    loc: np.ndarray,
    conf: np.ndarray,
    landms: np.ndarray,
    original_wh: tuple[int, int],
    conf_threshold: float = CONF_THRESHOLD,
    nms_threshold: float = NMS_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boxes/landmarks/scores in original image coordinates."""
    scores = conf[:, 1]
    valid = scores >= conf_threshold
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

    keep = _nms(boxes_640, scores, nms_threshold)
    boxes_640 = boxes_640[keep]
    landmarks_640 = landmarks_640[keep]
    scores = scores[keep]

    orig_w, orig_h = original_wh
    boxes_orig = boxes_640.copy()
    boxes_orig[:, [0, 2]] *= orig_w
    boxes_orig[:, [1, 3]] *= orig_h
    landmarks_orig = landmarks_640.copy()
    landmarks_orig[:, :, 0] *= orig_w
    landmarks_orig[:, :, 1] *= orig_h
    return boxes_orig, scores, landmarks_orig


def detect_image(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the ONNX RetinaFace oracle on a single BGR image."""
    if not RETINA_ONNX.exists():
        raise FileNotFoundError(f"RetinaFace ONNX model not found at {RETINA_ONNX}")

    session = ort.InferenceSession(str(RETINA_ONNX), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    tensor = preprocess(image_bgr)
    loc, conf, landms = session.run(None, {input_name: tensor})

    h, w = image_bgr.shape[:2]
    return decode(loc[0], conf[0], landms[0], (w, h))
