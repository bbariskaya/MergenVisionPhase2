"""Shared helpers for detector parity tests and diagnostics."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[3]

ENGINE = REPO / "backend" / "artifacts" / "engines" / "retinaface_r50_dynamic.bs1.opt64.max256.fp16.trt1014.engine"
CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
HOST_SCRATCH = Path("/tmp/mergenvision_engine_parity")

_VARIANCE = np.array([0.1, 0.2], dtype=np.float32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def run_onnx(tensor: np.ndarray, onnx_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    return session.run(None, {in_name: tensor})


def run_trtexec(frame_idx: int, tensor: np.ndarray, scratch: Path = HOST_SCRATCH) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    input_bin = scratch / f"input_{frame_idx}.bin"
    output_json = scratch / f"trt_{frame_idx}.json"
    tensor.tofile(input_bin)
    cmd = [
        "docker", "run", "--rm", "--gpus", "device=0",
        "-v", f"{scratch}:/tmp/out",
        "-v", f"{REPO}:/app",
        "-w", "/app",
        CONTAINER,
        TRTEXEC,
        f"--loadEngine=/app/{ENGINE.relative_to(REPO)}",
        f"--loadInputs=input:/tmp/out/input_{frame_idx}.bin",
        f"--exportOutput=/tmp/out/trt_{frame_idx}.json",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(output_json.read_text())


def parse_trt_outputs(raw: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = [entry["name"] for entry in raw]
    values = [np.array(entry["values"], dtype=np.float32) for entry in raw]
    order = [names.index(n) for n in ["loc", "conf", "landms"]]
    loc = values[order[0]].reshape(1, 16800, 4)
    conf = values[order[1]].reshape(1, 16800, 2)
    landms = values[order[2]].reshape(1, 16800, 10)
    return loc, conf, landms


def cpu_bgr(frame_idx: int, video: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx}")
    return frame


def decode_probs(
    loc: np.ndarray,
    conf_p: np.ndarray,
    landms: np.ndarray,
    conf_threshold: float,
    priors: np.ndarray,
    nms_threshold: float,
    original_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = conf_p[:, 1]
    valid = scores >= conf_threshold
    if valid.sum() == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 5, 2))

    loc = loc[valid]
    landms = landms[valid]
    scores = scores[valid]
    pr = priors[valid]

    cx = pr[:, 0] + loc[:, 0] * _VARIANCE[0] * pr[:, 2]
    cy = pr[:, 1] + loc[:, 1] * _VARIANCE[0] * pr[:, 3]
    w = pr[:, 2] * np.exp(loc[:, 2] * _VARIANCE[1])
    h = pr[:, 3] * np.exp(loc[:, 3] * _VARIANCE[1])
    boxes_640 = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    landmarks_640 = np.zeros((loc.shape[0], 5, 2), dtype=np.float32)
    for k in range(5):
        landmarks_640[:, k, 0] = pr[:, 0] + landms[:, k * 2] * _VARIANCE[0] * pr[:, 2]
        landmarks_640[:, k, 1] = pr[:, 1] + landms[:, k * 2 + 1] * _VARIANCE[0] * pr[:, 3]

    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes_640[:, 0], boxes_640[:, 1], boxes_640[:, 2], boxes_640[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w_int = np.maximum(0.0, xx2 - xx1)
        h_int = np.maximum(0.0, yy2 - yy1)
        inter = w_int * h_int
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= nms_threshold]

    boxes_640 = boxes_640[keep]
    landmarks_640 = landmarks_640[keep]
    scores = scores[keep]

    ow, oh = original_wh
    boxes = boxes_640.copy()
    boxes[:, [0, 2]] *= ow
    boxes[:, [1, 3]] *= oh
    landmarks = landmarks_640.copy()
    landmarks[:, :, 0] *= ow
    landmarks[:, :, 1] *= oh
    return boxes, scores, landmarks


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def semantic_compare(cpu_boxes, cpu_scores, cpu_lms, trt_boxes, trt_scores, trt_lms, iou_min: float):
    if cpu_boxes.shape[0] == 0 and trt_boxes.shape[0] == 0:
        return {"matches": 0, "iou_min": None, "landmark_max": None, "score_max": None}
    if cpu_boxes.shape[0] == 0 or trt_boxes.shape[0] == 0:
        return {"matches": 0, "iou_min": 0.0, "landmark_max": None, "score_max": None}
    matches = []
    used = set()
    order = np.argsort(-cpu_scores, kind="stable")
    for i in order:
        best_j = None
        best_iou = iou_min
        for j in range(len(trt_boxes)):
            if j in used:
                continue
            val = iou(cpu_boxes[i], trt_boxes[j])
            if val > best_iou:
                best_iou = val
                best_j = j
        if best_j is None:
            continue
        used.add(best_j)
        lm_err = float(np.max(np.linalg.norm(cpu_lms[i] - trt_lms[best_j], axis=1)))
        matches.append({
            "cpu_idx": int(i), "trt_idx": int(best_j),
            "iou": round(float(best_iou), 6),
            "landmark_err_px": round(lm_err, 6),
            "score_delta": round(abs(float(cpu_scores[i]) - float(trt_scores[best_j])), 6),
        })
    if not matches:
        return {"matches": 0, "iou_min": 0.0, "landmark_max": None, "score_max": None}
    return {
        "matches": len(matches),
        "iou_min": round(min(m["iou"] for m in matches), 6),
        "landmark_max": round(max(m["landmark_err_px"] for m in matches), 6),
        "score_max": round(max(m["score_delta"] for m in matches), 6),
        "details": matches,
    }
