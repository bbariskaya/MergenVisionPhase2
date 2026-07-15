#!/usr/bin/env python3
"""Native-tensor decisive experiment.

Load the exact NVDEC/nvdspreprocess tensor dumped for frames 41/42, feed it to:
  - ONNX Runtime CPU (FP32)
  - the production TensorRT FP16 engine (trtexec --loadInputs)
without any further preprocessing.

Report anchor 4683, top anchors, and post-threshold/post-NMS counts. This
proves whether a missing detection (frame 41) is caused by preprocess drift or
by CUDA postprocess.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend" / "tests" / "fixtures" / "cpu_oracle"))

from cpu_oracle_retinaface import (
    RETINA_INPUT_SIZE,
    NMS_THRESHOLD,
    preprocess,
    RETINA_ONNX,
    _build_priors,
    _nms,
)

ENGINE = REPO / "backend" / "artifacts" / "engines" / "retinaface_r50_dynamic.bs1.opt64.max256.fp16.trt1014.engine"
DUMP_DIR = REPO / "backend" / "out" / "preproc_dump"
REPORT_PATH = REPO / "backend" / "out" / "native_tensor_decision_report.json"
CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
HOST_SCRATCH = Path("/tmp/mergenvision_native_tensor_parity")
ANCHOR_FOCUS = 4683
_VARIANCE = np.array([0.1, 0.2], dtype=np.float32)
_PRIORS = _build_priors(RETINA_INPUT_SIZE)


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _load_tensor(kind: str, frame_idx: int) -> np.ndarray:
    if kind == "native":
        path = DUMP_DIR / f"preproc_{frame_idx}.bin"
    else:
        path = HOST_SCRATCH / f"cpu_{frame_idx}.bin"
    return np.fromfile(path, dtype=np.float32).reshape(1, 3, 640, 640)


def _run_onnx(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    session = ort.InferenceSession(str(RETINA_ONNX), providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    return session.run(None, {in_name: tensor})


def _run_trtexec(frame_idx: int, tensor: np.ndarray) -> dict:
    HOST_SCRATCH.mkdir(parents=True, exist_ok=True)
    input_bin = HOST_SCRATCH / f"native_tensor_{frame_idx}.bin"
    output_json = HOST_SCRATCH / f"trt_native_tensor_{frame_idx}.json"
    tensor.tofile(input_bin)
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "device=0",
        "-v",
        f"{HOST_SCRATCH}:/tmp/out",
        "-v",
        f"{REPO}:/app",
        "-w",
        "/app",
        CONTAINER,
        TRTEXEC,
        f"--loadEngine=/app/{ENGINE.relative_to(REPO)}",
        f"--loadInputs=input:/tmp/out/native_tensor_{frame_idx}.bin",
        f"--exportOutput=/tmp/out/trt_native_tensor_{frame_idx}.json",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(output_json.read_text())


def _parse_trt_outputs(raw: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = [entry["name"] for entry in raw]
    values = [np.array(entry["values"], dtype=np.float32) for entry in raw]
    order = [names.index(n) for n in ["loc", "conf", "landms"]]
    loc = values[order[0]].reshape(1, 16800, 4)
    conf = values[order[1]].reshape(1, 16800, 2)
    landms = values[order[2]].reshape(1, 16800, 10)
    return loc, conf, landms


def _decode_probs(
    loc: np.ndarray,
    conf_p: np.ndarray,
    landms: np.ndarray,
    conf_threshold: float,
    original_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy decode using probabilities rather than logits."""
    scores = conf_p[:, 1]
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

    keep = _nms(boxes_640, scores, NMS_THRESHOLD)
    boxes_640 = boxes_640[keep]
    landmarks_640 = landmarks_640[keep]
    scores = scores[keep]

    orig_w, orig_h = original_wh
    boxes = boxes_640.copy()
    boxes[:, [0, 2]] *= orig_w
    boxes[:, [1, 3]] *= orig_h
    landmarks = landmarks_640.copy()
    landmarks[:, :, 0] *= orig_w
    landmarks[:, :, 1] *= orig_h
    return boxes, scores, landmarks


def _anchor_report(
    label: str,
    loc: np.ndarray,
    conf_logits: np.ndarray,
    conf_p: np.ndarray,
    landms: np.ndarray,
    frame_idx: int,
) -> dict:
    focus_logit = float(conf_logits[0, ANCHOR_FOCUS, 1])
    focus_prob = float(conf_p[0, ANCHOR_FOCUS, 1])
    top_idx = int(np.argmax(conf_p[0, :, 1]))
    top_prob = float(conf_p[0, top_idx, 1])

    anchors_ge_40 = [int(i) for i in np.where(conf_p[0, :, 1] >= 0.40)[0]]
    anchors_ge_40.sort(key=lambda i: conf_p[0, i, 1], reverse=True)
    anchors_report = []
    for i in anchors_ge_40[:25]:
        # Decode anchor i to a 640-space bbox for intuitive inspection.
        prior = _PRIORS[i]
        l = loc[0, i]
        cx = prior[0] + l[0] * _VARIANCE[0] * prior[2]
        cy = prior[1] + l[1] * _VARIANCE[0] * prior[3]
        w = prior[2] * math.exp(l[2] * _VARIANCE[1])
        h = prior[3] * math.exp(l[3] * _VARIANCE[1])
        boxes = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        kps = np.zeros((5, 2), dtype=np.float32)
        for k in range(5):
            kps[k, 0] = prior[0] + landms[0, i, k * 2] * _VARIANCE[0] * prior[2]
            kps[k, 1] = prior[1] + landms[0, i, k * 2 + 1] * _VARIANCE[0] * prior[3]
        anchors_report.append(
            {
                "anchor": i,
                "prob": float(round(conf_p[0, i, 1], 6)),
                "logit": float(round(conf_logits[0, i, 1], 6)),
                "bbox_640": boxes.round(4).tolist(),
                "landmarks_640": kps.round(4).tolist(),
            }
        )

    decoded_counts = {}
    for thr in (0.45, 0.5):
        boxes, scores, _ = _decode_probs(
            loc[0], conf_p[0], landms[0], thr, (1280, 720)
        )
        decoded_counts[str(thr)] = int(len(boxes))

    return {
        "label": label,
        "frame": frame_idx,
        "anchor_4683_logit": round(focus_logit, 6),
        "anchor_4683_prob": round(focus_prob, 6),
        "max_prob": round(top_prob, 6),
        "max_prob_anchor": int(top_idx),
        "anchors_ge_0.40_count": len(anchors_ge_40),
        "top_anchors_ge_0.40": anchors_report,
        "nms_survivor_counts": decoded_counts,
    }


def _analyze_frame(frame_idx: int) -> dict:
    HOST_SCRATCH.mkdir(parents=True, exist_ok=True)
    native = _load_tensor("native", frame_idx)
    cpu = preprocess(_cpu_bgr(frame_idx))
    cpu.tofile(HOST_SCRATCH / f"cpu_{frame_idx}.bin")

    cpu_loc, cpu_conf_logits, cpu_landms = _run_onnx(cpu)
    cpu_conf_p = _softmax(cpu_conf_logits[0])
    trt_raw = _run_trtexec(frame_idx, native)
    trt_loc, trt_conf_logits, trt_landms = _parse_trt_outputs(trt_raw)
    trt_conf_p = _softmax(trt_conf_logits[0])

    report = {
        "frame": frame_idx,
        "tensor_sha256_native": _sha256(DUMP_DIR / f"preproc_{frame_idx}.bin"),
        "tensor_sha256_cpu": _sha256(HOST_SCRATCH / f"cpu_{frame_idx}.bin"),
        "native_tensor_per_channel_mean": {
            "b": round(float(native[0, 0].mean()), 4),
            "g": round(float(native[0, 1].mean()), 4),
            "r": round(float(native[0, 2].mean()), 4),
        },
        "cpu_tensor_per_channel_mean": {
            "b": round(float(cpu[0, 0].mean()), 4),
            "g": round(float(cpu[0, 1].mean()), 4),
            "r": round(float(cpu[0, 2].mean()), 4),
        },
        "engine": _sha256(ENGINE),
        "onnx": _sha256(RETINA_ONNX),
        "onnx_report": _anchor_report(
            "ONNX_on_CPU_tensor", cpu_loc, cpu_conf_logits, cpu_conf_p[np.newaxis, ...], cpu_landms, frame_idx
        ),
        "trt_on_native_tensor_report": _anchor_report(
            "TensorRT_on_native_tensor", trt_loc, trt_conf_logits, trt_conf_p[np.newaxis, ...], trt_landms, frame_idx
        ),
    }

    # Same native tensor on CPU ONNX for reference.
    native_loc, native_conf_logits, native_landms = _run_onnx(native)
    native_conf_p = _softmax(native_conf_logits[0])
    report["onnx_on_native_tensor_report"] = _anchor_report(
        "ONNX_on_native_tensor",
        native_loc,
        native_conf_logits,
        native_conf_p[np.newaxis, ...],
        native_landms,
        frame_idx,
    )
    return report


def _cpu_bgr(frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(REPO / "backend" / "artifacts" / "videos" / "friendsshort_50f.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read CPU frame {frame_idx}")
    return frame


def main() -> int:
    if not ENGINE.exists():
        print(f"FAIL: engine missing {ENGINE}", file=sys.stderr)
        return 1
    if not RETINA_ONNX.exists():
        print(f"FAIL: onnx missing {RETINA_ONNX}", file=sys.stderr)
        return 1
    if not DUMP_DIR.exists():
        print(f"FAIL: native tensor dump dir missing: {DUMP_DIR}", file=sys.stderr)
        return 1

    report = {"frames": []}
    for f in (41, 42):
        report["frames"].append(_analyze_frame(f))

    report["decision_branch"] = ""
    f41_trt = report["frames"][0]["trt_on_native_tensor_report"]
    f41_native_count = f41_trt["nms_survivor_counts"]["0.45"]
    f41_trt_max = f41_trt["max_prob"]
    if f41_trt_max >= 0.45 and f41_native_count == 0:
        report["decision_branch"] = (
            "RAW_TRT_CONF_ABOVE_0.45_BUT_NO_OUTPUT: postprocess/association bug"
        )
    elif f41_trt_max < 0.45:
        report["decision_branch"] = (
            "RAW_TRT_CONF_BELOW_0.45: preprocess/decode drift; not a CUDA postprocess bug"
        )
    else:
        report["decision_branch"] = "LEGITIMATE_DETECTION_AT_LOWER_THRESHOLD"

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Native tensor decision report: {REPORT_PATH}")
    print(json.dumps(report["frames"][0]["trt_on_native_tensor_report"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
