#!/usr/bin/env python3
"""Engine/tensor parity gate.

Feeds the same 640x640 FP32 BGR tensor to CPU ONNX Runtime and the TensorRT
FP16 engine, then compares raw tensors and semantic post-NMS detections.

FP16 quantization shifts loc/landms across the 16800 anchor grid; the verdict
is based on face-confidence active anchors and decoded output parity.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "backend" / "tests" / "fixtures" / "cpu_oracle"))
sys.path.insert(0, str(REPO / "backend" / "tests" / "native"))

from cpu_oracle_retinaface import preprocess, RETINA_ONNX, _build_priors
from detector_parity_lib import (
    ENGINE, sha256, softmax, run_onnx, run_trtexec, parse_trt_outputs,
    decode_probs, semantic_compare, HOST_SCRATCH,
)

REPORT_PATH = REPO / "backend" / "out" / "engine_parity_report.json"
PIPELINE_REPORT = REPO / "backend" / "out" / "pipeline_parity_report.json"
MANIFEST = REPO / "backend" / "out" / "sprint01_50f_acceptance" / "run_manifest.json"
VIDEO = REPO / "backend" / "artifacts" / "videos" / "friendsshort_50f.mp4"

PRIORS = _build_priors(640)
NMS_THRESHOLD = 0.4

# Frozen thresholds for FP16 engine parity.
ACTIVE_CONF_MEAN_MAX = 0.005
ACTIVE_CONF_MAX_MAX = 0.02
SEM_IOU_MIN = 0.98
SEM_LANDMARK_MAX = 2.0
SEM_SCORE_MAX = 0.01


def runtime_info():
    info = {
        "tensorrt_version": "unknown",
        "cuda_runtime_version": "unknown",
        "gpu_name": "unknown",
        "gstreamer_version": "unknown",
    }
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        info["tensorrt_version"] = m.get("tensorrt_version", "unknown")
        info["cuda_runtime_version"] = m.get("cuda_runtime_version", "unknown")
        info["gpu_name"] = m.get("gpu_name", "unknown")
        info["gstreamer_version"] = m.get("gstreamer_version", "unknown")
    return info


def choose_frames():
    if not PIPELINE_REPORT.exists():
        return [0, 41, 42]
    data = json.loads(PIPELINE_REPORT.read_text())
    matched = sorted({int(m["frame"]) for m in data.get("match_details", [])})
    chosen = [0, 41, 42]
    for f in matched:
        if f not in chosen:
            chosen.append(f)
    return chosen[:24]


def analyze_frame(frame_idx: int):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx}")
    h, w = frame.shape[:2]

    tensor = preprocess(frame)
    cpu_loc, cpu_conf_logits, cpu_landms = run_onnx(tensor, RETINA_ONNX)
    trt_raw = run_trtexec(frame_idx, tensor)
    trt_loc, trt_conf_logits, trt_landms = parse_trt_outputs(trt_raw)

    cpu_conf_p = softmax(cpu_conf_logits[0])
    trt_conf_p = softmax(trt_conf_logits[0])

    d_loc = np.abs(cpu_loc - trt_loc)
    d_conf = np.abs(cpu_conf_p - trt_conf_p)
    d_landms = np.abs(cpu_landms - trt_landms)

    active = cpu_conf_p[:, 1] >= 0.1
    active_count = int(active.sum())

    top_anchors = np.argsort(-cpu_conf_p[:, 1])[:25]
    top_comparison = []
    for a in top_anchors:
        top_comparison.append({
            "anchor": int(a),
            "cpu_face_conf": float(round(cpu_conf_p[a, 1], 6)),
            "trt_face_conf": float(round(trt_conf_p[a, 1], 6)),
            "conf_delta": float(round(abs(cpu_conf_p[a, 1] - trt_conf_p[a, 1]), 6)),
            "loc_delta_max": float(round(d_loc[0, a].max(), 6)),
            "landms_delta_max": float(round(d_landms[0, a].max(), 6)),
        })

    # Decode at 0.45 so boundary-scored detections (like frame 41) are included
    # in the semantic comparison; the production 0.5 threshold is tested by the
    # pipeline parity gate, which explicitly handles the [0.49,0.51] band.
    cpu_boxes, cpu_scores, cpu_kps = decode_probs(
        cpu_loc[0], cpu_conf_p, cpu_landms[0], 0.45, PRIORS, NMS_THRESHOLD, (w, h)
    )
    trt_boxes, trt_scores, trt_kps = decode_probs(
        trt_loc[0], trt_conf_p, trt_landms[0], 0.45, PRIORS, NMS_THRESHOLD, (w, h)
    )
    semantic = semantic_compare(cpu_boxes, cpu_scores, cpu_kps, trt_boxes, trt_scores, trt_kps, SEM_IOU_MIN)

    return {
        "frame": frame_idx,
        "all_anchor_loc": {
            "mean_abs": float(d_loc.mean()), "max_abs": float(d_loc.max()), "p95_abs": float(np.percentile(d_loc, 95))
        },
        "all_anchor_conf_face": {
            "mean_abs": float(d_conf[:, 1].mean()), "max_abs": float(d_conf[:, 1].max()), "p95_abs": float(np.percentile(d_conf[:, 1], 95))
        },
        "all_anchor_landms": {
            "mean_abs": float(d_landms.mean()), "max_abs": float(d_landms.max()), "p95_abs": float(np.percentile(d_landms, 95))
        },
        "active_anchors_count": active_count,
        "active_anchor_conf_face": {
            "mean_abs": float(d_conf[active, 1].mean()) if active_count else None,
            "max_abs": float(d_conf[active, 1].max()) if active_count else None,
        },
        "top_anchors": top_comparison,
        "semantic_post_nms": semantic,
        "post_nms_cpu_count": int(len(cpu_boxes)),
        "post_nms_trt_count": int(len(trt_boxes)),
    }


def main():
    if not ENGINE.exists():
        print(f"FAIL: engine missing {ENGINE}", file=sys.stderr)
        return 1
    if not RETINA_ONNX.exists():
        print(f"FAIL: onnx missing {RETINA_ONNX}", file=sys.stderr)
        return 1

    frames = choose_frames()
    if not frames:
        print("FAIL: no frames selected", file=sys.stderr)
        return 1

    if HOST_SCRATCH.exists():
        shutil.rmtree(HOST_SCRATCH)

    report = {
        "status": "unknown",
        "thresholds": {
            "active_conf_mean_abs_max": ACTIVE_CONF_MEAN_MAX,
            "active_conf_max_abs_max": ACTIVE_CONF_MAX_MAX,
            "semantic_iou_min": SEM_IOU_MIN,
            "semantic_landmark_max_px": SEM_LANDMARK_MAX,
            "semantic_score_max": SEM_SCORE_MAX,
        },
        "engine_sha256": sha256(ENGINE),
        "onnx_sha256": sha256(RETINA_ONNX),
        "runtime": runtime_info(),
        "frames_evaluated": frames,
        "frames": [],
        "violations": [],
    }

    passed = True
    for f in frames:
        fr = analyze_frame(f)
        report["frames"].append(fr)
        ac = fr["active_anchor_conf_face"]
        sem = fr["semantic_post_nms"]

        if ac["mean_abs"] is not None:
            if ac["mean_abs"] > ACTIVE_CONF_MEAN_MAX:
                report["violations"].append({"frame": f, "gate": "active_conf_mean_abs", "value": ac["mean_abs"]})
                passed = False
            if ac["max_abs"] > ACTIVE_CONF_MAX_MAX:
                report["violations"].append({"frame": f, "gate": "active_conf_max_abs", "value": ac["max_abs"]})
                passed = False

        if sem["iou_min"] is not None and sem["iou_min"] < SEM_IOU_MIN:
            report["violations"].append({"frame": f, "gate": "semantic_iou_min", "value": sem["iou_min"]})
            passed = False
        if sem["landmark_max"] is not None and sem["landmark_max"] > SEM_LANDMARK_MAX:
            report["violations"].append({"frame": f, "gate": "semantic_landmark_max", "value": sem["landmark_max"]})
            passed = False
        if sem["score_max"] is not None and sem["score_max"] > SEM_SCORE_MAX:
            report["violations"].append({"frame": f, "gate": "semantic_score_max", "value": sem["score_max"]})
            passed = False

        print(
            f"frame {f}: active_conf_mean={ac['mean_abs']:.6f} active_conf_max={ac['max_abs']:.6f} | "
            f"iou_min={sem['iou_min']} landmark_max={sem['landmark_max']} score_max={sem['score_max']}"
        )

    report["status"] = "PASS" if passed else "FAIL"
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Engine parity report: {REPORT_PATH}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
