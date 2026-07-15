#!/usr/bin/env python3
"""Compare native NVDEC/nvdspreprocess output to CPU oracle preprocess tensor.

Native tensor is produced by running deepstream_face_worker with
MV_DUMP_PREPROC_TENSOR=<dir>. This is test-only instrumentation and is off by
default in production.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend" / "tests" / "fixtures" / "cpu_oracle"))
from cpu_oracle_retinaface import preprocess

VIDEO = REPO / "backend" / "artifacts" / "videos" / "friendsshort_50f.mp4"
DUMP_DIR = REPO / "backend" / "out" / "preproc_dump"
CHANNEL_NAMES = ["B", "G", "R"]
MEAN = np.array([104.0, 117.0, 123.0], dtype=np.float32)


def _load_native(frame_idx: int) -> np.ndarray:
    path = DUMP_DIR / f"preproc_{frame_idx}.bin"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.fromfile(path, dtype=np.float32).reshape(1, 3, 640, 640)


def _compare_frame(frame_idx: int) -> dict:
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx}")

    cpu = preprocess(frame)
    native = _load_native(frame_idx)

    diff = np.abs(cpu - native)
    result = {"frame": int(frame_idx)}
    overall_mae = float(diff.mean())
    overall_max = float(diff.max())
    result["mae"] = round(overall_mae, 6)
    result["max_abs"] = round(overall_max, 6)
    result["channels"] = []

    # channel-wise statistics (NCHW)
    for ch, name in enumerate(CHANNEL_NAMES):
        ch_diff = diff[0, ch]
        mae = float(ch_diff.mean())
        max_abs = float(ch_diff.max())
        max_pos = np.unravel_index(np.argmax(ch_diff), ch_diff.shape)
        result["channels"].append(
            {
                "channel": name,
                "mae": round(mae, 6),
                "max_abs": round(max_abs, 6),
                "max_pos": [int(max_pos[0]), int(max_pos[1])],
            }
        )

    # Report the worst pixel (after mean offset values)
    flat_idx = int(np.argmax(diff))
    n, c, h, w = np.unravel_index(flat_idx, diff.shape)
    result["worst_pixel"] = {
        "channel": CHANNEL_NAMES[int(c)],
        "y": int(h),
        "x": int(w),
        "cpu_value": float(cpu[n, c, h, w]),
        "native_value": float(native[n, c, h, w]),
        "diff": float(diff[n, c, h, w]),
    }

    # Crop MAE / max over center 160x160 and over face region if known.
    # For the friendsshort clip the face is roughly in the middle-right.
    crop = diff[0, :, 240:400, 240:400]
    result["center_crop_mae"] = round(float(crop.mean()), 6)
    result["center_crop_max"] = round(float(crop.max()), 6)
    return result


def main() -> int:
    if not DUMP_DIR.exists():
        print(f"FAIL: native preprocess dump dir not found: {DUMP_DIR}", file=sys.stderr)
        print("Run deepstream_face_worker with MV_DUMP_PREPROC_TENSOR=<dir>", file=sys.stderr)
        return 1

    for f in (41, 42):
        r = _compare_frame(f)
        print(f"\n=== preprocess parity frame {f} ===")
        print(f"MAE over full image: {r['mae']:.6f}")
        print(f"Max absolute diff:   {r['max_abs']:.6f}")
        for ch in r["channels"]:
            print(
                f"  channel {ch['channel']}: MAE={ch['mae']:.6f} "
                f"max={ch['max_abs']:.6f} @ {ch['max_pos']}"
            )
        print(f"Worst pixel: {r['worst_pixel']}")
        print(f"Center 160x160 crop MAE={r['center_crop_mae']:.6f} max={r['center_crop_max']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
