#!/usr/bin/env python3
"""Frame identity gate for native preprocess tensor dumps.

For frames 38-45 the native worker must dump the NVDEC/nvdspreprocess tensor
with correct frame/PTS association. This test compares each native tensor
against the CPU OpenCV/ffmpeg reference tensor for the same frame index and
against every other frame index. The diagonal (native frame k vs CPU frame k)
must be the best match.

The test fails if dumps are stale, truncated, duplicated, mislabelled, contain
NaN/Inf, or if frame identity cannot be proven.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "backend" / "tests" / "fixtures" / "cpu_oracle"))

from cpu_oracle_retinaface import RETINA_INPUT_SIZE, RETINA_MEAN, preprocess

VIDEO = REPO / "backend" / "artifacts" / "videos" / "friendsshort_50f.mp4"
DUMP_DIR = REPO / "backend" / "out" / "preproc_dump"
REPORT_PATH = REPO / "backend" / "out" / "frame_identity_report.json"
META_FRAMES = list(range(38, 46))
TENSOR_ELM = 1 * 3 * 640 * 640
EXPECTED_BYTES = TENSOR_ELM * 4


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _ssim_channel(a: np.ndarray, b: np.ndarray) -> float:
    """Compute SSIM on a single 2D array (already rescaled to [0,255])."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ksize = 11
    sigma = 1.5
    mu1 = cv2.GaussianBlur(a, (ksize, ksize), sigma)
    mu2 = cv2.GaussianBlur(b, (ksize, ksize), sigma)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(a * a, (ksize, ksize), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b * b, (ksize, ksize), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(a * b, (ksize, ksize), sigma) - mu12
    ssim_map = ((2.0 * mu12 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(np.mean(ssim_map))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over the three channels after restoring [0,255] dynamic range."""
    a_u8 = np.clip(a + RETINA_MEAN.reshape(1, 3, 1, 1), 0.0, 255.0).astype(np.uint8)
    b_u8 = np.clip(b + RETINA_MEAN.reshape(1, 3, 1, 1), 0.0, 255.0).astype(np.uint8)
    vals = []
    for c in range(3):
        vals.append(_ssim_channel(a_u8[0, c], b_u8[0, c]))
    return float(np.mean(vals))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_native(frame_idx: int) -> tuple[np.ndarray, dict]:
    bin_path = DUMP_DIR / f"preproc_{frame_idx}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)
    if bin_path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(
            f"{bin_path} size {bin_path.stat().st_size} != {EXPECTED_BYTES}"
        )
    tensor = np.fromfile(bin_path, dtype=np.float32).reshape(1, 3, 640, 640)
    if not np.all(np.isfinite(tensor)):
        raise ValueError(f"{bin_path} contains NaN or Inf")

    meta_path = DUMP_DIR / f"preproc_{frame_idx}.json"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    meta["sha256"] = _sha256_file(bin_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    return tensor, meta


def _cpu_tensor(frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(VIDEO))
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read CPU frame {frame_idx}")
    return preprocess(frame)


def _build_matrix(native: dict[int, np.ndarray], cpu: dict[int, np.ndarray]) -> dict:
    frames = sorted(native.keys())
    cpu_frames = sorted(cpu.keys())
    matrix = []
    diagonal_best = True
    for nf in frames:
        row = []
        best_mae = float("inf")
        best_cf = None
        for cf in cpu_frames:
            mae = float(np.mean(np.abs(native[nf] - cpu[cf])))
            psnr = _psnr(native[nf], cpu[cf])
            ssim = _ssim(native[nf], cpu[cf])
            row.append(
                {
                    "native_frame": nf,
                    "cpu_frame": cf,
                    "mae": round(mae, 6),
                    "psnr": round(psnr, 4) if math.isfinite(psnr) else "inf",
                    "ssim": round(ssim, 6),
                }
            )
            if mae < best_mae:
                best_mae = mae
                best_cf = cf
        matrix.append(row)
        if best_cf != nf:
            diagonal_best = False

    return {
        "frames_evaluated": frames,
        "cpu_frames": cpu_frames,
        "matrix": matrix,
        "diagonal_is_best_match": diagonal_best,
    }


def main() -> int:
    if not VIDEO.exists():
        print(f"FAIL: video missing: {VIDEO}", file=sys.stderr)
        return 1
    if not DUMP_DIR.exists():
        print(f"FAIL: native dump dir missing: {DUMP_DIR}", file=sys.stderr)
        return 1

    native: dict[int, np.ndarray] = {}
    metas: dict[int, dict] = {}
    failed: list[str] = []

    for f in META_FRAMES:
        try:
            tensor, meta = _load_native(f)
            native[f] = tensor
            metas[f] = meta
        except Exception as e:
            failed.append(f"frame {f}: {e}")

    sha_values = [m.get("sha256", "") for m in metas.values()]
    if len(set(sha_values)) != len(sha_values):
        failed.append("duplicate tensor SHA-256 detected across frames")

    if failed:
        for msg in failed:
            print(f"FAIL: {msg}", file=sys.stderr)
        return 1

    cpu: dict[int, np.ndarray] = {}
    for f in META_FRAMES:
        try:
            cpu[f] = _cpu_tensor(f)
        except Exception as e:
            print(f"FAIL: CPU tensor error: {e}", file=sys.stderr)
            return 1

    matrix_report = _build_matrix(native, cpu)

    # Per-frame native vs CPU diagonal summary.
    per_frame = []
    for f in META_FRAMES:
        diff = np.abs(native[f] - cpu[f])
        per_frame.append(
            {
                "frame": f,
                "native_to_cpu_mae": round(float(diff.mean()), 6),
                "native_to_cpu_max": round(float(diff.max()), 6),
                "native_min": round(float(native[f].min()), 4),
                "native_max": round(float(native[f].max()), 4),
                "native_mean": round(float(native[f].mean()), 4),
                "cpu_min": round(float(cpu[f].min()), 4),
                "cpu_max": round(float(cpu[f].max()), 4),
                "cpu_mean": round(float(cpu[f].mean()), 4),
            }
        )
    matrix_report["per_frame"] = per_frame
    matrix_report["metadata"] = metas

    REPORT_PATH.write_text(json.dumps(matrix_report, indent=2))
    print(f"Frame identity report: {REPORT_PATH}")

    if not matrix_report["diagonal_is_best_match"]:
        print("FAIL: diagonal is not the best CPU match for every native frame", file=sys.stderr)
        return 1

    print("PASS: frame identity proven for frames 38-45")
    for row in matrix_report["matrix"]:
        diag = [e for e in row if e["native_frame"] == e["cpu_frame"]][0]
        print(
            f"  native {diag['native_frame']} vs CPU {diag['cpu_frame']}: "
            f"MAE={diag['mae']:.6f} PSNR={diag['psnr']} SSIM={diag['ssim']:.6f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
