#!/usr/bin/env python3
"""GPU hot-path contract test for DeepStream RetinaFace worker.

Verifies that per-frame processing does not copy full detector output tensors
(DTOH) or stage them back to the device (H2D). Only compact metadata D2H is
allowed. This test runs the worker under Nsight Systems and inspects the CUDA
memcpy trace.
"""
import json
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
WORKER = "/app/native/build/deepstream_face_worker"
INPUT_PATH = "/app/test_videos/friendsshort_50f.mp4"

# RetinaFace R50 640x640 full-output tensor byte sizes.
FULL_OUTPUT_D2H_SIZES = {
    16800 * 4 * 4,     # loc   [A, 4]  FP32
    16800 * 2 * 4,     # conf  [A, 2]  FP32
    16800 * 10 * 4,    # landms [A, 10] FP32
}
# Compact metadata budget: at most a few hundred bytes per detected face per
# frame; for 50 frames a generous bound.
MAX_TOTAL_DTOH_BYTES = 50 * 1024


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def _dockerize(entrypoint: str, *args: str, extra_env: dict | None = None) -> list[str]:
    env = {"CUDA_VISIBLE_DEVICES": "0", "GST_PLUGIN_PATH": "/app/native/build"}
    env.update(extra_env or {})
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "device=0",
        "-v", f"{REPO}:/app",
        "-w", "/app",
    ]
    for k, v in env.items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.extend(["--entrypoint", entrypoint, CONTAINER, *args])
    return cmd


def run_worker_under_nsys(output_basename: str) -> int:
    run_dir = REPO / "out" / f"{output_basename}_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    rep_path = f"/app/out/{output_basename}.nsys-rep"
    sqlite_path = f"/app/out/{output_basename}.sqlite"

    profile_cmd = _dockerize(
        "/usr/local/cuda/bin/nsys",
        "profile",
        "-t", "cuda,nvtx",
        "-o", f"/app/out/{output_basename}",
        "--force-overwrite", "true",
        WORKER,
        INPUT_PATH,
        f"/app/out/{output_basename}_run",
        "0",
    )
    result = _run(profile_cmd, check=False)
    worker_exit = result.returncode

    # Export to sqlite for inspection.
    _run(_dockerize(
        "/usr/local/cuda/bin/nsys",
        "export",
        "-t", "sqlite",
        "-o", sqlite_path,
        "--force-overwrite", "true",
        rep_path,
    ), check=True)

    return worker_exit, Path(REPO / "out" / f"{output_basename}.sqlite")


def load_memcpy_summary(sqlite_path: Path) -> dict:
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT oper.name,
               SUM(m.bytes), COUNT(*), MIN(m.bytes), MAX(m.bytes)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        JOIN ENUM_CUDA_MEMCPY_OPER oper ON m.copyKind = oper.id
        GROUP BY oper.name
        """
    )
    summary = {row[0]: {
        "total_bytes": row[1] or 0,
        "count": row[2],
        "min_bytes": row[3],
        "max_bytes": row[4],
    } for row in cur.fetchall()}
    cur.execute(
        """
        SELECT bytes, COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind = 2 AND bytes IN ({seq})
        GROUP BY bytes
        """.format(seq=",".join("?" * len(FULL_OUTPUT_D2H_SIZES))),
        list(FULL_OUTPUT_D2H_SIZES),
    )
    full_dtoh_counts = dict(cur.fetchall())
    conn.close()
    return summary, full_dtoh_counts


def test_no_full_output_tensor_d2h_and_clean_exit():
    exit_code, sqlite_path = run_worker_under_nsys("hotpath_contract")
    assert exit_code == 0, f"worker exited with {exit_code}"

    summary, full_dtoh_counts = load_memcpy_summary(sqlite_path)

    dtoh_total = summary.get("CUDA_MEMCPY_KIND_DTOH", {}).get("total_bytes", 0)
    htod_total = summary.get("CUDA_MEMCPY_KIND_HTOD", {}).get("total_bytes", 0)

    full_dtoh_total = sum(
        bytes_ * count for bytes_, count in full_dtoh_counts.items()
    )

    assert full_dtoh_total == 0, (
        f"full output tensor D2H detected: {full_dtoh_counts} "
        f"(total dtoh={dtoh_total}, htod={htod_total})"
    )
    assert dtoh_total <= MAX_TOTAL_DTOH_BYTES, (
        f"DTOH budget exceeded: {dtoh_total} bytes (budget {MAX_TOTAL_DTOH_BYTES})"
    )

    # Run sanity check on detection output (host path).
    proc = _run(
        ["python3", str(REPO / "scripts" / "sanity_check_detections.py"),
         str(REPO / "out" / "hotpath_contract_run" / "detections.jsonl")],
        check=False,
    )
    assert proc.returncode == 0, f"detection sanity failed:\n{proc.stdout}\n{proc.stderr}"


if __name__ == "__main__":
    test_no_full_output_tensor_d2h_and_clean_exit()
    print("PASS")
