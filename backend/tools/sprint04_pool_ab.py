#!/usr/bin/env python3
"""Targeted A/B test for nvstreammux render buffer-pool size.

Compare pool=16 vs pool=128 for:
  Friends.mp4, batch=8, tracker=off, render=on

Keep the smallest pool that does not regress throughput by >5%.
"""
from __future__ import annotations

import json
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path("/home/user/Workspace/MergenVisionPhase2")
WORKER = "/app/backend/native/build/deepstream_face_worker"
CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
GST_PLUGIN_PATH = "/app/backend/native/build/gst-plugins"
VIDEO = REPO / "backend" / "artifacts" / "videos" / "Friends.mp4"
GPU_ID = 0
OUT_ROOT = REPO / "backend" / "out" / "sprint04_pool_ab"


def host_to_container(path: Path) -> str:
    return str(Path("/app") / path.relative_to(REPO))


def run_cmd(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def sampler_thread(gpu_id: int, samples: list[dict[str, Any]], stop: threading.Event) -> None:
    proc = subprocess.Popen(
        [
            "nvidia-smi", "-i", str(gpu_id),
            "--query-gpu=timestamp,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits", "-l", "1",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        while not stop.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 3:
                continue
            samples.append({
                "timestamp": parts[0],
                "memory_used_mb": int(parts[1]),
                "gpu_util": int(parts[2]),
            })
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def parse_worker_stdout(out: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    m = re.search(r"Done\. frames=(\d+) detections=(\d+) enqueue=(\d+) wall=([\d.]+)s error=(\d+)", out)
    if m:
        data["frames"] = int(m.group(1))
        data["detections"] = int(m.group(2))
        data["enqueue_count"] = int(m.group(3))
        data["worker_wall_sec"] = float(m.group(4))
        data["worker_error"] = int(m.group(5))
    m = re.search(r"streammux summary: buffers=(\d+) frames_total=(\d+) avg_batch=([\d.]+)", out)
    if m:
        data["streammux_buffers"] = int(m.group(1))
        data["streammux_frames_total"] = int(m.group(2))
        data["avg_batch"] = float(m.group(3))
    m = re.search(r"completed=(\w+)", out)
    if m:
        data["completed"] = m.group(1) == "true"
    return data


def run_one(pool_size: int) -> dict[str, Any]:
    run_name = f"pool_{pool_size}"
    out_dir = OUT_ROOT / run_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", f"device={GPU_ID}",
        "-e", f"CUDA_VISIBLE_DEVICES={GPU_ID}",
        "-e", f"GST_PLUGIN_PATH={GST_PLUGIN_PATH}",
        "-e", "USE_NEW_NVSTREAMMUX=0",
        "-e", f"MV_MUX_POOL_SIZE={pool_size}",
        "-v", f"{REPO}:/app",
        "-w", "/app",
        "--entrypoint", WORKER,
        CONTAINER,
        host_to_container(VIDEO),
        host_to_container(out_dir),
        str(GPU_ID),
        "--batch-size", "8",
        "--tracker", "off",
        "--annotated-output", host_to_container(out_dir / "annotated.mp4"),
    ]

    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sampler_thread, args=(GPU_ID, samples, stop))
    sampler.start()

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    wall = time.monotonic() - t0

    stop.set()
    sampler.join(timeout=3)

    out = proc.stdout + proc.stderr
    data = parse_worker_stdout(out)
    data.update({
        "pool_size": pool_size,
        "exit_code": proc.returncode,
        "container_wall_sec": round(wall, 3),
        "peak_memory_mb": max((s["memory_used_mb"] for s in samples), default=0),
        "avg_gpu_util": round(statistics.mean(s["gpu_util"] for s in samples), 1) if samples else 0,
        "stdout_tail": "\n".join(out.strip().splitlines()[-8:]),
    })

    annotated = out_dir / "annotated.mp4"
    if annotated.exists():
        data["annotated_bytes"] = annotated.stat().st_size
        fp = run_cmd([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=nb_read_packets,duration,r_frame_rate",
            "-of", "default=noprint_wrappers=1", str(annotated),
        ], timeout=30).stdout
        ff: dict[str, Any] = {}
        for line in fp.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                ff[k.strip()] = v.strip()
        data["ffprobe"] = ff
    return data


def main() -> int:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=== Pool A/B: batch=8 tracker=off render=on ===", flush=True)
    a = run_one(16)
    time.sleep(3)
    b = run_one(128)

    report = {"pool_16": a, "pool_128": b}
    with open(OUT_ROOT / "pool_ab_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Comparison ===")
    print(f"{'metric':<25} {'pool=16':>16} {'pool=128':>16}")
    for k in ["exit_code", "completed", "frames", "avg_batch", "worker_wall_sec", "container_wall_sec", "peak_memory_mb", "annotated_bytes"]:
        print(f"{k:<25} {str(a.get(k)):>16} {str(b.get(k)):>16}")

    if not a.get("completed") or not b.get("completed"):
        print("\nResult: REJECT lower pool — one config failed", file=sys.stderr)
        return 1

    fps_a = a["frames"] / a["worker_wall_sec"]
    fps_b = b["frames"] / b["worker_wall_sec"]
    regression = (fps_b - fps_a) / fps_b if fps_b else 0.0
    print(f"\nfps_16={fps_a:.1f} fps_128={fps_b:.1f} regression={regression*100:.2f}%")

    if regression > 0.05:
        print("Verdict: keep pool=128 (>5% throughput regression with pool=16)", flush=True)
    else:
        print("Verdict: keep pool=16 (within 5% of pool=128)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
