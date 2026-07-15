#!/usr/bin/env python3
"""Sweep nvstreammux buffer-pool-size for render path.

Finds the smallest pool value that keeps avg_batch, throughput, and clean EOS
across different batch sizes. Does not rerun if --from-report is given.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path("/home/user/Workspace/MergenVisionPhase2")
WORKER = "/app/backend/native/build/deepstream_face_worker"
CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
GST_PLUGIN_PATH = "/app/backend/native/build/gst-plugins"
GPU_ID = 0
DEFAULT_VIDEO = REPO / "backend" / "artifacts" / "videos" / "Friends.mp4"
OUT_ROOT = REPO / "backend" / "out" / "mux_pool_sweep"


def host_to_container(path: Path) -> str:
    return str(Path("/app") / path.relative_to(REPO))


def run_cmd(cmd: list[str], capture: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)


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


def count_video_frames(path: Path) -> int | None:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    try:
        out = run_cmd(cmd, timeout=30).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


def run_experiment(video: Path, batch_size: int, pool_size: int, repeat: int, max_run_sec: int) -> dict[str, Any]:
    tag = f"b{batch_size}_pool{pool_size}_r{repeat}"
    out_dir = OUT_ROOT / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = out_dir / "annotated.mp4"
    cmd = [
        "docker", "run", "--rm", "--gpus", f"device={GPU_ID}",
        "-e", f"CUDA_VISIBLE_DEVICES={GPU_ID}",
        "-e", f"GST_PLUGIN_PATH={GST_PLUGIN_PATH}",
        "-e", "USE_NEW_NVSTREAMMUX=0",
        "-e", f"MV_MUX_POOL_SIZE={pool_size}",
        "-v", f"{REPO}:/app",
        "-w", "/app",
        "--entrypoint", "timeout",
        CONTAINER,
        str(max_run_sec),
        WORKER,
        host_to_container(video),
        host_to_container(out_dir),
        str(GPU_ID),
        "--batch-size", str(batch_size),
        "--tracker", "off",
        "--render",
        "--annotated-output", host_to_container(annotated),
    ]

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - t0
    out = proc.stdout + proc.stderr
    stdout_data = parse_worker_stdout(out)

    source_frames = count_video_frames(video)
    render_frames = count_video_frames(annotated) if annotated.exists() else None

    result: dict[str, Any] = {
        "tag": tag,
        "batch_size": batch_size,
        "pool_size": pool_size,
        "repeat": repeat,
        "container_wall_sec": round(wall, 3),
        "exit_code": proc.returncode,
    }
    result.update(stdout_data)
    result["source_frames"] = source_frames
    result["render_frames"] = render_frames
    result["frame_count_match"] = (source_frames == render_frames) if source_frames and render_frames else False
    result["stdout_tail"] = "\n".join(out.strip().splitlines()[-8:])
    return result


def print_table(results: list[dict[str, Any]]) -> None:
    print("\n=== Mux pool sweep results ===")
    print(f"{'pool':>5} {'batch':>5} {'wall':>8} {'fps':>7} {'avg_batch':>9} {'ok':>4} {'match':>6}")
    for r in sorted(results, key=lambda x: (x["pool_size"], x["batch_size"])):
        fps = (r.get("frames", 0) / r["worker_wall_sec"]) if r.get("worker_wall_sec") else 0.0
        ok = "OK" if r.get("completed") and r.get("worker_error") == 0 and r.get("frame_count_match") else "FAIL"
        print(f"{r['pool_size']:>5} {r['batch_size']:>5} {r.get('worker_wall_sec', -1):>8.3f} {fps:>7.1f} {r.get('avg_batch', -1):>9.2f} {ok:>4} {str(r.get('frame_count_match')):>6}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-run-sec", type=int, default=300)
    parser.add_argument("--from-report", type=Path, default=None)
    args = parser.parse_args()

    if args.from_report:
        with open(args.from_report) as f:
            report = json.load(f)
        print_table(report.get("results", []))
        return 0

    video = args.video.resolve()
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    total = len(args.batch_sizes) * len(args.pool_sizes) * args.repeats
    idx = 0
    for batch_size in args.batch_sizes:
        for pool_size in args.pool_sizes:
            for r in range(args.repeats):
                idx += 1
                print(f"[{idx}/{total}] batch={batch_size} pool={pool_size} repeat={r}", flush=True)
                res = run_experiment(video, batch_size, pool_size, r, args.max_run_sec)
                results.append(res)
                fps = (res.get("frames", 0) / res["worker_wall_sec"]) if res.get("worker_wall_sec") else 0.0
                print(
                    f"  -> wall={res.get('worker_wall_sec', -1):.3f}s fps={fps:.1f} "
                    f"avg_batch={res.get('avg_batch', -1):.2f} completed={res.get('completed')} "
                    f"exit={res['exit_code']} match={res.get('frame_count_match')}",
                    flush=True,
                )
                time.sleep(1)

    report = {
        "video": str(video),
        "container": CONTAINER,
        "gpu_id": GPU_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    report_path = OUT_ROOT / "mux_pool_sweep_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {report_path}")
    print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
