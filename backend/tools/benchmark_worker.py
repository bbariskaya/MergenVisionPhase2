#!/usr/bin/env python3
"""Benchmark deepstream_face_worker across batch/tracker/render combinations."""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

REPO = Path("/home/user/Workspace/MergenVisionPhase2")
OUT_BASE = REPO / "backend" / "out" / "benchmark_worker"
DEFAULT_VIDEO = REPO / "backend" / "artifacts" / "videos" / "friendsshort_50f.mp4"
WORKER = "/app/backend/native/build/deepstream_face_worker"
CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
GST_PLUGIN_PATH = "/app/backend/native/build/gst-plugins"
GPU_ID = "0"
TIMEOUT_US = os.environ.get("MV_BENCH_TIMEOUT_US", "40000")


def to_container_path(host_path: Path) -> str:
    rel = host_path.relative_to(REPO)
    return str(Path("/app") / rel)


def run_worker(video: Path, batch_size: int, tracker: bool, render: bool, run_idx: int, max_run_sec: int) -> dict[str, Any]:
    tag = f"b{batch_size}_t{'on' if tracker else 'off'}_r{'on' if render else 'off'}_run{run_idx}"
    out_dir = OUT_BASE / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_container = to_container_path(video)
    out_container = to_container_path(out_dir)
    annotated_container = str(Path(out_container) / "annotated.mp4")

    args = [
        WORKER,
        video_container,
        out_container,
        GPU_ID,
        "--batch-size", str(batch_size),
    ]
    if not tracker:
        args += ["--tracker", "off"]
    else:
        args += ["--tracker", "/app/backend/native/configs/tracker_NvDCF_mergen.yml"]
    if render:
        args += ["--annotated-output", annotated_container]

    cmd = [
        "docker", "run", "--rm", "--gpus", f"device={GPU_ID}",
        "-e", f"CUDA_VISIBLE_DEVICES={GPU_ID}",
        "-e", f"GST_PLUGIN_PATH={GST_PLUGIN_PATH}",
        "-e", "USE_NEW_NVSTREAMMUX=0",
        "-e", f"MV_BATCH_PUSH_TIMEOUT_US={TIMEOUT_US}",
        "-v", f"{REPO}:/app",
        "-w", "/app",
        "--entrypoint", "timeout",
        CONTAINER,
        str(max_run_sec),
    ] + args

    start = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - start

    out = proc.stdout + proc.stderr
    result: dict[str, Any] = {
        "tag": tag,
        "batch_size": batch_size,
        "tracker": tracker,
        "render": render,
        "run_idx": run_idx,
        "container_wall_sec": round(wall, 3),
        "exit_code": proc.returncode,
        "raw_tail": "\n".join(out.strip().splitlines()[-15:]),
    }

    m = re.search(r"streammux summary: buffers=(\d+) frames_total=(\d+) avg_batch=([\d.]+)", out)
    if m:
        result["streammux_buffers"] = int(m.group(1))
        result["streammux_frames_total"] = int(m.group(2))
        result["avg_batch"] = float(m.group(3))

    m = re.search(
        r"Done\. frames=(\d+) detections=(\d+) enqueue=(\d+) wall=([\d.]+)s error=(\d+)", out
    )
    if m:
        result["frames"] = int(m.group(1))
        result["detections"] = int(m.group(2))
        result["enqueue_count"] = int(m.group(3))
        result["worker_wall_sec"] = float(m.group(4))
        result["worker_error"] = int(m.group(5))

    m = re.search(r"tracklets=(\d+)", out)
    if m:
        result["tracklets"] = int(m.group(1))

    m = re.search(r"completed=(\w+)", out)
    if m:
        result["completed"] = m.group(1) == "true"

    annotated = out_dir / "annotated.mp4"
    if annotated.exists():
        result["render_file_bytes"] = annotated.stat().st_size
        result["render_file_mb"] = round(annotated.stat().st_size / (1024 * 1024), 2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark deepstream_face_worker")
    parser.add_argument("video", type=Path, nargs="?", default=DEFAULT_VIDEO, help="input video path")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--timeouts", type=int, nargs="+", default=[int(TIMEOUT_US)])
    parser.add_argument("--tracker", action="store_true", help="include tracker-on combinations")
    parser.add_argument("--render", action="store_true", help="include render-on combinations")
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("MV_BENCH_REPEATS", "1")))
    parser.add_argument("--max-run-sec", type=int, default=60)
    args = parser.parse_args()

    video: Path = args.video.resolve()

    if OUT_BASE.exists():
        shutil.rmtree(OUT_BASE)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    batch_sizes = args.batch_sizes
    timeouts = args.timeouts
    trackers = [False]
    if args.tracker:
        trackers.append(True)
    renders = [False]
    if args.render:
        renders.append(True)

    results: list[dict[str, Any]] = []
    total = len(batch_sizes) * len(trackers) * len(renders) * len(timeouts) * args.repeats
    idx = 0
    for batch_size in batch_sizes:
        for tracker in trackers:
            for render in renders:
                for timeout_us in timeouts:
                    for run_idx in range(args.repeats):
                        idx += 1
                        print(
                            f"[{idx}/{total}] batch={batch_size} tracker={'on' if tracker else 'off'} "
                            f"render={'on' if render else 'off'} timeout={timeout_us} run={run_idx}",
                            flush=True,
                        )
                        res = run_worker(
                            video, batch_size, tracker, render, run_idx,
                            max_run_sec=args.max_run_sec,
                        )
                        res["timeout_us"] = timeout_us
                        results.append(res)
                        fps = "n/a"
                        if res.get("worker_wall_sec", 0) > 0 and res.get("frames") is not None:
                            fps = f"{res['frames'] / res['worker_wall_sec']:.1f}"
                        status = "OK" if res.get("completed") and res.get("worker_error") == 0 else f"FAIL({res.get('exit_code')})"
                        print(
                            f"  -> batch={res['batch_size']} tracker={'on' if tracker else 'off'} "
                            f"render={'on' if render else 'off'} timeout={timeout_us}: "
                            f"wall={res.get('worker_wall_sec', -1):.3f}s fps={fps} "
                            f"avg_batch={res.get('avg_batch', -1):.2f} enqueue={res.get('enqueue_count', -1)} "
                            f"status={status}",
                            flush=True,
                        )

    summary = {
        "video": str(video),
        "container": CONTAINER,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }

    report_path = OUT_BASE / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReport written to: {report_path}")

    # Print a concise table of completed runs.
    print("\n=== Summary ===")
    print(f"{'batch':>5} {'tracker':>7} {'render':>6} {'wall(s)':>8} {'enqueue':>7} {'avg_batch':>9} {'status':>10}")
    for r in results:
        status = "OK" if r.get("completed") and r.get("worker_error") == 0 else f"FAIL({r.get('exit_code')})"
        print(
            f"{r['batch_size']:>5} {('on' if r['tracker'] else 'off'):>7} "
            f"{('on' if r['render'] else 'off'):>6} "
            f"{r.get('worker_wall_sec', -1):>8.3f} {r.get('enqueue_count', -1):>7} "
            f"{r.get('avg_batch', -1):>9.2f} {status:>10}"
        )


if __name__ == "__main__":
    main()
