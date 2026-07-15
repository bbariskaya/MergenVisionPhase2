#!/usr/bin/env python3
"""Batch correctness + tracker continuity + render integrity acceptance matrix."""
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path("/home/user/Workspace/MergenVisionPhase2")
WORKER = "/app/backend/native/build/deepstream_face_worker"
CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
GST_PLUGIN_PATH = "/app/backend/native/build/gst-plugins"
GPU_ID = 0
DEFAULT_VIDEO = REPO / "backend" / "artifacts" / "videos" / "Friends.mp4"
OUT_ROOT = REPO / "backend" / "out" / "correctness_matrix"


def host_to_container(path: Path) -> str:
    return str(Path("/app") / path.relative_to(REPO))


def run_cmd(cmd: list[str], capture: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)


def get_gpu_state() -> dict[str, Any]:
    out = run_cmd([
        "nvidia-smi", "-i", str(GPU_ID),
        "--query-gpu=uuid,memory.used,memory.total,utilization.gpu,utilization.memory,utilization.decoder,utilization.encoder",
        "--format=csv,noheader,nounits",
    ]).stdout.strip()
    parts = [p.strip() for p in out.split(",")]
    return {
        "uuid": parts[0],
        "memory_used_mb": int(parts[1]),
        "memory_total_mb": int(parts[2]),
        "gpu_util": int(parts[3]),
        "mem_util": int(parts[4]),
        "decoder_util": int(parts[5]),
        "encoder_util": int(parts[6]),
    }


def get_gpu_processes() -> list[dict[str, Any]]:
    out = run_cmd([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
        "--format=csv,noheader",
    ]).stdout.strip()
    procs = []
    for line in out.splitlines():
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        procs.append({"pid": parts[0], "name": parts[1], "used_mb": int(parts[2].replace(" MiB", "")), "gpu_uuid": parts[3]})
    return procs


def sampler_thread(gpu_id: int, samples: list[dict[str, Any]], stop: threading.Event) -> None:
    proc = subprocess.Popen(
        [
            "nvidia-smi", "-i", str(gpu_id),
            "--query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory,utilization.decoder,utilization.encoder",
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
            if len(parts) < 6:
                continue
            samples.append({
                "timestamp": parts[0],
                "memory_used_mb": int(parts[1]),
                "gpu_util": int(parts[2]),
                "mem_util": int(parts[3]),
                "decoder_util": int(parts[4]),
                "encoder_util": int(parts[5]),
            })
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    return {
        "peak_memory_mb": max(s["memory_used_mb"] for s in samples),
        "avg_memory_mb": round(statistics.mean(s["memory_used_mb"] for s in samples), 1),
        "peak_gpu_util": max(s["gpu_util"] for s in samples),
        "avg_gpu_util": round(statistics.mean(s["gpu_util"] for s in samples), 1),
        "peak_mem_util": max(s["mem_util"] for s in samples),
        "avg_mem_util": round(statistics.mean(s["mem_util"] for s in samples), 1),
        "peak_decoder_util": max(s["decoder_util"] for s in samples),
        "avg_decoder_util": round(statistics.mean(s["decoder_util"] for s in samples), 1),
        "peak_encoder_util": max(s["encoder_util"] for s in samples),
        "avg_encoder_util": round(statistics.mean(s["encoder_util"] for s in samples), 1),
        "samples": len(samples),
    }


def parse_manifest(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "run_manifest.json"
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


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
    m = re.search(r"tracklets=(\d+)", out)
    if m:
        data["tracklets"] = int(m.group(1))
    return data


def run_experiment(video: Path, batch_size: int, tracker: bool, render: bool, run_idx: int, max_run_sec: int) -> dict[str, Any]:
    tag = f"b{batch_size}_t{'on' if tracker else 'off'}_r{'on' if render else 'off'}_run{run_idx}"
    out_dir = OUT_ROOT / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [
        WORKER,
        host_to_container(video),
        host_to_container(out_dir),
        str(GPU_ID),
        "--batch-size", str(batch_size),
    ]
    if tracker:
        args += ["--tracker", "/app/backend/native/configs/tracker_NvDCF_mergen.yml"]
    else:
        args += ["--tracker", "off"]
    if render:
        args += ["--annotated-output", host_to_container(out_dir / "annotated.mp4")]

    cmd = [
        "docker", "run", "--rm", "--gpus", f"device={GPU_ID}",
        "-e", f"CUDA_VISIBLE_DEVICES={GPU_ID}",
        "-e", f"GST_PLUGIN_PATH={GST_PLUGIN_PATH}",
        "-e", "USE_NEW_NVSTREAMMUX=0",
        "-e", f"MV_BATCH_PUSH_TIMEOUT_US=40000",
        "-v", f"{REPO}:/app",
        "-w", "/app",
        "--entrypoint", "timeout",
        CONTAINER,
        str(max_run_sec),
    ] + args

    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sampler_thread, args=(GPU_ID, samples, stop))
    sampler.start()

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - t0

    stop.set()
    sampler.join(timeout=3)

    out = proc.stdout + proc.stderr
    stdout_data = parse_worker_stdout(out)
    manifest = parse_manifest(out_dir)

    result: dict[str, Any] = {
        "tag": tag,
        "batch_size": batch_size,
        "tracker": tracker,
        "render": render,
        "run_idx": run_idx,
        "container_wall_sec": round(wall, 3),
        "exit_code": proc.returncode,
        "stdout_tail": "\n".join(out.strip().splitlines()[-10:]),
    }
    result.update(stdout_data)
    result.update(manifest)
    result["gpu_telemetry"] = summarize_samples(samples)

    # Render file check.
    annotated = out_dir / "annotated.mp4"
    if annotated.exists():
        result["render_file_bytes"] = annotated.stat().st_size

    # Count rendered frames via ffprobe.
    if render and annotated.exists():
        try:
            fp = run_cmd([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,duration,r_frame_rate,width,height",
                "-of", "default=noprint_wrappers=1", str(annotated),
            ], timeout=30).stdout
            rd: dict[str, Any] = {}
            for line in fp.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    rd[k.strip()] = v.strip()
            result["render_ffprobe"] = rd
        except Exception as e:
            result["render_ffprobe_error"] = str(e)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-run-sec", type=int, default=300)
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    video = args.video.resolve()
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    baseline_state = {
        "gpu": get_gpu_state(),
        "gpu_processes": get_gpu_processes(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if not args.skip_warmup:
        print("Warmup run (batch=1 tracker=off render=off)...", flush=True)
        warmup = run_experiment(video, 1, False, False, -1, args.max_run_sec)
        print(f"  warmup wall={warmup.get('worker_wall_sec'):.3f}s status={'OK' if warmup.get('completed') else 'FAIL'}", flush=True)
        # Pause briefly for GPU to quiesce.
        time.sleep(2)

    results: list[dict[str, Any]] = []
    batch_sizes = args.batch_sizes
    trackers = [False, True]
    renders = [False, True]
    total = len(batch_sizes) * len(trackers) * len(renders) * args.repeats
    idx = 0
    for batch_size in batch_sizes:
        for tracker in trackers:
            for render in renders:
                for r in range(args.repeats):
                    idx += 1
                    print(
                        f"[{idx}/{total}] batch={batch_size} tracker={'on' if tracker else 'off'} "
                        f"render={'on' if render else 'off'} repeat={r}", flush=True,
                    )
                    res = run_experiment(video, batch_size, tracker, render, r, args.max_run_sec)
                    results.append(res)
                    fps = res.get("frames", 0) / res["worker_wall_sec"] if res.get("worker_wall_sec") else 0.0
                    print(
                        f"  -> wall={res.get('worker_wall_sec', -1):.3f}s fps={fps:.1f} "
                        f"avg_batch={res.get('avg_batch', -1):.2f} "
                        f"detections={res.get('detections', -1)} "
                        f"completed={res.get('completed')} exit={res['exit_code']}",
                        flush=True,
                    )
                    # small gap between runs.
                    time.sleep(1)

    end_state = {
        "gpu": get_gpu_state(),
        "gpu_processes": get_gpu_processes(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    summary = {
        "video": str(video),
        "container": CONTAINER,
        "gpu_id": GPU_ID,
        "baseline_state": baseline_state,
        "end_state": end_state,
        "repeats": args.repeats,
        "max_run_sec": args.max_run_sec,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }

    report_path = OUT_ROOT / "correctness_matrix_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRaw report written to: {report_path}", flush=True)

    # Print aggregated median table.
    print("\n=== Median wall time per combination ===")
    print(f"{'batch':>5} {'tracker':>7} {'render':>6} {'median_wall':>11} {'median_fps':>10} {'avg_batch':>9} {'ok_runs':>7}")
    for batch_size in batch_sizes:
        for tracker in trackers:
            for render in renders:
                subset = [r for r in results if r["batch_size"] == batch_size and r["tracker"] == tracker and r["render"] == render]
                walls = [r["worker_wall_sec"] for r in subset if r.get("worker_wall_sec") and r.get("completed")]
                frames = [r.get("frames", 0) for r in subset if r.get("frames")]
                avg_batches = [r.get("avg_batch", 0) for r in subset if r.get("avg_batch")]
                ok = sum(1 for r in subset if r.get("completed") and r.get("worker_error") == 0)
                if walls:
                    med_wall = statistics.median(walls)
                    med_fps = statistics.median([f / w for f, w in zip(frames, walls)])
                    med_avg = statistics.median(avg_batches) if avg_batches else 0
                    print(f"{batch_size:>5} {('on' if tracker else 'off'):>7} {('on' if render else 'off'):>6} {med_wall:>11.3f} {med_fps:>10.1f} {med_avg:>9.2f} {ok:>7}/{len(subset)}")
                else:
                    print(f"{batch_size:>5} {('on' if tracker else 'off'):>7} {('on' if render else 'off'):>6} {'FAIL':>11} {'FAIL':>10} {'n/a':>9} {ok:>7}/{len(subset)}")


if __name__ == "__main__":
    main()
