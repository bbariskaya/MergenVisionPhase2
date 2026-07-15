"""Thin CLI to exercise the backend application → port → adapter chain.

Run from the ``backend/`` directory:

    python -m app.cli detect \
        --video ../artifacts/videos/friendsshort_50f.mp4 \
        --output ../backend/out/sprint-02-cli \
        --host-gpu 0

The CLI performs no GPU compute; it delegates the entire job to the native
worker and prints the structured result.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.application.services.run_video_detection import RunVideoDetectionService
from app.domain.native_job import NativeJobError, NativeJobRequest
from app.infrastructure.native_worker.subprocess_adapter import (
    SubprocessNativeWorkerAdapter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def detect(video: Path, output: Path, host_gpu: int) -> int:
    adapter = SubprocessNativeWorkerAdapter(
        repo_root=REPO_ROOT,
        gpu_device=host_gpu,
    )
    service = RunVideoDetectionService(adapter)
    request = NativeJobRequest(
        job_id=f"cli-{video.stem}",
        video_path=video,
        output_dir=output,
        gpu_device=host_gpu,
    )

    print(f"Submitting job {request.job_id} to native worker ...")
    result = await service.execute(request)

    if isinstance(result, NativeJobError):
        print(f"FAILED: {result.code.value} - {result.message}", file=sys.stderr)
        if result.stdout:
            print("--- worker stdout ---", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print("--- worker stderr ---", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        return 1

    print("Job completed.")
    print(f"  decoded_frames: {result.decoded_frames}")
    print(f"  processed_frames: {result.processed_frames}")
    print(f"  detections: {result.detections}")
    print(f"  output_dir: {result.detections_path.parent if result.detections_path else result.detections_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MergenVision backend CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="run native face detection on a video")
    detect_parser.add_argument("--video", required=True, type=Path, help="input video path")
    detect_parser.add_argument("--output", required=True, type=Path, help="output directory")
    detect_parser.add_argument("--host-gpu", type=int, default=0, help="host GPU device index")

    args = parser.parse_args(argv)
    if args.command == "detect":
        return asyncio.run(detect(args.video, args.output, args.host_gpu))
    return 0


if __name__ == "__main__":
    sys.exit(main())
