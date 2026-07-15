"""Application service tests using a mock native worker port."""
from __future__ import annotations

import asyncio

import pytest

from app.application.services.run_video_detection import RunVideoDetectionService
from app.domain.native_job import (
    NativeJobError,
    NativeJobErrorCode,
    NativeJobRequest,
    NativeJobResult,
)


class FakeSuccessWorker:
    async def process_video(
        self, request: NativeJobRequest
    ) -> NativeJobResult | NativeJobError:
        return NativeJobResult(
            job_id=request.job_id,
            completed=True,
            decoded_frames=50,
            processed_frames=50,
            detections=25,
        )


class FakeFailingWorker:
    async def process_video(
        self, request: NativeJobRequest
    ) -> NativeJobResult | NativeJobError:
        return NativeJobError(
            job_id=request.job_id,
            code=NativeJobErrorCode.WORKER_FAILED,
            message="boom",
        )


def test_service_delegates_to_port(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    out = tmp_path / "out"
    request = NativeJobRequest(job_id="j1", video_path=video, output_dir=out)
    service = RunVideoDetectionService(FakeSuccessWorker())

    result = asyncio.run(service.execute(request))

    assert isinstance(result, NativeJobResult)
    assert result.job_id == "j1"
    assert result.detections == 25


def test_service_returns_typed_error(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    request = NativeJobRequest(job_id="j2", video_path=video, output_dir=tmp_path / "out")
    service = RunVideoDetectionService(FakeFailingWorker())

    result = asyncio.run(service.execute(request))

    assert isinstance(result, NativeJobError)
    assert result.code == NativeJobErrorCode.WORKER_FAILED
