"""Subprocess adapter tests with shell stand-ins for the native worker.

No Docker/GPU is exercised here; we replace the Docker command builder with
plain shell scripts and inspect the adapter's parsed result.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.domain.native_job import (
    NativeJobError,
    NativeJobErrorCode,
    NativeJobRequest,
    NativeJobResult,
)
from app.infrastructure.native_worker.subprocess_adapter import (
    SubprocessNativeWorkerAdapter,
)


class _FakeClient:
    """Returns an arbitrary argv list instead of a docker command."""

    def __init__(self, *argv: str) -> None:
        self.argv = list(argv)

    def run_command(
        self, input_video_path, output_dir, *, tracker_config=None
    ) -> list[str]:
        return list(self.argv)


def _make_request(tmp_path, job_id: str = "j1") -> NativeJobRequest:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    out = tmp_path / "out"
    return NativeJobRequest(job_id=job_id, video_path=video, output_dir=out)


def test_success_event_parsed(tmp_path):
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=10)
    summary = "completed=true decoded_frames=50 processed_frames=50 detections=25 tracklets=1 eos_clean=true exit_code=0"
    adapter.client = _FakeClient("bash", "-c", f"echo '{summary}'")

    result = asyncio.run(adapter.process_video(_make_request(tmp_path)))

    assert isinstance(result, NativeJobResult)
    assert result.completed is True
    assert result.decoded_frames == 50
    assert result.processed_frames == 50


def test_nonzero_exit_returns_typed_error(tmp_path):
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=10)
    adapter.client = _FakeClient("bash", "-c", "echo 'completed=true'; exit 7")

    result = asyncio.run(adapter.process_video(_make_request(tmp_path)))

    assert isinstance(result, NativeJobError)
    assert result.code == NativeJobErrorCode.WORKER_FAILED
    assert result.exit_code == 7


def test_malformed_event_returns_protocol_error(tmp_path):
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=10)
    adapter.client = _FakeClient("bash", "-c", "echo 'no summary here'")

    result = asyncio.run(adapter.process_video(_make_request(tmp_path)))

    assert isinstance(result, NativeJobError)
    assert result.code == NativeJobErrorCode.PROTOCOL_ERROR


def test_timeout_returns_typed_error(tmp_path):
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=0.1)
    adapter.client = _FakeClient("bash", "-c", "sleep 5")

    result = asyncio.run(adapter.process_video(_make_request(tmp_path)))

    assert isinstance(result, NativeJobError)
    assert result.code == NativeJobErrorCode.TIMEOUT


def test_cancellation_returns_typed_error(tmp_path):
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=10)
    adapter.client = _FakeClient("bash", "-c", "sleep 5")

    async def run_and_cancel() -> None:
        task = asyncio.create_task(adapter.process_video(_make_request(tmp_path)))
        await asyncio.sleep(0.05)
        task.cancel()
        return await task

    result = asyncio.run(run_and_cancel())

    assert isinstance(result, NativeJobError)
    assert result.code == NativeJobErrorCode.CANCELLED
