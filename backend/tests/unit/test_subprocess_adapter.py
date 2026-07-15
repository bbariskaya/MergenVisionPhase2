"""Subprocess adapter tests with shell stand-ins for the native worker.

No Docker/GPU is exercised here; we replace the Docker command builder with
plain shell scripts and inspect the adapter's parsed result.
"""
from __future__ import annotations

import asyncio
import os
import warnings
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
        self,
        input_video_path,
        output_dir,
        *,
        tracker_config=None,
        gpu_device=None,
    ) -> list[str]:
        return list(self.argv)


class _RecordingFakeClient:
    """Records the kwargs passed to run_command."""

    def __init__(self, *argv: str) -> None:
        self.argv = list(argv)
        self.calls: list[dict] = []

    def run_command(
        self,
        input_video_path,
        output_dir,
        *,
        tracker_config=None,
        gpu_device=None,
    ) -> list[str]:
        self.calls.append(
            {
                "input": input_video_path,
                "output": output_dir,
                "tracker_config": tracker_config,
                "gpu_device": gpu_device,
            }
        )
        return list(self.argv)


def _make_request(tmp_path, job_id: str = "j1", **kwargs) -> NativeJobRequest:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    out = tmp_path / "out"
    return NativeJobRequest(job_id=job_id, video_path=video, output_dir=out, **kwargs)


def test_adapter_accepts_gpu_device_kwarg() -> None:
    """CLI passes gpu_device to the adapter constructor."""
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", gpu_device=1)
    assert adapter.client.gpu_device == 1


def test_gpu_device_from_request_is_passed_to_command_builder(tmp_path) -> None:
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=10)
    recording = _RecordingFakeClient("bash", "-c", "echo 'completed=true'")
    adapter.client = recording

    result = asyncio.run(
        adapter.process_video(_make_request(tmp_path, gpu_device=2))
    )

    assert isinstance(result, NativeJobResult)
    assert len(recording.calls) == 1
    assert recording.calls[0]["gpu_device"] == 2


def test_success_event_parsed(tmp_path) -> None:
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=10)
    summary = (
        "completed=true decoded_frames=50 processed_frames=50 "
        "detections=25 tracklets=1 eos_clean=true exit_code=0"
    )
    adapter.client = _FakeClient("bash", "-c", f"echo '{summary}'")
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "detections.jsonl").write_text(
        '{"frame": 0, "detections": []}\n'
    )

    result = asyncio.run(adapter.process_video(_make_request(tmp_path)))

    assert isinstance(result, NativeJobResult)
    assert result.completed is True
    assert result.decoded_frames == 50
    assert result.processed_frames == 50


def test_nonzero_exit_returns_typed_error(tmp_path) -> None:
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=10)
    adapter.client = _FakeClient("bash", "-c", "echo 'completed=true'; exit 7")

    result = asyncio.run(adapter.process_video(_make_request(tmp_path)))

    assert isinstance(result, NativeJobError)
    assert result.code == NativeJobErrorCode.WORKER_FAILED
    assert result.exit_code == 7
    assert "stderr secret" not in result.message.lower()


def test_completed_false_returns_typed_error(tmp_path) -> None:
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=10)
    adapter.client = _FakeClient(
        "bash", "-c", "echo 'completed=false exit_code=0'"
    )

    result = asyncio.run(adapter.process_video(_make_request(tmp_path)))

    assert isinstance(result, NativeJobError)
    assert result.code == NativeJobErrorCode.WORKER_FAILED
    # Raw worker output must stay out of the client-facing message.
    assert "completed=false" not in result.message


def test_summary_exit_code_mismatch_returns_protocol_error(tmp_path) -> None:
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=10)
    adapter.client = _FakeClient(
        "bash", "-c", "echo 'completed=true exit_code=0'; exit 5"
    )

    result = asyncio.run(adapter.process_video(_make_request(tmp_path)))

    assert isinstance(result, NativeJobError)
    assert result.code == NativeJobErrorCode.PROTOCOL_ERROR


def test_missing_detections_file_returns_protocol_error(tmp_path) -> None:
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=10)
    adapter.client = _FakeClient(
        "bash",
        "-c",
        "echo 'completed=true decoded_frames=1 processed_frames=1 detections=1 exit_code=0'",
    )

    result = asyncio.run(adapter.process_video(_make_request(tmp_path)))

    assert isinstance(result, NativeJobError)
    assert result.code == NativeJobErrorCode.PROTOCOL_ERROR


def test_malformed_event_returns_protocol_error(tmp_path) -> None:
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=10)
    adapter.client = _FakeClient("bash", "-c", "echo 'no summary here'")

    result = asyncio.run(adapter.process_video(_make_request(tmp_path)))

    assert isinstance(result, NativeJobError)
    assert result.code == NativeJobErrorCode.PROTOCOL_ERROR


def test_timeout_returns_typed_error(tmp_path) -> None:
    adapter = SubprocessNativeWorkerAdapter(repo_root=".", timeout_seconds=0.1)
    adapter.client = _FakeClient("bash", "-c", "sleep 5")

    result = asyncio.run(adapter.process_video(_make_request(tmp_path)))

    assert isinstance(result, NativeJobError)
    assert result.code == NativeJobErrorCode.TIMEOUT


def test_cancellation_returns_typed_error(tmp_path) -> None:
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


def test_no_resource_warning_under_pytest_w_error(tmp_path) -> None:
    """Confirm stdout/stderr transports are closed under -W error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        adapter = SubprocessNativeWorkerAdapter(
            repo_root=".", timeout_seconds=10
        )
        summary = (
            "completed=true decoded_frames=1 processed_frames=1 "
            "detections=0 exit_code=0"
        )
        adapter.client = _FakeClient("bash", "-c", f"echo '{summary}'")

        result = asyncio.run(adapter.process_video(_make_request(tmp_path)))

        assert isinstance(result, NativeJobResult)
