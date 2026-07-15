"""Domain model validation tests."""
from __future__ import annotations

import pytest
from app.domain.native_job import (
    JobStatus,
    NativeJobError,
    NativeJobErrorCode,
    NativeJobRequest,
    NativeJobResult,
)


def test_job_status_values() -> None:
    assert JobStatus.COMPLETED.value == "completed"
    assert JobStatus.FAILED.value == "failed"


def test_native_job_request_has_no_filesystem_side_effects(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    out = tmp_path / "out" / "nested"
    req = NativeJobRequest(job_id="j1", video_path=video, output_dir=out)
    assert req.job_id == "j1"
    assert not video.exists()
    assert not out.exists()


def test_native_job_request_accepts_missing_video(tmp_path) -> None:
    """Constructor validation is deferred to the infrastructure preflight."""
    req = NativeJobRequest(
        job_id="j1",
        video_path=tmp_path / "missing.mp4",
        output_dir=tmp_path / "out",
    )
    assert req.video_path == tmp_path / "missing.mp4"


def test_native_job_result_defaults() -> None:
    result = NativeJobResult(
        job_id="j1",
        completed=True,
        decoded_frames=10,
        processed_frames=10,
        detections=3,
    )
    assert result.exit_code == 0
    assert result.metadata == {}


def test_native_job_error_typed() -> None:
    err = NativeJobError(
        job_id="j1",
        code=NativeJobErrorCode.TIMEOUT,
        message="too slow",
    )
    assert err.code == NativeJobErrorCode.TIMEOUT
    assert "too slow" in err.message
