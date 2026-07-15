"""Domain models for a native video-detection job.

These models are intentionally free of infrastructure concerns (subprocess,
Docker, storage). They describe the contract between the API/application layer
and the native worker port.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NativeJobErrorCode(str, Enum):
    WORKER_FAILED = "worker_failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PROTOCOL_ERROR = "protocol_error"
    INPUT_NOT_FOUND = "input_not_found"


@dataclass(frozen=True, kw_only=True)
class NativeJobRequest:
    """Request to process one video through the native worker."""

    job_id: str
    video_path: Path
    output_dir: Path
    gpu_device: int = 0
    tracker_config: Path | None = None

    def __post_init__(self) -> None:
        # Intentionally no filesystem side effects here. Input existence and
        # output directory creation belong to the infrastructure preflight.
        pass


@dataclass(frozen=True, kw_only=True)
class NativeJobProgress:
    """Structured progress event emitted by the native worker."""

    job_id: str
    stage: str
    decoded_frames: int = 0
    processed_frames: int = 0
    detections: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, kw_only=True)
class NativeJobResult:
    """Final result of a successfully completed native worker job."""

    job_id: str
    completed: bool
    decoded_frames: int
    processed_frames: int
    detections: int
    detections_path: Path | None = None
    exit_code: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class NativeJobError:
    """Typed error returned when a native worker job fails."""

    job_id: str
    code: NativeJobErrorCode
    message: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
