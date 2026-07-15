"""Run video face-detection through the native worker port.

This service knows nothing about Docker, GStreamer, CUDA or subprocesses.  It
only touches the domain model and the injected :class:`NativeWorkerPort`.
"""
from __future__ import annotations

from app.domain.native_job import (
    NativeJobError,
    NativeJobRequest,
    NativeJobResult,
)
from app.ports.native_worker import NativeWorkerPort


class RunVideoDetectionService:
    """Application service that dispatches a video job to the native worker."""

    def __init__(self, worker: NativeWorkerPort) -> None:
        self.worker = worker

    async def execute(
        self, request: NativeJobRequest
    ) -> NativeJobResult | NativeJobError:
        """Process the video and return a typed result or error."""
        return await self.worker.process_video(request)
