"""Native worker port — the boundary between application and infrastructure.

The concrete adapter lives in :mod:`app.infrastructure.native_worker`.
It must never leak Docker, subprocess, or GPU details into application code.
"""
from __future__ import annotations

from typing import Protocol

from app.domain.native_job import NativeJobError, NativeJobRequest, NativeJobResult


class NativeWorkerPort(Protocol):
    """Process a video through the native GPU worker.

    Implementations are responsible for:
    - validating that the input exists,
    - invoking the worker exactly once per request,
    - parsing structured events emitted by the worker,
    - mapping nonzero exits, protocol errors and timeouts to typed errors.
    """

    async def process_video(self, request: NativeJobRequest) -> NativeJobResult | NativeJobError:
        ...
