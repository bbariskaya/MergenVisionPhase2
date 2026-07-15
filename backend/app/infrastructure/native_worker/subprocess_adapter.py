"""Concrete native-worker adapter using Docker/subprocess.

This module imports no domain/application code; it implements the
:py:class:`app.ports.native_worker.NativeWorkerPort` protocol.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from app.domain.native_job import (
    NativeJobError,
    NativeJobErrorCode,
    NativeJobProgress,
    NativeJobRequest,
    NativeJobResult,
)
from app.infrastructure.native_worker.client import (
    DEFAULT_CONTAINER,
    DEFAULT_GST_PLUGIN_PATH,
    DEFAULT_WORKER,
    NativeDetectorClient,
)

logger = logging.getLogger(__name__)


class SubprocessNativeWorkerAdapter:
    """Run the native worker inside Docker once per job and parse its events.

    The worker is expected to emit structured text/json events on stdout and a
    final summary line of the form ``completed=true decoded_frames=...``.  All
    stderr is forwarded to the configured logger and never exposed to clients.
    """

    def __init__(
        self,
        repo_root: Path | str,
        *,
        container: str | None = None,
        worker_path: str | None = None,
        gst_plugin_path: str | None = None,
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.client = NativeDetectorClient(
            repo_root,
            container=container or DEFAULT_CONTAINER,
            worker_path=worker_path or DEFAULT_WORKER,
            gst_plugin_path=gst_plugin_path or DEFAULT_GST_PLUGIN_PATH,
        )
        self.timeout_seconds = timeout_seconds

    async def process_video(
        self, request: NativeJobRequest
    ) -> NativeJobResult | NativeJobError:
        cmd = self.client.run_command(
            request.video_path,
            request.output_dir,
            tracker_config=request.tracker_config,
        )
        logger.info("starting native job %s: %s", request.job_id, " ".join(cmd))

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            stdout_task = asyncio.create_task(
                self._read_stdout(proc.stdout, request.job_id, stdout_lines)
            )
            stderr_task = asyncio.create_task(
                self._read_stderr(proc.stderr, stderr_lines)
            )

            try:
                await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task, proc.wait()),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                self._terminate(proc)
                return NativeJobError(
                    job_id=request.job_id,
                    code=NativeJobErrorCode.TIMEOUT,
                    message=f"native worker timed out after {self.timeout_seconds}s",
                    stderr="\n".join(stderr_lines),
                )
            except asyncio.CancelledError:
                self._terminate(proc)
                return NativeJobError(
                    job_id=request.job_id,
                    code=NativeJobErrorCode.CANCELLED,
                    message="native worker cancelled",
                    stderr="\n".join(stderr_lines),
                )

            if proc.returncode != 0:
                return NativeJobError(
                    job_id=request.job_id,
                    code=NativeJobErrorCode.WORKER_FAILED,
                    message=f"native worker exited with code {proc.returncode}",
                    stdout="\n".join(stdout_lines),
                    stderr="\n".join(stderr_lines),
                    exit_code=proc.returncode,
                )

            return self._parse_result(request, stdout_lines, stderr_lines)
        except FileNotFoundError as exc:
            return NativeJobError(
                job_id=request.job_id,
                code=NativeJobErrorCode.INPUT_NOT_FOUND,
                message=str(exc),
            )
        except Exception as exc:  # pragma: no cover - defensive
            return NativeJobError(
                job_id=request.job_id,
                code=NativeJobErrorCode.WORKER_FAILED,
                message=f"unexpected adapter error: {exc}",
            )
        finally:
            if proc is not None and proc.returncode is None:
                self._terminate(proc)

    async def _read_stdout(
        self, stream: asyncio.StreamReader | None, job_id: str, lines: list[str]
    ) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            lines.append(text)
            self._handle_event(text, job_id)

    async def _read_stderr(
        self, stream: asyncio.StreamReader | None, lines: list[str]
    ) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            lines.append(text)
            logger.debug("native stderr: %s", text)

    def _handle_event(self, line: str, job_id: str) -> None:
        """Parse and log structured worker events; ignore plain logs."""
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                event = json.loads(stripped)
                logger.info("native event %s: %s", job_id, event)
            except json.JSONDecodeError:
                logger.debug("native stdout: %s", line)
            return
        if stripped.startswith("completed="):
            logger.info("native summary %s: %s", job_id, stripped)
            return
        logger.debug("native stdout: %s", line)

    def _parse_result(
        self,
        request: NativeJobRequest,
        stdout_lines: list[str],
        stderr_lines: list[str],
    ) -> NativeJobResult | NativeJobError:
        summary = self._find_summary(stdout_lines)
        if summary is None:
            return NativeJobError(
                job_id=request.job_id,
                code=NativeJobErrorCode.PROTOCOL_ERROR,
                message="native worker completed but no summary line found",
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
            )

        fields = _parse_key_value_summary(summary)
        detections_path = request.output_dir / "detections.jsonl"
        if not detections_path.exists():
            detections_path = None

        return NativeJobResult(
            job_id=request.job_id,
            completed=fields.get("completed", False),
            decoded_frames=fields.get("decoded_frames", 0),
            processed_frames=fields.get("processed_frames", 0),
            detections=fields.get("detections", 0),
            exit_code=fields.get("exit_code", 0),
            detections_path=detections_path,
            metadata=fields,
        )

    @staticmethod
    def _find_summary(lines: list[str]) -> str | None:
        for line in reversed(lines):
            if line.strip().startswith("completed="):
                return line.strip()
        return None

    @staticmethod
    def _terminate(proc: asyncio.subprocess.Process) -> None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass


def _parse_key_value_summary(line: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for part in line.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if value.isdigit():
            result[key] = int(value)
        elif value in ("true", "false"):
            result[key] = value == "true"
        else:
            result[key] = value
    return result
