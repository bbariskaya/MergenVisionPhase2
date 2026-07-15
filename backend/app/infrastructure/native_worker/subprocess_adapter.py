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

_PROCESS_TERMINATION_TIMEOUT_SECONDS = 5.0


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
        gpu_device: int = 0,
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.client = NativeDetectorClient(
            repo_root,
            container=container or DEFAULT_CONTAINER,
            worker_path=worker_path or DEFAULT_WORKER,
            gst_plugin_path=gst_plugin_path or DEFAULT_GST_PLUGIN_PATH,
            gpu_device=gpu_device,
        )
        self.timeout_seconds = timeout_seconds

    async def process_video(
        self, request: NativeJobRequest
    ) -> NativeJobResult | NativeJobError:
        proc: asyncio.subprocess.Process | None = None
        stdout_task: asyncio.Task[None] | None = None
        stderr_task: asyncio.Task[None] | None = None
        try:
            cmd = self.client.run_command(
                request.video_path,
                request.output_dir,
                tracker_config=request.tracker_config,
                gpu_device=request.gpu_device,
            )
            logger.info("starting native job %s: %s", request.job_id, " ".join(cmd))

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
                await self._terminate(proc, stdout_task, stderr_task)
                return NativeJobError(
                    job_id=request.job_id,
                    code=NativeJobErrorCode.TIMEOUT,
                    message="native worker timed out",
                    stderr="\n".join(stderr_lines),
                )
            except asyncio.CancelledError:
                await self._terminate(proc, stdout_task, stderr_task)
                return NativeJobError(
                    job_id=request.job_id,
                    code=NativeJobErrorCode.CANCELLED,
                    message="native worker cancelled",
                    stderr="\n".join(stderr_lines),
                )

            return self._parse_result(
                request, stdout_lines, stderr_lines, proc.returncode
            )
        except FileNotFoundError as exc:
            exc_message = str(exc)
            if "input video not found" in exc_message:
                return NativeJobError(
                    job_id=request.job_id,
                    code=NativeJobErrorCode.INPUT_NOT_FOUND,
                    message="input video not found",
                    stderr=exc_message,
                )
            return NativeJobError(
                job_id=request.job_id,
                code=NativeJobErrorCode.WORKER_FAILED,
                message="native worker executable not found",
                stderr=exc_message,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("unexpected adapter error for %s", request.job_id)
            return NativeJobError(
                job_id=request.job_id,
                code=NativeJobErrorCode.WORKER_FAILED,
                message="unexpected adapter error",
                stderr=str(exc),
            )
        finally:
            if proc is not None and proc.returncode is None:
                await self._terminate(proc, stdout_task, stderr_task)

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
        returncode: int,
    ) -> NativeJobResult | NativeJobError:
        summary = self._find_summary(stdout_lines)
        if summary is None:
            return NativeJobError(
                job_id=request.job_id,
                code=NativeJobErrorCode.PROTOCOL_ERROR,
                message="native worker completed but no summary line was found",
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
                exit_code=returncode,
            )

        fields = _parse_key_value_summary(summary)
        completed = fields.get("completed", False)
        summary_exit_code = fields.get("exit_code", returncode)
        detections_path = request.output_dir / "detections.jsonl"

        if summary_exit_code != returncode:
            return NativeJobError(
                job_id=request.job_id,
                code=NativeJobErrorCode.PROTOCOL_ERROR,
                message="native worker exit code disagreed with its summary",
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
                exit_code=returncode,
            )

        if returncode != 0:
            return NativeJobError(
                job_id=request.job_id,
                code=NativeJobErrorCode.WORKER_FAILED,
                message="native worker exited with a non-zero status",
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
                exit_code=returncode,
            )

        if not completed:
            return NativeJobError(
                job_id=request.job_id,
                code=NativeJobErrorCode.WORKER_FAILED,
                message="native worker reported an incomplete result",
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
                exit_code=returncode,
            )

        detections_count = fields.get("detections", 0)
        if detections_count > 0 and not detections_path.exists():
            return NativeJobError(
                job_id=request.job_id,
                code=NativeJobErrorCode.PROTOCOL_ERROR,
                message="native worker result is missing its detections file",
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
                exit_code=returncode,
            )

        return NativeJobResult(
            job_id=request.job_id,
            completed=True,
            decoded_frames=fields.get("decoded_frames", 0),
            processed_frames=fields.get("processed_frames", 0),
            detections=detections_count,
            exit_code=returncode,
            detections_path=detections_path if detections_path.exists() else None,
            metadata=fields,
        )

    @staticmethod
    def _find_summary(lines: list[str]) -> str | None:
        for line in reversed(lines):
            if line.strip().startswith("completed="):
                return line.strip()
        return None

    @staticmethod
    async def _terminate(
        proc: asyncio.subprocess.Process,
        stdout_task: asyncio.Task[None] | None,
        stderr_task: asyncio.Task[None] | None,
    ) -> None:
        if proc.returncode is not None:
            pass
        else:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(
                        proc.wait(),
                        timeout=_PROCESS_TERMINATION_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            except ProcessLookupError:
                pass

        for task in (stdout_task, stderr_task):
            if task is None or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
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
