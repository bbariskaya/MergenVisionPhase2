"""Thin client that builds commands to invoke the native worker.

This is an infrastructure helper. Domain/application code should depend on
:mod:`app.ports.native_worker`, not on this module directly.
"""
from __future__ import annotations

from pathlib import Path


DEFAULT_CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
DEFAULT_WORKER = "/app/backend/native/build/deepstream_face_worker"
DEFAULT_GST_PLUGIN_PATH = "/app/backend/native/build"


class NativeDetectorClient:
    """Build Docker commands to call the native worker once per job."""

    def __init__(
        self,
        repo_root: Path | str,
        *,
        container: str = DEFAULT_CONTAINER,
        worker_path: str = DEFAULT_WORKER,
        gst_plugin_path: str = DEFAULT_GST_PLUGIN_PATH,
        gpu_device: int = 0,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.container = container
        self.worker_path = worker_path
        self.gst_plugin_path = gst_plugin_path
        self.gpu_device = gpu_device

    def _docker_args(
        self, *, host_gpu: int, working_dir: str = "/app"
    ) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "--gpus",
            f"device={host_gpu}",
            "-e",
            "CUDA_VISIBLE_DEVICES=0",
            "-e",
            f"GST_PLUGIN_PATH={self.gst_plugin_path}",
            "-v",
            f"{self.repo_root}:/app",
            "-w",
            working_dir,
        ]

    def _resolve_host_path(self, path: Path | str, *, must_exist: bool = True) -> Path:
        """Resolve a path against the repo root and current working directory.

        Absolute paths are used as-is. Relative paths are first resolved against
        ``Path.cwd()`` and then ``self.repo_root``; the first candidate that is
        inside the repo and, when ``must_exist`` is true, exists is returned.
        """
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
            if must_exist and not resolved.exists():
                raise FileNotFoundError(f"input video not found: {resolved}")
            return resolved

        candidates = [(Path.cwd() / p).resolve(), (self.repo_root / p).resolve()]
        for candidate in candidates:
            try:
                candidate.relative_to(self.repo_root)
            except ValueError:
                continue
            if not must_exist or candidate.exists():
                return candidate

        raise FileNotFoundError(f"input video not found: {candidates[0]}")

    def run_command(
        self,
        input_video_path: Path | str,
        output_dir: Path | str,
        *,
        tracker_config: Path | str | None = None,
        gpu_device: int | None = None,
    ) -> list[str]:
        """Return the docker command to run the worker on one video.

        ``gpu_device`` is the host GPU index passed to ``--gpus device=...``.
        Inside the container logical CUDA device stays 0 (CUDA_VISIBLE_DEVICES).
        """
        host_input = self._resolve_host_path(input_video_path)
        host_output = self._resolve_host_path(output_dir, must_exist=False)

        host_output.mkdir(parents=True, exist_ok=True)

        # Inside the container the repository is mounted at /app, so use
        # repo-relative paths for the worker arguments.
        container_input = Path("/app") / host_input.relative_to(self.repo_root)
        container_output = Path("/app") / host_output.relative_to(self.repo_root)

        args: list[str] = [
            str(container_input),
            str(container_output),
            "0",  # logical GPU inside container
        ]
        if tracker_config is not None:
            host_cfg = Path(tracker_config)
            if not host_cfg.is_absolute():
                host_cfg = self.repo_root / host_cfg
            args.append(str(Path("/app") / host_cfg.relative_to(self.repo_root)))

        host_gpu = gpu_device if gpu_device is not None else self.gpu_device
        return [
            *self._docker_args(host_gpu=host_gpu),
            "--entrypoint",
            self.worker_path,
            self.container,
            *args,
        ]
