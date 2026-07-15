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

    def _docker_args(self, working_dir: str = "/app") -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "--gpus",
            f"device={self.gpu_device}",
            "-e",
            "CUDA_VISIBLE_DEVICES=0",
            "-e",
            f"GST_PLUGIN_PATH={self.gst_plugin_path}",
            "-v",
            f"{self.repo_root}:/app",
            "-w",
            working_dir,
        ]

    def run_command(
        self,
        input_video_path: Path | str,
        output_dir: Path | str,
        *,
        tracker_config: Path | str | None = None,
    ) -> list[str]:
        """Return the docker command to run the worker on one video."""
        host_input = Path(input_video_path)
        host_output = Path(output_dir)

        if not host_input.is_absolute():
            host_input = self.repo_root / host_input
        if not host_output.is_absolute():
            host_output = self.repo_root / host_output

        if not host_input.exists():
            raise FileNotFoundError(f"input video not found: {host_input}")
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

        return [
            *self._docker_args(),
            "--entrypoint",
            self.worker_path,
            self.container,
            *args,
        ]
