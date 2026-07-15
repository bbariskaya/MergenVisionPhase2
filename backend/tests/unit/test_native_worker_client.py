#!/usr/bin/env python3
"""Unit tests for app.infrastructure.native_worker.client.

These tests do **not** execute Docker; they verify command construction.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.infrastructure.native_worker.client import NativeDetectorClient


class NativeDetectorClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = NativeDetectorClient(".")

    def test_default_container_and_worker(self) -> None:
        self.assertEqual(
            self.client.container, "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
        )
        self.assertEqual(
            self.client.worker_path,
            "/app/backend/native/build/deepstream_face_worker",
        )

    def test_default_gpu_device_is_zero(self) -> None:
        self.assertEqual(self.client.gpu_device, 0)

    def test_run_command_uses_container_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "backend" / "artifacts" / "videos").mkdir(parents=True)
            video = repo / "backend" / "artifacts" / "videos" / "clip.mp4"
            video.write_bytes(b"fake")
            client = NativeDetectorClient(repo)
            cmd = client.run_command(
                "backend/artifacts/videos/clip.mp4", "backend/out/cli"
            )
            self.assertIn(
                "/app/backend/native/build/deepstream_face_worker", cmd
            )
            self.assertIn("/app/backend/artifacts/videos/clip.mp4", cmd)
            self.assertIn("/app/backend/out/cli", cmd)
            # Logical CUDA device inside the container stays 0.
            self.assertEqual(cmd[-1], "0")

    def test_run_command_uses_request_gpu_device_for_host_gpu(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "backend" / "artifacts" / "videos").mkdir(parents=True)
            video = repo / "backend" / "artifacts" / "videos" / "clip.mp4"
            video.write_bytes(b"fake")
            client = NativeDetectorClient(repo, gpu_device=0)
            cmd = client.run_command(
                "backend/artifacts/videos/clip.mp4",
                "backend/out/cli",
                gpu_device=2,
            )
            gpus_index = cmd.index("--gpus")
            self.assertEqual(cmd[gpus_index + 1], "device=2")
            # Container CUDA_VISIBLE_DEVICES stays 0 (logical device).
            cuda_index = cmd.index("CUDA_VISIBLE_DEVICES=0")
            self.assertIsNotNone(cuda_index)

    def test_run_command_rejects_missing_video(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.client.run_command("/does/not/exist.mp4", "out")


if __name__ == "__main__":
    unittest.main()
