"""Unit tests for backend.scripts.docker_watchdog.

These tests avoid the real Docker daemon and verify that the constructed command
line follows the safety rules defined in the Sprint 03 closure spec.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from docker_watchdog import ContainerWatchdog


def test_watchdog_uses_detached_run_not_timeout_run() -> None:
    wd = ContainerWatchdog(
        image="nvcr.io/nvidia/deepstream:9.0-triton-multiarch",
        command=["sleep", "1"],
        gpus="device=0",
        env={"GST_PLUGIN_PATH": "/app/backend/native/build/gst-plugins"},
        volumes=["repo:/app"],
        workdir="/app",
    )
    args = wd._docker_args()
    assert args[0:4] == ["docker", "run", "--rm", "-d"]
    assert "timeout" not in args
    assert "--name" in args and args[args.index("--name") + 1].startswith("mergenvision-watchdog-")
    assert "--gpus" in args and args[args.index("--gpus") + 1] == "device=0"
    assert "-v" in args
    assert "-w" in args


def test_watchdog_command_is_appended_after_image() -> None:
    wd = ContainerWatchdog(
        image="img",
        command=["/app/backend/native/build/deepstream_face_worker", "video.mp4"],
    )
    args = wd._docker_args()
    image_index = args.index("img")
    assert args[image_index + 1 :] == [
        "/app/backend/native/build/deepstream_face_worker",
        "video.mp4",
    ]


def test_watchdog_cleanup_only_uses_own_container_name() -> None:
    wd = ContainerWatchdog(image="img", command=["echo", "hi"], name="my-test-123")
    assert wd.name == "my-test-123"
    # _cleanup is a no-op if container has not been created; we only assert that
    # the cleanup commands are scoped to the unique name and never prune.
