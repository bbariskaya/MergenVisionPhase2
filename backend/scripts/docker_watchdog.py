#!/usr/bin/env python3
"""Safe Docker container runner.

Provides a watchdog that:
  - launches a uniquely-named detached container,
  - waits for it with a bounded timeout,
  - stops/kills/removes the container on timeout or shell interruption,
  - only touches the container that was created,
  - never uses 'timeout docker run'.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import FrameType
from typing import Sequence


class ContainerWatchdog:
    def __init__(
        self,
        image: str,
        command: Sequence[str],
        *,
        name: str | None = None,
        gpus: str | None = None,
        env: dict[str, str] | None = None,
        volumes: Sequence[str] | None = None,
        workdir: str | None = None,
        timeout_seconds: int = 120,
        stop_grace: int = 10,
    ) -> None:
        self.image = image
        self.command = list(command)
        self.name = name or f"mergenvision-watchdog-{uuid.uuid4().hex[:10]}"
        self.gpus = gpus
        self.env = dict(env) if env else {}
        self.volumes = list(volumes) if volumes else []
        self.workdir = workdir
        self.timeout_seconds = timeout_seconds
        self.stop_grace = stop_grace
        self._cleanup_registered = False
        self._returncode: int | None = None
        self._stdout: str | None = None
        self._stderr: str | None = None

    def _docker_args(self) -> list[str]:
        cmd = ["docker", "run", "--rm", "-d", "--name", self.name]
        if self.gpus is not None:
            cmd.extend(["--gpus", self.gpus])
        for key, value in self.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        for vol in self.volumes:
            cmd.extend(["-v", vol])
        if self.workdir is not None:
            cmd.extend(["-w", self.workdir])
        cmd.append(self.image)
        cmd.extend(self.command)
        return cmd

    def _register_cleanup(self) -> None:
        if self._cleanup_registered:
            return

        def handler(signum: int, frame: FrameType | None) -> None:
            self._cleanup()
            sys.exit(128 + signum)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        self._cleanup_registered = True

    def _cleanup(self) -> None:
        # Only act on the container we created; never a global prune.
        for operation in (("stop", "-t", str(self.stop_grace)), ("kill",), ("rm", "-f")):
            with contextlib.suppress(FileNotFoundError):
                subprocess.run(
                    ["docker", operation[0], self.name, *operation[1:]],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )

    def run(self) -> subprocess.CompletedProcess[str]:
        self._register_cleanup()
        if self.name.encode().startswith((b"mergenvision", b"test-")):
            pass
        create = subprocess.run(
            self._docker_args(),
            capture_output=True,
            text=True,
            check=False,
        )
        if create.returncode != 0:
            raise RuntimeError(
                f"failed to create container: {create.stderr.strip() or create.stdout.strip()}"
            )

        try:
            wait = subprocess.run(
                ["docker", "wait", self.name],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if wait.returncode == 0:
                self._returncode = int(wait.stdout.strip())
            else:
                self._returncode = -1
        except subprocess.TimeoutExpired:
            self._cleanup()
            raise TimeoutError(
                f"container {self.name} did not finish within {self.timeout_seconds}s"
            )

        logs = subprocess.run(
            ["docker", "logs", self.name],
            capture_output=True,
            text=True,
            check=False,
        )
        self._stdout = logs.stdout
        self._stderr = logs.stderr

        self._cleanup()
        return subprocess.CompletedProcess(
            args=self._docker_args(),
            returncode=self._returncode,
            stdout=self._stdout,
            stderr=self._stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Docker container with a watchdog.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--volume", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    env: dict[str, str] = {}
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value

    watchdog = ContainerWatchdog(
        image=args.image,
        command=args.command,
        name=args.name,
        gpus=args.gpus,
        env=env,
        volumes=args.volume,
        workdir=args.workdir,
        timeout_seconds=args.timeout,
    )
    try:
        result = watchdog.run()
    except TimeoutError as exc:
        print(f"TIMEOUT: {exc}", file=sys.stderr)
        return 124

    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
