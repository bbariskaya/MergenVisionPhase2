#!/usr/bin/env python3
"""Native dynamic-link health check.

Verifies that:
- the plugins our build produces have a sane dependency map (no missing libs),
- no CUDA runtime symbols are left unresolved in our .so files,
- gst-inspect-1.0 can load the nvdsretinaface GStreamer plugin.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO / "backend" / "native" / "build"
CONTAINER = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"

OUR_LIBS = [
    BUILD_DIR / "libgstnvdsretinaface.so",
    BUILD_DIR / "libretinaface_parser.so",
]

DEPENDENCY_WHITELIST_RE = re.compile(
    r"^(libgst|libgobject|libglib|libgstreamer|libnvds|libcuda|"
    r"libcudart|libstdc\+\+|libgcc_s|libpthread|libc\.so|libm\.so|"
    r"libdl\.so|librt\.so|ld-linux)"
)


def _dockerized(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "device=0",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-e", "GST_PLUGIN_PATH=/app/backend/native/build",
        "-v", f"{REPO}:/app",
        "-w", "/app",
        "--entrypoint", args[0],
        CONTAINER,
        *args[1:],
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _container_path(path: Path) -> str:
    return f"/app/{path.relative_to(REPO).as_posix()}"


def check_dependencies() -> int:
    failures = 0
    for so in OUR_LIBS:
        cpath = _container_path(so)
        result = _dockerized(["ldd", cpath])
        if result.returncode != 0:
            print(f"FAIL: ldd {so}: {result.stderr}", file=sys.stderr)
            failures += 1
            continue
        missing = [line for line in result.stdout.splitlines() if "not found" in line]
        if missing:
            print(f"FAIL: {so.name} has missing dependencies:", file=sys.stderr)
            for line in missing:
                print(f"  {line}", file=sys.stderr)
            failures += 1
        else:
            print(f"OK: {so.name} dependencies resolved")
    return failures


def check_no_unresolved_cuda() -> int:
    failures = 0
    for so in OUR_LIBS:
        cpath = _container_path(so)
        result = _dockerized(["nm", "-D", cpath])
        if result.returncode != 0:
            print(f"FAIL: nm -D {so}: {result.stderr}", file=sys.stderr)
            failures += 1
            continue
        undefined_cuda = [
            line.strip()
            for line in result.stdout.splitlines()
            if " U " in line and ("cuda" in line.lower() or "cudart" in line.lower())
        ]
        # Versioned references such as cudaPointerGetAttributes@libcudart.so.13
        # are resolved via a declared DT_NEEDED dependency and are fine.
        unversioned_cuda = [line for line in undefined_cuda if "@" not in line]
        if unversioned_cuda:
            print(
                f"FAIL: {so.name} has unversioned/unresolved CUDA symbols:", file=sys.stderr
            )
            for line in unversioned_cuda:
                print(f"  {line}", file=sys.stderr)
            failures += 1
        else:
            print(f"OK: {so.name} has no unversioned unresolved CUDA symbols")
    return failures


def check_gst_inspect() -> int:
    result = _dockerized(["gst-inspect-1.0", "nvdsretinaface"])
    if result.returncode != 0 or "Plugin Details" not in result.stdout:
        print(
            f"FAIL: gst-inspect-1.0 nvdsretinaface failed:\n{result.stdout}\n{result.stderr}",
            file=sys.stderr,
        )
        return 1
    # Make sure the old diagnostic parser no longer causes a load failure.
    our_lib_names = {lib.name for lib in OUR_LIBS}
    for line in result.stderr.splitlines():
        if "undefined symbol" in line and any(name in line for name in our_lib_names):
            print(
                f"FAIL: gst-inspect-1.0 reported an undefined symbol in one of our libs:\n{line}",
                file=sys.stderr,
            )
            return 1
    print("OK: gst-inspect-1.0 nvdsretinaface loads cleanly")
    return 0


def main() -> int:
    if not all(lib.exists() for lib in OUR_LIBS):
        print("FAIL: build products missing; run make backend-native-build first", file=sys.stderr)
        return 1

    failures = 0
    failures += check_dependencies()
    failures += check_no_unresolved_cuda()
    failures += check_gst_inspect()

    if failures:
        print(f"\nNative link check FAILED ({failures} issue(s))", file=sys.stderr)
        return 1
    print("\nNative link check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
