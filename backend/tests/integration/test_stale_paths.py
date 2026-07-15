"""Stale-path regression test.

Ensures no production source/config/docs reintroduce pre-monorepo root paths
or the old /app/{native,models,engines} layout.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

_SOURCE_GLOBS = [
    "backend/**/*.py",
    "backend/**/*.cpp",
    "backend/**/*.cu",
    "backend/**/*.h",
    "backend/**/*.hpp",
    "frontend/src/**/*.ts",
    "frontend/src/**/*.tsx",
]

_CONFIG_GLOBS = [
    "backend/native/configs/*.txt",
    "docker/*",
    "Makefile",
    ".gitignore",
    "backend/pyproject.toml",
]

_DOC_GLOBS = [
    "README.md",
    "backend/README.md",
    "docs/**/*.md",
    "frontend/docs/**/*.md",
]

# Each pattern must match the stale usage, not legitimate backend/out/...
_FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("root test_videos/", re.compile(r"\btest_videos/")),
    ("root DATASET/", re.compile(r"\bDATASET/")),
    ("/app/native/", re.compile(r"/app/native/")),
    ("/app/models/", re.compile(r"/app/models/")),
    ("/app/engines/", re.compile(r"/app/engines/")),
    ("root ./out/", re.compile(r"\./out/")),
    ("root ../out/", re.compile(r"\.\./out/")),
    ("root /out/", re.compile(r"(?<![\w/])/out/")),
    ("root out/", re.compile(r"(?<![\w/])out/")),
]

_EXCLUDED_PATHS = {
    # Rule file intentionally discusses legacy names as examples.
    REPO_ROOT / "AGENTS.md",
    # This test itself enumerates the forbidden patterns.
    REPO_ROOT / "backend" / "tests" / "integration" / "test_stale_paths.py",
}


def _collect_files(*globs: str) -> set[Path]:
    files: set[Path] = set()
    for pattern in globs:
        for path in REPO_ROOT.glob(pattern):
            if path.is_file():
                files.add(path.resolve())
    return files


def _is_excluded(path: Path) -> bool:
    return path in {p.resolve() for p in _EXCLUDED_PATHS}


def test_no_stale_paths_in_source_config_or_docs() -> None:
    all_files = _collect_files(*_SOURCE_GLOBS, *_CONFIG_GLOBS, *_DOC_GLOBS)
    failures: list[str] = []
    for path in all_files:
        if _is_excluded(path):
            continue
        text = path.read_text(errors="replace")
        for label, pattern in _FORBIDDEN_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                failures.append(f"{path}:{line_no}: {label} ({match.group()!r})")
    if failures:
        pytest.fail("\n".join(["stale paths found:"] + failures))
