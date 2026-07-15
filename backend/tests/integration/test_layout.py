"""Repository layout acceptance tests for Sprint 02.

These tests verify that production source is located in the expected monorepo
locations and that backend/frontend code does not depend on each other.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
FRONTEND_ROOT = REPO_ROOT / "frontend"

PRODUCTION_SOURCE_EXTS = {".py", ".cpp", ".cu", ".h", ".hpp"}
GENERATED_DIRS = {
    "native/build",
    "frontend/node_modules",
    "frontend/dist",
    "frontend/artifacts",
    "frontend/test-results",
    "out",
    ".git",
}
NON_PRODUCTION_DIRS = {
    "docs",
    "opensourcereferences",
}


def _is_production_source(path: Path) -> bool:
    return path.suffix in PRODUCTION_SOURCE_EXTS and not path.name.endswith(".d")


def _walk_skipping_generated(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out generated directories in-place to avoid recursing into them.
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [
            d
            for d in dirnames
            if str(rel_dir / d) not in GENERATED_DIRS
            and not str(rel_dir / d).startswith("native/build")
            and not str(rel_dir / d).startswith("frontend/node_modules")
            and not str(rel_dir / d).startswith("frontend/dist")
            and not str(rel_dir / d).startswith("out")
        ]
        for filename in filenames:
            yield Path(dirpath) / filename


def test_backend_native_contains_cmake():
    assert (BACKEND_ROOT / "native" / "CMakeLists.txt").exists()


def test_backend_contains_python_app():
    assert (BACKEND_ROOT / "app" / "__init__.py").exists()
    assert (BACKEND_ROOT / "app" / "domain" / "native_job.py").exists()
    assert (BACKEND_ROOT / "app" / "ports" / "native_worker.py").exists()
    assert (BACKEND_ROOT / "app" / "application" / "services" / "run_video_detection.py").exists()
    assert (BACKEND_ROOT / "app" / "infrastructure" / "native_worker" / "subprocess_adapter.py").exists()


def test_root_has_no_stray_production_source():
    for root, dirs, files in os.walk(REPO_ROOT):
        rel = Path(root).relative_to(REPO_ROOT)
        # Skip known generated/dependency trees and documentation-only dirs.
        if any(
            str(rel).startswith(skip)
            for skip in GENERATED_DIRS | NON_PRODUCTION_DIRS | {"backend", "frontend"}
        ):
            dirs[:] = []
            continue
        for filename in files:
            path = Path(root) / filename
            if _is_production_source(path):
                pytest.fail(f"production source outside backend/frontend: {path}")


def test_backend_does_not_import_frontend_typescript():
    # Only production source is forbidden from referencing frontend paths;
    # tests legitimately enumerate frontend files for layout/contract checks.
    production_roots = [BACKEND_ROOT / "app", BACKEND_ROOT / "native"]
    for root in production_roots:
        for path in _walk_skipping_generated(root):
            if path.suffix != ".py":
                continue
            text = path.read_text(errors="ignore")
            assert "frontend/src" not in text, f"{path} references frontend/src"


def test_frontend_does_not_import_backend_python():
    for path in _walk_skipping_generated(FRONTEND_ROOT):
        if path.suffix not in {".ts", ".tsx", ".js", ".jsx"}:
            continue
        # A frontend may reasonably import a relative API client path such as
        # "../../backend/..." only as a string/URL; we disallow actual Python
        # module references.
        text = path.read_text(errors="ignore")
        assert "from app." not in text, f"{path} imports Python app module"
        assert "import app." not in text, f"{path} imports Python app module"


def test_application_layer_does_not_import_infrastructure():
    """Application services must depend on ports, not concrete adapters."""
    app_services = BACKEND_ROOT / "app" / "application" / "services"
    for path in app_services.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "infrastructure" not in module, (
                    f"{path} imports infrastructure module {module}"
                )
