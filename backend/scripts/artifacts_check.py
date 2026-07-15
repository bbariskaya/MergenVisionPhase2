#!/usr/bin/env python3
"""Verify required model artifacts exist and match their declared SHA-256 hashes."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "docs" / "model_artifacts" / "MANIFEST.yaml"
    if not manifest_path.exists():
        print(f"FAIL: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)

    errors: list[str] = []
    for artifact in manifest.get("artifacts", []):
        for entry in artifact.get("files", []):
            rel_path = entry["path"]
            expected_sha = entry.get("sha256")
            full_path = repo_root / rel_path
            if not full_path.exists():
                errors.append(f"missing: {rel_path}")
                continue
            if expected_sha:
                actual_sha = _compute_sha256(full_path)
                if actual_sha != expected_sha:
                    errors.append(
                        f"hash mismatch: {rel_path} expected={expected_sha} got={actual_sha}"
                    )

    if errors:
        print("artifacts-check failed:", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
        return 1

    print("artifacts-check: all required artifacts present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
