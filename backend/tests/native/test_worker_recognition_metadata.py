"""Sprint 05: worker fast-mode must produce recognition metadata."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path("/home/user/Workspace/MergenVisionPhase2")
CONTAINER = "mergenvision/deepstream-dev:9.0"
WORKER = "/app/backend/native/build/deepstream_face_worker"
GALLERY = "/app/backend/artifacts/gallery/gallery_centroids.json"
INPUT_VIDEO = "/app/backend/artifacts/videos/friendsshort_50f.mp4"
SKIP_REASON = "fast mode not yet implemented"


def _run_worker_in_docker(tmpdir: Path) -> subprocess.CompletedProcess[str]:
    run_name = tmpdir.name
    out_dir = REPO / "backend" / "out" / run_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    container_out = f"/app/backend/out/{run_name}"
    annotated = container_out + "/annotated.mp4"
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "device=0",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-e", "GST_PLUGIN_PATH=/app/backend/native/build/gst-plugins",
        "-v", f"{REPO}:/app",
        "-w", "/app",
        "--entrypoint", WORKER,
        CONTAINER,
        INPUT_VIDEO,
        container_out,
        "0",
        "--mode", "fast",
        "--batch-size", "8",
        "--gallery", GALLERY,
        "--threshold", "0.35",
        "--margin", "0.10",
        "--tracker", "off",
        "--render",
        "--annotated-output", annotated,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


@pytest.mark.skipif(not (REPO / "backend/native/build/deepstream_face_worker").exists(),
                    reason="worker binary not built")
def test_fast_mode_worker_accepts_gallery_and_runs_to_eos() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        result = _run_worker_in_docker(tmpdir)
        assert result.returncode == 0, (
            f"worker failed: stderr={result.stderr}\nstdout={result.stdout}"
        )


@pytest.mark.skipif(not (REPO / "backend/native/build/deepstream_face_worker").exists(),
                    reason="worker binary not built")
def test_fast_mode_recognition_jsonl_contains_identity_fields() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        result = _run_worker_in_docker(tmpdir)
        if result.returncode != 0:
            pytest.skip(SKIP_REASON)
        out_dir = REPO / "backend" / "out" / tmpdir.name
        jsonl_path = out_dir / "recognized_detections.jsonl"
        assert jsonl_path.exists(), "recognized_detections.jsonl must exist"
        found_any = False
        with jsonl_path.open() as f:
            for line in f:
                obj = json.loads(line)
                for det in obj.get("detections", []):
                    found_any = True
                    assert "identity_id" in det, "missing identity_id"
                    assert "identity_name" in det, "missing identity_name"
                    assert "status" in det, "missing status"
                    assert "top1_similarity" in det, "missing top1_similarity"
                    assert "top2_similarity" in det, "missing top2_similarity"
                    assert "margin" in det, "missing margin"
        assert found_any, "no detections in recognized JSONL"
