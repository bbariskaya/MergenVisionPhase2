"""Tests for the frozen GlintR100 preprocess contract.

The contract must exist before any recognition code is written. This guards
against decisions that are changed after seeing test outputs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONTRACT_PATH = REPO / "backend" / "native" / "configs" / "glintr100_preprocess_contract.json"


def _load() -> dict:
    assert CONTRACT_PATH.exists(), f"GlintR100 preprocess contract not found: {CONTRACT_PATH}"
    with CONTRACT_PATH.open() as f:
        return json.load(f)


def test_contract_exists_and_has_required_fields() -> None:
    c = _load()
    required = {
        "schema_version",
        "model_sha256",
        "engine_sha256",
        "input_name",
        "output_name",
        "input_shape",
        "output_shape",
        "input_dtype",
        "output_dtype",
        "color_order",
        "normalization",
        "landmark_order",
        "canonical_template",
        "engine_profile",
        "output_l2_normalization",
        "pixel_center_rule",
        "border_mode",
        "preprocess_pipeline",
    }
    missing = required - c.keys()
    assert not missing, f"Missing contract fields: {missing}"


def test_contract_hashes_match_artifacts() -> None:
    import hashlib

    c = _load()
    model_path = REPO / "backend" / "artifacts" / "models" / "glintr100.onnx"
    engine_path = REPO / "backend" / "artifacts" / "engines" / "glintr100.bs1.opt128.max256.fp16.trt1014.engine"
    assert model_path.exists() and engine_path.exists()
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    engine_hash = hashlib.sha256(engine_path.read_bytes()).hexdigest()
    assert c["model_sha256"] == model_hash, "model_sha256 mismatch"
    assert c["engine_sha256"] == engine_hash, "engine_sha256 mismatch"


def test_input_output_contract() -> None:
    c = _load()
    assert c["input_name"] == "input.1"
    assert c["output_name"] == "1333"
    assert c["input_shape"] == [-1, 3, 112, 112]
    assert c["output_shape"] == [-1, 512]
    assert c["input_dtype"] == "float32"
    assert c["output_dtype"] == "float32"


def test_normalization_and_color_order() -> None:
    c = _load()
    assert c["color_order"] == "RGB", "recognizer input must be RGB for this checkpoint"
    norm = c["normalization"]
    assert norm["mode"] == "per_pixel"
    assert norm["subtract"] == 127.5
    assert norm["divide"] == 127.5


def test_engine_profile_bounds() -> None:
    c = _load()
    profile = c["engine_profile"]
    assert profile["min_batch"] == 1
    assert profile["opt_batch"] == 128
    assert profile["max_batch"] == 256


def test_landmark_order_and_template_shape() -> None:
    c = _load()
    assert c["landmark_order"] == [
        "left_eye",
        "right_eye",
        "nose",
        "left_mouth",
        "right_mouth",
    ]
    template = c["canonical_template"]
    assert isinstance(template, list) and len(template) == 5
    for pt in template:
        assert len(pt) == 2


def test_contract_version_is_semantic() -> None:
    c = _load()
    assert re.match(r"^\d+\.\d+", c["schema_version"])
