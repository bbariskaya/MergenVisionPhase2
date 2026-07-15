"""Batch-1 baseline source-invariant regression tests.

These tests fail immediately if anyone reintroduces batch>1, invalid tracker
properties, or the MV_DISABLE_TRACKER debug bypass accidentally.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
NATIVE = REPO / "backend" / "native"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_streammux_batch_size_is_one():
    main_cpp = _read(NATIVE / "worker" / "main.cpp")
    match = re.search(r'"batch-size"\s*,\s*(\d+)', main_cpp)
    assert match is not None, "streammux batch-size not found in main.cpp"
    assert match.group(1) == "1", f"streammux batch-size must be 1, got {match.group(1)}"


def test_preprocess_network_input_shape_is_batch_one():
    cfg = _read(NATIVE / "configs" / "retinaface_preprocess.txt")
    match = re.search(r'network-input-shape\s*=\s*([^\n]+)', cfg)
    assert match is not None, "network-input-shape missing"
    assert match.group(1).strip() == "1;3;640;640", f"unexpected preprocess shape {match.group(1)}"


def test_plugin_inference_uses_batch_one():
    plugin = _read(NATIVE / "plugins" / "gst-nvdsretinaface" / "gstnvdsretinaface.cpp")
    match = re.search(r'engine->infer\([^,]+,\s*(\d+)', plugin)
    assert match is not None, "engine->infer call not found"
    assert match.group(1) == "1", f"plugin inference batch must be 1, got {match.group(1)}"


def test_tracker_has_no_batch_size_property():
    main_cpp = _read(NATIVE / "worker" / "main.cpp")
    tracker_block = re.search(r'g_object_set\(G_OBJECT\(tracker\).*?\);', main_cpp, re.S)
    assert tracker_block is not None, "tracker g_object_set block not found"
    assert "batch-size" not in tracker_block.group(0), (
        "nvtracker element does not expose a batch-size property"
    )


def test_no_disable_tracker_debug_branch_in_production():
    main_cpp = _read(NATIVE / "worker" / "main.cpp")
    assert "MV_DISABLE_TRACKER" not in main_cpp, (
        "MV_DISABLE_TRACKER debug bypass must not be compiled into production"
    )


def test_no_batch_greater_than_one_literals():
    """Reject accidental batch-4/8/16 source changes in worker and plugin."""
    for src in [NATIVE / "worker" / "main.cpp", NATIVE / "plugins" / "gst-nvdsretinaface" / "gstnvdsretinaface.cpp"]:
        text = _read(src)
        for literal in ["batch-size", 4, 8, 16]:
            # streammux batch-size=1 is allowed; other explicit batch literals are not.
            if literal == "batch-size":
                continue
            assert f" {literal} " not in text and f"\n{literal}" not in text, (
                f"found raw batch literal {literal} in {src.name}"
            )
