"""Batch-N invariant regression tests.

These tests fail immediately if anyone:
- hard-codes streammux batch-size to a literal,
- hard-codes the plugin inference batch to a literal,
- removes the batch-size > 1 + tracker-on rejection, or
- reintroduces the MV_DISABLE_TRACKER debug bypass.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
NATIVE = REPO / "backend" / "native"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_streammux_batch_size_uses_cli_option():
    main_cpp = _read(NATIVE / "worker" / "main.cpp")
    # Reject accidental hard-coded batch-size=1 literal.
    bad = re.search(r'"batch-size"\s*,\s*\d+', main_cpp)
    if bad:
        raise AssertionError(
            f"streammux batch-size must be set from a variable, found literal: {bad.group(0)}"
        )
    assert '"batch-size", opts.batch_size' in main_cpp, (
        "streammux batch-size must be driven by opts.batch_size"
    )


def test_plugin_inference_uses_runtime_batch():
    plugin = _read(NATIVE / "plugins" / "gst-nvdsretinaface" / "gstnvdsretinaface.cpp")
    bad = re.search(r'engine->infer\([^,]+,\s*\d+', plugin)
    if bad:
        raise AssertionError(
            f"plugin inference must use actual_batch variable, found literal call: {bad.group(0)}"
        )
    assert "impl->engine->infer(tensor_meta->raw_tensor_buffer, actual_batch," in plugin, (
        "plugin inference must use actual_batch variable"
    )


def test_tracker_requires_batch_size_one():
    main_cpp = _read(NATIVE / "worker" / "main.cpp")
    assert "opts->tracker_enabled && opts->batch_size > 1" in main_cpp, (
        "worker must reject tracker + batch-size > 1"
    )
    assert "NvMOT contract violation" in main_cpp, (
        "worker must emit a clear NvMOT contract error"
    )


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


def test_preprocess_config_uses_dynamic_batch():
    main_cpp = _read(NATIVE / "worker" / "main.cpp")
    assert 'fprintf(f, "network-input-shape=%d;3;640;640\\n", batch_size);' in main_cpp, (
        "runtime preprocess config must write network-input-shape from batch_size"
    )


def test_static_preprocess_config_matches_max_batch_one_default():
    cfg = _read(NATIVE / "configs" / "retinaface_preprocess.txt")
    match = re.search(r'network-input-shape\s*=\s*([^\n]+)', cfg)
    assert match is not None, "network-input-shape missing in static config"
    # Static config is a fallback/default; runtime config overrides the batch dimension.
    shape = match.group(1).strip()
    assert shape in {"1;3;640;640", "8;3;640;640"}, (
        f"unexpected static preprocess shape {shape}"
    )
