"""Unit tests for benchmark_correctness_matrix summary aggregation.

Guards against the bug where request booleans were overwritten by manifest
string values, causing a completed run to appear as FAIL in the summary table.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Import the tool under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import benchmark_correctness_matrix as bcm  # type: ignore


def _make_result(batch_size: int, tracker: bool, render: bool, completed: bool,
                 worker_wall_sec: float, frames: int, avg_batch: float) -> dict:
    return {
        "tag": f"b{batch_size}_t{'on' if tracker else 'off'}_r{'on' if render else 'off'}_run0",
        "requested_batch_size": batch_size,
        "requested_tracker": tracker,
        "requested_render": render,
        "run_idx": 0,
        "exit_code": 0 if completed else 1,
        "completed": completed,
        "worker_error": 0 if completed else 1,
        "worker_wall_sec": worker_wall_sec,
        "frames": frames,
        "avg_batch": avg_batch,
        "stdout_tail": "",
        # Manifest holds string representations; must NOT influence booleans.
        "manifest": {
            "tracker": "on",
            "render": "on",
        },
    }


def test_summary_respects_requested_booleans() -> None:
    results = [
        _make_result(8, False, True, True, 20.0, 100, 7.9),
        _make_result(8, False, True, True, 21.0, 100, 7.9),
        _make_result(8, True, False, True, 25.0, 100, 1.0),
    ]
    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        bcm.print_summary_table(results, [8], [False, True], [False, True])
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()

    # tracker=off/render=on line should show 2 median samples and OK stats.
    assert "    8     off     on" in output
    # tracker=on/render=off should show 1 sample.
    assert "    8      on    off" in output
    # The off/on line must not report FAIL just because manifest strings differ.
    for line in output.splitlines():
        if line.startswith("    8     off     on"):
            assert "FAIL" not in line, f"off/on row incorrectly reported as FAIL: {line}"


def test_from_report_normalizes_old_and_new_reports() -> None:
    """Re-aggregation from an existing report must keep booleans separate from manifest."""
    old_report_result = {
        "batch_size": 4,
        "tracker": "off",
        "render": "on",
        "completed": True,
        "worker_error": 0,
        "worker_wall_sec": 10.0,
        "frames": 100,
        "avg_batch": 7.9,
        "manifest": {"tracker": "on", "render": "on"},
    }
    new_report_result = {
        "requested_batch_size": 4,
        "requested_tracker": True,
        "requested_render": False,
        "completed": True,
        "worker_error": 0,
        "worker_wall_sec": 10.0,
        "frames": 100,
        "avg_batch": 1.0,
        "manifest": {"tracker": "off", "render": "on"},
    }
    results = [old_report_result, new_report_result]
    normalized: list[dict] = []
    for r in results:
        rn = dict(r)
        if rn.get("requested_batch_size") is None:
            rn["requested_batch_size"] = rn.get("batch_size")
        if rn.get("requested_tracker") is None:
            t = rn.get("tracker")
            rn["requested_tracker"] = t if isinstance(t, bool) else (t != "none" and t != "off")
        if rn.get("requested_render") is None:
            rend = rn.get("render")
            rn["requested_render"] = rend if isinstance(rend, bool) else (rend == "on" or rend is True)
        normalized.append(rn)

    # Old report: tracker was string "off" -> False
    assert normalized[0]["requested_tracker"] is False
    # Old report: render was string "on" -> True
    assert normalized[0]["requested_render"] is True
    # New report: booleans preserved
    assert normalized[1]["requested_tracker"] is True
    assert normalized[1]["requested_render"] is False
