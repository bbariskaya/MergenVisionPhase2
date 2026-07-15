"""Verify the mvfacerecognizer GStreamer plugin can be loaded."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/user/Workspace/MergenVisionPhase2")
GST_INSPECT = "gst-inspect-1.0"


def test_mvfacerecognizer_plugin_is_registered() -> None:
    if not shutil.which(GST_INSPECT):
        pytest.skip("gst-inspect-1.0 not available on host")

    plugin_dir = REPO / "backend" / "native" / "build" / "gst-plugins"
    env = {
        **dict(subprocess.os.environ),
        "GST_PLUGIN_PATH": str(plugin_dir),
    }
    result = subprocess.run(
        [GST_INSPECT, "mvfacerecognizer"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"mvfacerecognizer not registered. stdout={result.stdout}, stderr={result.stderr}"
    )
    assert "mvfacerecognizer" in result.stdout
