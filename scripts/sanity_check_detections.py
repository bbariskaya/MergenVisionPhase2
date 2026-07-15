#!/usr/bin/env python3
"""Basic sanity check for worker detections.jsonl output."""
import json
import sys
from pathlib import Path

def main(path: str) -> int:
    p = Path(path)
    if not p.exists():
        print(f"FAIL: {path} not found", file=sys.stderr)
        return 1
    frames = 0
    total = 0
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"FAIL: invalid JSON: {e}", file=sys.stderr)
                return 1
            if "frame" not in rec or "pts_ms" not in rec or "detections" not in rec:
                print("FAIL: missing keys", file=sys.stderr)
                return 1
            frames += 1
            for d in rec["detections"]:
                total += 1
                for k in ("x1", "y1", "x2", "y2", "score"):
                    if k not in d:
                        print(f"FAIL: missing detection key {k}", file=sys.stderr)
                        return 1
                if d["x1"] >= d["x2"] or d["y1"] >= d["y2"]:
                    print(f"FAIL: invalid bbox {d}", file=sys.stderr)
                    return 1
                if not (0.0 <= d["score"] <= 1.0):
                    print(f"FAIL: invalid score {d['score']}", file=sys.stderr)
                    return 1
    print(f"OK: {frames} frames, {total} detections")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: sanity_check_detections.py <detections.jsonl>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
