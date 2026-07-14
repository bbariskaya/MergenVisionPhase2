"""Render annotated Phoebe faces on disk as visual PNG files.

Reads data/annotations/Phoebe/annotations.yaml and writes one image per entry
with the bounding box and five landmarks drawn. Output goes to
out/phoebe_annotated/ by default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_PATH = REPO_ROOT / "data" / "annotations" / "Phoebe" / "annotations.yaml"
OUT_DIR = REPO_ROOT / "out" / "phoebe_annotated"


def main() -> int:
    if not ANNOTATIONS_PATH.exists():
        print(f"FATAL: annotations not found: {ANNOTATIONS_PATH}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_PATH) as f:
        doc = yaml.safe_load(f)

    count = 0
    for ann in doc.get("annotations", []):
        img_path = REPO_ROOT / ann["media_path"]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"WARNING: cannot read {img_path}")
            continue

        x1, y1, x2, y2 = map(int, ann["bbox_xyxy"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for idx, (lx, ly) in enumerate(ann["landmarks_5x2"]):
            color = (0, 255, 0) if idx < 2 else (0, 255, 255) if idx == 2 else (255, 0, 0)
            cv2.circle(img, (int(lx), int(ly)), 3, color, -1)

        label = f"{doc['identity']['canonical_face_id']} score={ann['score']:.3f}"
        cv2.putText(img, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out_name = Path(ann["media_path"]).stem + "_annotated.jpg"
        out_path = OUT_DIR / out_name
        cv2.imwrite(str(out_path), img)
        count += 1

    print(f"Wrote {count} annotated images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
