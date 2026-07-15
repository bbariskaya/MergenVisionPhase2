"""Generate one representative embedding per gallery image for every gallery identity."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    import yaml
except ImportError as exc:
    print(f"FATAL: missing dependency: {exc}", file=sys.stderr)
    sys.exit(1)

from test_engines_and_annotate_phoebe import (
    CONF_THRESHOLD,
    OrtSession,
    decode_retinaface,
    preprocess_detector,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

ARC_FACE_SRC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _similarity_transform(landmarks: np.ndarray, dest: np.ndarray) -> np.ndarray:
    src = landmarks.astype(np.float64)
    dst = dest.astype(np.float64)
    n = src.shape[0]
    src_mean = src.sum(axis=0) / n
    dst_mean = dst.sum(axis=0) / n
    num_a = num_b = denom = 0.0
    for i in range(n):
        xs = src[i, 0] - src_mean[0]
        ys = src[i, 1] - src_mean[1]
        xd = dst[i, 0] - dst_mean[0]
        yd = dst[i, 1] - dst_mean[1]
        num_a += xs * xd + ys * yd
        num_b += xs * yd - ys * xd
        denom += xs * xs + ys * ys
    a = num_a / denom if denom != 0 else 0.0
    b = num_b / denom if denom != 0 else 0.0
    tx = dst_mean[0] - a * src_mean[0] + b * src_mean[1]
    ty = dst_mean[1] - b * src_mean[0] - a * src_mean[1]
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float32)


def _align_face(image_bgr: np.ndarray, landmarks_orig: np.ndarray, size: int = 112) -> np.ndarray:
    M = _similarity_transform(landmarks_orig.astype(np.float32), ARC_FACE_SRC)
    aligned_bgr = cv2.warpAffine(image_bgr, M, (size, size), borderValue=0.0)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return aligned_rgb


def _preprocess_recognizer(aligned_rgb: np.ndarray) -> np.ndarray:
    arr = (aligned_rgb - 127.5) / 127.5
    arr = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(arr[np.newaxis, ...], dtype=np.float32)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


def _select_faces_for_identity(
    retina_onnx: OrtSession,
    recognizer_onnx: OrtSession,
    identity: str,
) -> dict[Path, dict]:
    folder = ARTIFACTS_DIR / "gallery" / identity
    paths = sorted(p for ext in ["*.jpg", "*.jpeg", "*.png"] for p in folder.glob(ext))
    print(f"{identity}: found {len(paths)} images")

    per_image = []
    for p in paths:
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"WARNING: cannot read {p}")
            continue
        h, w = img_bgr.shape[:2]
        det_input = preprocess_detector(img_bgr)
        outputs = retina_onnx.run(det_input)
        boxes, scores, landms = decode_retinaface(
            outputs["loc"][0], outputs["conf"][0], outputs["landms"][0], (w, h)
        )
        detections = []
        for box, score, lms in zip(boxes, scores, landms):
            aligned = _align_face(img_bgr, lms)
            rec_input = _preprocess_recognizer(aligned)
            emb = recognizer_onnx.run(rec_input)["1333"][0]
            detections.append({
                "bbox_xyxy": box.tolist(),
                "landmarks_5x2": lms.tolist(),
                "score": float(score),
                "embedding": _l2_normalize(emb.reshape(1, -1))[0].tolist(),
            })
        per_image.append({"path": p, "w": w, "h": h, "detections": detections})

    clean_embeddings = [
        d["detections"][0]["embedding"]
        for d in per_image
        if len(d["detections"]) == 1 and d["detections"][0]["score"] >= 0.7
    ]
    if clean_embeddings:
        centroid = _l2_normalize(np.mean(clean_embeddings, axis=0, keepdims=True))[0]
        print(f"  clean reference centroid built from {len(clean_embeddings)} images")
    else:
        centroid = None
        print("  WARNING: no clean single-face images; falling back to highest score")

    selected: dict[Path, dict] = {}
    for item in per_image:
        dets = item["detections"]
        if not dets:
            print(f"WARNING: no face in {item['path'].name}")
            continue
        if centroid is not None and len(dets) > 1:
            embs = np.array([d["embedding"] for d in dets])
            sims = embs @ centroid
            idx = int(np.argmax(sims))
        else:
            idx = int(np.argmax([d["score"] for d in dets]))
        chosen = dets[idx]
        chosen["original_width"] = item["w"]
        chosen["original_height"] = item["h"]
        selected[item["path"]] = chosen
    return selected


def _centroid(embeddings: list[list[float]]) -> list[float]:
    arr = np.array(embeddings, dtype=np.float32)
    return _l2_normalize(np.mean(arr, axis=0, keepdims=True))[0].tolist()


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    artifacts_dir = repo_root / "artifacts"
    models_dir = artifacts_dir / "models"
    retina_path = models_dir / "retinaface_r50_dynamic.onnx"
    recognizer_path = models_dir / "glintr100.onnx"
    gallery_dir = artifacts_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    retina_onnx = OrtSession(retina_path)
    recognizer_onnx = OrtSession(recognizer_path)

    identities = ["Chandler", "Joey", "Monica", "Phoebe", "Rachel", "Ross"]
    manifest: dict[str, Any] = {"schema_version": "1.0.0", "identities": {}}

    for identity in identities:
        selected = _select_faces_for_identity(retina_onnx, recognizer_onnx, identity)

        annotations = []
        embeddings: list[list[float]] = []
        for p, det in selected.items():
            annotations.append({
                "media_path": p.as_posix(),
                "media_type": "gallery_image",
                "canonical_face_id": identity.lower(),
                "display_name": identity,
                "original_width": det["original_width"],
                "original_height": det["original_height"],
                "bbox_xyxy": det["bbox_xyxy"],
                "landmarks_5x2": det["landmarks_5x2"],
                "embedding": det["embedding"],
            })
            embeddings.append(det["embedding"])

        identity_dir = gallery_dir / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        with open(identity_dir / "annotations.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "schema_version": "1.0.0",
                    "identity": {
                        "canonical_face_id": identity.lower(),
                        "display_name": identity,
                    },
                    "annotations": annotations,
                },
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        manifest["identities"][identity] = {
            "canonical_face_id": identity.lower(),
            "display_name": identity,
            "image_count": len(annotations),
            "centroid": _centroid(embeddings) if embeddings else [],
        }

    with open(gallery_dir / "gallery_centroids.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote gallery manifest to {gallery_dir / 'gallery_centroids.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
