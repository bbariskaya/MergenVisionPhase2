"""
Dockerized Phoebe annotation + engine batch-parity smoke test.

Runs inside the `mergenvision/test:milestone-a` container:
  1. Detect all faces in artifacts/gallery/Phoebe using the RetinaFace ONNX model (CPU oracle).
  2. Pick the Phoebe face per image via embedding similarity to clean single-face refs.
  3. Save selected bbox/landmarks/embedding to artifacts/annotations/Phoebe/annotations.yaml.
  4. Load the inherited RetinaFace and glintr100 TensorRT engines and verify that
     batched inference (1, 2, 4, 8, 16) produces the same outputs as batch-size=1.

This script intentionally keeps full-frame decoded images in CPU because it is a
parity/oracle harness, not the production DeepStream hot path.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

try:
    import cv2
    import numpy as np
    import onnxruntime as ort
    import tensorrt as trt
    import yaml
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError as exc:
    print(f"FATAL: missing dependency: {exc}", file=sys.stderr)
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
ENGINES_DIR = ARTIFACTS_DIR / "engines"
GALLERY_DIR = ARTIFACTS_DIR / "gallery" / "Phoebe"
ANNOTATIONS_DIR = ARTIFACTS_DIR / "annotations" / "Phoebe"
OUT_DIR = REPO_ROOT / "out"

RETINA_ONNX = MODELS_DIR / "retinaface_r50_dynamic.onnx"
RETINA_ENGINE = ENGINES_DIR / "retinaface_r50_dynamic.bs1.opt64.max256.fp16.trt1014.engine"
GLINTR100_ONNX = MODELS_DIR / "glintr100.onnx"
GLINTR100_ENGINE = ENGINES_DIR / "glintr100.bs1.opt128.max256.fp16.trt1014.engine"

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
RETINA_MEAN = np.array([104.0, 117.0, 123.0], dtype=np.float32)
RETINA_INPUT_SIZE = 640

# Standard ArcFace 112x112 reference landmarks (RGB pipeline)
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


def _build_priors(image_size: int = 640) -> np.ndarray:
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    anchors = []
    for k, step in enumerate(steps):
        f_h = int(np.ceil(image_size / step))
        f_w = int(np.ceil(image_size / step))
        for i, j in product(range(f_h), range(f_w)):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_size
                s_ky = min_size / image_size
                cx = (j + 0.5) * step / image_size
                cy = (i + 0.5) * step / image_size
                anchors += [cx, cy, s_kx, s_ky]
    priors = np.array(anchors, dtype=np.float32).reshape(-1, 4)
    return priors


_PRIORS = _build_priors(RETINA_INPUT_SIZE)
_VARIANCE = np.array([0.1, 0.2], dtype=np.float32)


def _nms_cpu(boxes: np.ndarray, scores: np.ndarray, threshold: float, top_k: int = 2000) -> list[int]:
    order = np.argsort(-scores, kind="stable")[:top_k]
    keep: list[int] = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= threshold]
    return keep


def decode_retinaface(
    loc: np.ndarray,
    conf: np.ndarray,
    landms: np.ndarray,
    original_wh: tuple[int, int],
    conf_threshold: float = CONF_THRESHOLD,
    nms_threshold: float = NMS_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boxes/landmarks/scores in original image coordinates.

    Input tensors are for a single image; loc/conf/landms have shapes
    (16800,4), (16800,2), (16800,10) respectively.
    """
    # conf is two-class; take positive class score.
    scores = conf[:, 1]
    valid = scores >= conf_threshold
    if valid.sum() == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 5, 2))

    loc = loc[valid]
    landms = landms[valid]
    scores = scores[valid]
    priors = _PRIORS[valid]

    cx = priors[:, 0] + loc[:, 0] * _VARIANCE[0] * priors[:, 2]
    cy = priors[:, 1] + loc[:, 1] * _VARIANCE[0] * priors[:, 3]
    w = priors[:, 2] * np.exp(loc[:, 2] * _VARIANCE[1])
    h = priors[:, 3] * np.exp(loc[:, 3] * _VARIANCE[1])
    boxes_640 = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    landmarks_640 = np.zeros((loc.shape[0], 5, 2), dtype=np.float32)
    for k in range(5):
        landmarks_640[:, k, 0] = priors[:, 0] + landms[:, k * 2] * _VARIANCE[0] * priors[:, 2]
        landmarks_640[:, k, 1] = priors[:, 1] + landms[:, k * 2 + 1] * _VARIANCE[0] * priors[:, 3]

    keep = _nms_cpu(boxes_640, scores, nms_threshold)
    boxes_640 = boxes_640[keep]
    landmarks_640 = landmarks_640[keep]
    scores = scores[keep]

    # Reverse squish-resize: decoder returns normalized [0,1] coordinates in 640-space.
    orig_w, orig_h = original_wh
    scale_x = float(orig_w)
    scale_y = float(orig_h)
    boxes_orig = boxes_640.copy()
    boxes_orig[:, [0, 2]] *= scale_x
    boxes_orig[:, [1, 3]] *= scale_y
    landmarks_orig = landmarks_640.copy()
    landmarks_orig[:, :, 0] *= scale_x
    landmarks_orig[:, :, 1] *= scale_y
    return boxes_orig, scores, landmarks_orig


def preprocess_detector(image_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image_bgr, (RETINA_INPUT_SIZE, RETINA_INPUT_SIZE)).astype(np.float32)
    resized -= RETINA_MEAN
    tensor = resized.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(tensor, dtype=np.float32)


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    A = np.empty((2 * len(src), 4), dtype=np.float64)
    for i, (sx, sy) in enumerate(src):
        A[2 * i] = [sx, -sy, 1.0, 0.0]
        A[2 * i + 1] = [sy, sx, 0.0, 1.0]
    b = dst.ravel()
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_param, tx, ty = x
    return np.array([[a, -b_param, tx], [b_param, a, ty]], dtype=np.float32)


def align_face(image_bgr: np.ndarray, landmarks_orig: np.ndarray, size: int = 112) -> np.ndarray:
    # OpenCV warpAffine on BGR image; we convert the aligned chip to RGB before recognition.
    M = _similarity_transform(landmarks_orig.astype(np.float32), ARC_FACE_SRC)
    aligned_bgr = cv2.warpAffine(image_bgr, M, (size, size), borderValue=0.0)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return aligned_rgb


def preprocess_recognizer(aligned_rgb: np.ndarray) -> np.ndarray:
    # ArcFace glintr100: RGB, mean=127.5, std=127.5
    arr = (aligned_rgb - 127.5) / 127.5
    arr = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(arr[np.newaxis, ...], dtype=np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


class OrtSession:
    def __init__(self, model_path: Path) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def run(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        return {o.name: v for o, v in zip(self.session.get_outputs(), self.session.run(None, {self.input_name: tensor}))}


class TrtEngine:
    """Minimal TensorRT 10 explicit-batch inference helper using pycuda."""

    def __init__(self, engine_path: Path) -> None:
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def list_io_tensors(self) -> list[tuple[str, str, trt.TensorIOMode]]:
        return [
            (
                self.engine.get_tensor_name(i),
                str(self.engine.get_tensor_dtype(self.engine.get_tensor_name(i))),
                self.engine.get_tensor_mode(self.engine.get_tensor_name(i)),
            )
            for i in range(self.engine.num_io_tensors)
        ]

    def infer(self, input_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Set explicit input shapes.
        for name, arr in input_dict.items():
            self.context.set_input_shape(name, arr.shape)

        buffers: dict[str, cuda.DeviceAllocation] = {}
        output_specs: list[tuple[str, tuple, np.dtype]] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            d_mem = cuda.mem_alloc(size)
            buffers[name] = d_mem
            if mode == trt.TensorIOMode.INPUT:
                arr_c = np.ascontiguousarray(input_dict[name].astype(dtype))
                cuda.memcpy_htod_async(d_mem, arr_c, self.stream)
            else:
                output_specs.append((name, tuple(shape), dtype))

        for name, d_mem in buffers.items():
            self.context.set_tensor_address(name, int(d_mem))

        self.context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()

        outputs: dict[str, np.ndarray] = {}
        for name, shape, dtype in output_specs:
            out = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh(out, buffers[name])
            outputs[name] = out

        for mem in buffers.values():
            mem.free()
        return outputs


def load_images(paths: list[Path]) -> list[np.ndarray]:
    return [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]


def find_phoebe_images() -> list[Path]:
    exts = {"*.jpg", "*.jpeg", "*.png"}
    paths = []
    for ext in exts:
        paths.extend(GALLERY_DIR.glob(ext))
    return sorted(paths)


def annotate_phoebe(
    retina_onnx: OrtSession,
    glintr100_onnx: OrtSession,
) -> dict[Path, dict]:
    """Return selected face annotation per image, keyed by path."""
    paths = find_phoebe_images()
    print(f"Found {len(paths)} Phoebe images in {GALLERY_DIR}")

    per_image: list[dict] = []
    for p in paths:
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"WARNING: cannot read {p}")
            continue
        h, w = img_bgr.shape[:2]
        det_input = preprocess_detector(img_bgr)
        outputs = retina_onnx.run(det_input)
        boxes, scores, landmarks = decode_retinaface(
            outputs["loc"][0], outputs["conf"][0], outputs["landms"][0], (w, h)
        )
        detections = []
        for box, score, lms in zip(boxes, scores, landmarks):
            aligned = align_face(img_bgr, lms)
            rec_input = preprocess_recognizer(aligned)
            emb = glintr100_onnx.run(rec_input)["1333"][0]
            detections.append({
                "bbox_xyxy": box.tolist(),
                "landmarks_5x2": lms.tolist(),
                "score": float(score),
                "embedding": l2_normalize(emb.reshape(1, -1))[0].tolist(),
            })
        per_image.append({"path": p, "w": w, "h": h, "detections": detections})

    # Build a clean reference centroid from images that have exactly one face.
    clean_embeddings = [
        d["detections"][0]["embedding"]
        for d in per_image
        if len(d["detections"]) == 1 and d["detections"][0]["score"] >= 0.7
    ]
    if not clean_embeddings:
        print("WARNING: no clean single-face images; using largest face per image")
        centroid = None
    else:
        centroid = l2_normalize(np.mean(clean_embeddings, axis=0, keepdims=True))[0]
        print(f"Clean reference centroid built from {len(clean_embeddings)} images")

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
            # Fallback to highest-score (or only) detection.
            idx = int(np.argmax([d["score"] for d in dets]))
        chosen = dets[idx]
        chosen["original_width"] = item["w"]
        chosen["original_height"] = item["h"]
        selected[item["path"]] = chosen

    return selected


def save_annotations(selected: dict[Path, dict]) -> None:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for p, d in sorted(selected.items()):
        rel = str(p.relative_to(REPO_ROOT))
        entries.append({
            "media_path": rel,
            "media_type": "gallery_image",
            "canonical_face_id": "phoebe_001",
            "original_width": d["original_width"],
            "original_height": d["original_height"],
            "bbox_xyxy": d["bbox_xyxy"],
            "landmarks_5x2": d["landmarks_5x2"],
            "embedding": d["embedding"],
            "score": d["score"],
            "visibility": "visible",
            "annotator": "cpu_oracle_detector+v1",
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        })
    annotations = {
        "schema_version": "1.0.0",
        "identity": {"canonical_face_id": "phoebe_001", "display_name": "Phoebe Buffay"},
        "annotations": entries,
    }
    out_path = ANNOTATIONS_DIR / "annotations.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(annotations, f, sort_keys=False)
    print(f"Wrote {len(entries)} annotations to {out_path}")


def _run_detector_batch_parity(engine: TrtEngine, images: list[np.ndarray], batch_sizes: list[int]) -> dict:
    """Run RetinaFace engine at several batch sizes and compare with batch=1."""
    results: dict = {"model": "retinaface_r50_dynamic", "tests": []}
    preprocessed = [preprocess_detector(img) for img in images]
    # batch=1 baseline
    baseline: list[dict] = []
    for x in preprocessed:
        out = engine.infer({"input": x})
        baseline.append({k: v.copy() for k, v in out.items()})

    for bs in batch_sizes:
        test: dict = {"batch_size": bs, "status": "PASS", "max_diff": {}}
        for start in range(0, len(preprocessed), bs):
            chunk = preprocessed[start : start + bs]
            if len(chunk) < bs:
                # Pad chunk with the final image to reach requested batch size.
                chunk = chunk + [chunk[-1].copy()] * (bs - len(chunk))
            batch_input = np.concatenate(chunk, axis=0)
            out_b = engine.infer({"input": batch_input})
            for i, b_idx in enumerate(range(start, min(start + bs, len(preprocessed)))):
                base = baseline[b_idx]
                for name in base:
                    a = base[name]
                    b = out_b[name][i]
                    max_diff = float(np.max(np.abs(a - b)))
                    test["max_diff"][name] = max(
                        test["max_diff"].get(name, 0.0), max_diff
                    )
                    if max_diff > 1e-2:
                        test["status"] = "FAIL"
                        test["fail_detail"] = f"batch={bs} sample={b_idx} tensor={name} max_diff={max_diff:.6f}"
                        print(f"  FAIL {test['fail_detail']}")
                        break
                if test["status"] == "FAIL":
                    break
            if test["status"] == "FAIL":
                break
        print(f"  RetinaFace batch_size={bs}: {test['status']}, max_diff={test['max_diff']}")
        results["tests"].append(test)
    return results


def _run_recognizer_batch_parity(
    engine: TrtEngine, aligned_faces: list[np.ndarray], batch_sizes: list[int]
) -> dict:
    """Run glintr100 engine at several batch sizes and compare with batch=1."""
    results: dict = {"model": "glintr100", "tests": []}
    preprocessed = [preprocess_recognizer(f) for f in aligned_faces]
    baseline: list[np.ndarray] = []
    for x in preprocessed:
        out = engine.infer({"input.1": x})
        emb = list(out.values())[0].reshape(-1)
        baseline.append(emb / np.linalg.norm(emb))

    for bs in batch_sizes:
        test: dict = {"batch_size": bs, "status": "PASS", "min_cosine": 1.0}
        for start in range(0, len(preprocessed), bs):
            chunk = preprocessed[start : start + bs]
            if len(chunk) < bs:
                chunk = chunk + [chunk[-1].copy()] * (bs - len(chunk))
            batch_input = np.concatenate(chunk, axis=0)
            out_b = engine.infer({"input.1": batch_input})
            embs_b = list(out_b.values())[0]
            norms = np.linalg.norm(embs_b, axis=1, keepdims=True)
            embs_b = embs_b / np.where(norms == 0, 1, norms)
            for i, b_idx in enumerate(range(start, min(start + bs, len(preprocessed)))):
                cos = float(baseline[b_idx] @ embs_b[i])
                test["min_cosine"] = min(test["min_cosine"], cos)
                if cos < 0.999:
                    test["status"] = "FAIL"
                    test["fail_detail"] = f"batch={bs} sample={b_idx} cosine={cos:.6f}"
                    print(f"  FAIL {test['fail_detail']}")
                    break
            if test["status"] == "FAIL":
                break
        print(f"  glintr100 batch_size={bs}: {test['status']}, min_cosine={test['min_cosine']:.6f}")
        results["tests"].append(test)
    return results


def run_batch_parity_tests(selected: dict[Path, dict]) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"timestamp": datetime.now(timezone.utc).isoformat(), "tests": []}

    paths = sorted(selected.keys())
    images = load_images(paths)
    print(f"\n=== TensorRT RetinaFace batch parity ({len(images)} images) ===")
    retina_engine = TrtEngine(RETINA_ENGINE)
    print("I/O tensors:", retina_engine.list_io_tensors())
    report["tests"].append(_run_detector_batch_parity(retina_engine, images, [1, 2, 4, 8, 16]))

    aligned_faces = []
    for p in paths:
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        lms = np.array(selected[p]["landmarks_5x2"], dtype=np.float32)
        aligned_faces.append(align_face(img_bgr, lms))

    print(f"\n=== TensorRT glintr100 batch parity ({len(aligned_faces)} faces) ===")
    glintr100_engine = TrtEngine(GLINTR100_ENGINE)
    print("I/O tensors:", glintr100_engine.list_io_tensors())
    report["tests"].append(_run_recognizer_batch_parity(glintr100_engine, aligned_faces, [1, 2, 4, 8, 16]))

    report_path = OUT_DIR / "phoebe_batch_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote batch test report to {report_path}")
    return report


def main() -> int:
    print("Loading ONNX models...")
    retina_onnx = OrtSession(RETINA_ONNX)
    glintr100_onnx = OrtSession(GLINTR100_ONNX)

    print("\n=== Annotating Phoebe faces ===")
    selected = annotate_phoebe(retina_onnx, glintr100_onnx)
    if not selected:
        print("ERROR: no annotations produced")
        return 1
    save_annotations(selected)

    print("\n=== TensorRT engine batch parity ===")
    report = run_batch_parity_tests(selected)
    all_ok = all(t["status"] == "PASS" for test_group in report["tests"] for t in test_group["tests"])
    if all_ok:
        print("\nALL BATCH PARITY TESTS PASSED")
    else:
        print("\nSOME BATCH PARITY TESTS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
