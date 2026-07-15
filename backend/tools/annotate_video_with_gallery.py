"""Assign DATASET identity labels to video detections using TensorRT recognition and track by spatial continuity."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError as exc:
    print(f"FATAL: missing dependency: {exc}", file=sys.stderr)
    sys.exit(1)


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

COLOR_MAP = {
    "Chandler": (255, 0, 0),
    "Joey": (0, 255, 0),
    "Monica": (0, 0, 255),
    "Phoebe": (255, 255, 0),
    "Rachel": (255, 0, 255),
    "Ross": (0, 255, 255),
    "unknown": (128, 128, 128),
}


class TrtEngine:
    def __init__(self, engine_path: Path) -> None:
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        input_name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(input_name, input_tensor.shape)
        buffers: dict[str, cuda.DeviceAllocation] = {}
        output_name: str | None = None
        output_shape: tuple | None = None
        output_dtype: np.dtype | None = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            buffers[name] = cuda.mem_alloc(size)
            if mode == trt.TensorIOMode.INPUT:
                arr_c = np.ascontiguousarray(input_tensor.astype(dtype))
                cuda.memcpy_htod_async(buffers[name], arr_c, self.stream)
            else:
                output_name = name
                output_shape = tuple(shape)
                output_dtype = dtype
        for name, d_mem in buffers.items():
            self.context.set_tensor_address(name, int(d_mem))
        self.context.execute_async_v3(self.stream.handle)
        if output_name is None or output_shape is None or output_dtype is None:
            raise RuntimeError("No output tensor found")
        out = np.empty(output_shape, dtype=output_dtype)
        cuda.memcpy_dtoh_async(out, buffers[output_name], self.stream)
        self.stream.synchronize()
        for mem in buffers.values():
            mem.free()
        return out


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
    return np.ascontiguousarray(arr, dtype=np.float32)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


def load_gallery_centroids(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path) as f:
        manifest = json.load(f)
    labels = []
    vectors = []
    for identity, info in manifest["identities"].items():
        labels.append(identity)
        vectors.append(info["centroid"])
    return labels, np.array(vectors, dtype=np.float32)


def recognize_faces(recognizer: TrtEngine, faces_rgb: list[np.ndarray]) -> np.ndarray:
    if not faces_rgb:
        return np.empty((0, 512), dtype=np.float32)
    batch = np.stack([_preprocess_recognizer(f) for f in faces_rgb], axis=0)
    raw = recognizer.infer(batch)
    return _l2_normalize(raw)


def assign_identities(
    embeddings: np.ndarray,
    labels: list[str],
    centroid_matrix: np.ndarray,
    threshold: float,
) -> list[tuple[str, float]]:
    sims = embeddings @ centroid_matrix.T
    best_idx = np.argmax(sims, axis=1)
    best_sim = sims[np.arange(len(best_idx)), best_idx]
    results = []
    for idx, sim in zip(best_idx, best_sim):
        if sim >= threshold:
            results.append((labels[int(idx)], float(sim)))
        else:
            results.append(("unknown", float(sim)))
    return results


def _center(box: dict) -> tuple[float, float]:
    return ((box["x1"] + box["x2"]) * 0.5, (box["y1"] + box["y2"]) * 0.5)


def _distance(p: tuple[float, float], q: tuple[float, float]) -> float:
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return (dx * dx + dy * dy) ** 0.5


def draw_label(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, text: str, color: tuple) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("--gallery", type=Path, default=Path("artifacts/gallery/gallery_centroids.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/recognition_annotations"))
    parser.add_argument("--engine", type=Path, default=Path("artifacts/engines/glintr100.bs1.opt128.max256.fp16.trt1014.engine"))

    parser.add_argument("--track-dist", type=float, default=120.0)
    parser.add_argument("--track-max-age", type=int, default=5)
    args = parser.parse_args()

    labels, centroids = load_gallery_centroids(args.gallery)
    recognizer = TrtEngine(args.engine)

    detections_by_frame: dict[int, list[dict]] = {}
    with open(args.detections) as f:
        for line in f:
            rec = json.loads(line)
            detections_by_frame[rec["frame"]] = rec["detections"]

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"FATAL: cannot open video {args.video}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = args.output_dir / f"{args.video.stem}_full_annotated.mp4"
    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    recognized_records: list[dict] = []
    next_track_id = 1
    active_tracks: list[dict] = []
    all_tracks: dict[int, dict] = {}
    frame_index = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        dets = detections_by_frame.get(frame_index, [])

        for t in active_tracks:
            t["age"] += 1
        active_tracks = [t for t in active_tracks if t["age"] <= args.track_max_age]

        if not dets:
            writer.write(frame)
            recognized_records.append({"frame": frame_index, "detections": []})
            continue

        faces_rgb = []
        centers = []
        for d in dets:
            lms = np.array(d["landmarks"], dtype=np.float32).reshape(5, 2)
            faces_rgb.append(_align_face(frame, lms))
            centers.append(_center(d))

        embeddings = recognize_faces(recognizer, faces_rgb)
        id_results = assign_identities(embeddings, labels, centroids, args.threshold)

        assignments = [-1] * len(dets)
        used = set()
        for i, center in enumerate(centers):
            best = -1
            best_dist = float("inf")
            for tid, t in enumerate(active_tracks):
                if tid in used:
                    continue
                dval = _distance(center, t["center"])
                if dval < best_dist and dval <= args.track_dist:
                    best_dist = dval
                    best = tid
            if best >= 0:
                used.add(best)
                assignments[i] = best

        labeled = []
        for i, (d, (name, sim)) in enumerate(zip(dets, id_results)):
            x1, y1, x2, y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])
            if assignments[i] >= 0:
                t = active_tracks[assignments[i]]
                t["center"] = centers[i]
                t["age"] = 0
                t["labels"][name] = t["labels"].get(name, 0) + 1
                t["sims"].append(sim)
                t["frames"].append(frame_index)
                track_id = t["track_id"]
            else:
                track_id = next_track_id
                next_track_id += 1
                new_track = {
                    "track_id": track_id,
                    "center": centers[i],
                    "age": 0,
                    "labels": {name: 1},
                    "sims": [sim],
                    "frames": [frame_index],
                }
                active_tracks.append(new_track)
                all_tracks[track_id] = new_track
                assignments[i] = len(active_tracks) - 1

            final_name = max(all_tracks[track_id]["labels"], key=lambda k: all_tracks[track_id]["labels"][k])
            color = COLOR_MAP.get(final_name, (128, 128, 128))
            text = f"T{track_id}:{final_name}:{d['score']:.2f}"
            draw_label(frame, x1, y1, x2, y2, text, color)
            labeled.append({
                "x1": d["x1"],
                "y1": d["y1"],
                "x2": d["x2"],
                "y2": d["y2"],
                "score": d["score"],
                "track_id": track_id,
                "label": final_name,
                "similarity": sim,
            })

        writer.write(frame)
        recognized_records.append({"frame": frame_index, "detections": labeled})

    cap.release()
    writer.release()

    with open(args.output_dir / "recognized_detections.jsonl", "w") as f:
        for rec in recognized_records:
            f.write(json.dumps(rec) + "\n")

    track_summary = []
    for track_id, t in all_tracks.items():
        final_name = max(t["labels"], key=lambda k: t["labels"][k])
        track_summary.append({
            "track_id": track_id,
            "label": final_name,
            "frames": [min(t["frames"]), max(t["frames"])],
            "detections": len(t["frames"]),
            "avg_similarity": round(sum(t["sims"]) / len(t["sims"]), 4),
        })
    with open(args.output_dir / "tracks.json", "w") as f:
        json.dump({"tracks": track_summary}, f, indent=2)

    print(f"Wrote annotated video to {out_video_path}")
    print(f"Wrote labels to {args.output_dir / 'recognized_detections.jsonl'}")
    print(f"Wrote track summary to {args.output_dir / 'tracks.json'} ({len(track_summary)} tracks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
