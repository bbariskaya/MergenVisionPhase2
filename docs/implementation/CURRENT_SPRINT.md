# MergenVision Phase 2 — Milestone A+B Sprint

**Objective:** Run `friendsshort.mp4` end-to-end through a DeepStream GPU pipeline (NVDEC → RetinaFace → NvDCF → alignment → GlintR100 → offline reconciliation), produce final canonical identity metadata, and render a fully annotated MP4.

**Deliverables:**
- DeepStream 9.0 worker Docker image with RetinaFace/GlintR100 engines rebuilt for TensorRT 10.14.
- Native C++ custom parser + landmark metadata plumbing.
- Per-frame `detections.jsonl`, `tracks.json`, and `result.json`.
- Offline reconciliation producing canonical identities.
- `out/friendsshort/annotated.mp4` with final labels.
- Parity and performance evidence files.

**Acceptance command:**

```bash
make deepstream-e2e VIDEO=test_videos/friendsshort.mp4 GPU=0 OUTPUT=out/friendsshort
```

**Non-goals:**
- Upload/streaming ingestion, API/SSE, PostgreSQL/MinIO/Qdrant persistence.
- Live RTSP/webcam.
- Generic object detection (YOLO) as production feature.
- Distributed multi-GPU job scheduling.
- git commits/pushes.
