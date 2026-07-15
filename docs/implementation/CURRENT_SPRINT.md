# MergenVision Phase 2 — Sprint 04

## Sprint 03 prerequisite status: PASS/COMPLETED

- `make phase2-foundation-acceptance` passed (exit 0, rerun log at `backend/out/sprint03_foundation_rerun.log`).
- `git status --short` reviewed; all Sprint 03 changes are preserved.
- No unresolved Sprint 03 blocker remains.

See `docs/implementation/review_packages/SPRINT-003-CODE-REVIEW-PACKAGE.md` for the final Sprint 03 evidence package.

---

## Objective

Implement true temporal batched RetinaFace inference and a native GPU annotated-video pipeline.

Primary user outcome: process a local MP4 significantly faster than real time with:

```text
NVDEC → nvstreammux temporal batching → nvdspreprocess → one TensorRT enqueue per actual batch
→ batched CUDA decode/NMS/landmark postprocess → per-frame ordered metadata
→ optional nvdsosd + NVENC + qtmux → playable annotated MP4
```

This is a detector batching + native rendering sprint. Tracker, recognition, identity names and canonical reconciliation are NOT in scope.

## Out of scope

- NvDCF/ByteTrack tracker correctness fix (tracker stays `off` for the batched/render path; the old batch-1 tracker-capable path is preserved but not exercised by new Sprint 04 acceptance).
- ArcFace/GlintR100 recognition and five-point CUDA alignment.
- Gallery enrollment, search, canonical reconciliation.
- FastAPI endpoints, PostgreSQL, MinIO, Qdrant.
- Multi-GPU scheduling beyond host GPU selection.
- Livestream/RTSP/HLS, audio preservation.
- CPU/OpenCV/NumPy production rendering or inference fallback.

## Deliverables

1. Dynamic engine contract validation (tensor names, shapes, FP32 input, optimization profile supports 1/4/8/16).
2. `make phase2-sprint-04-feasibility` proving single-source temporal batches > 1 and partial-EOS flushing on DeepStream 9.0.
3. Batched `RetinaFacePostproc` and plugin transform that performs one TensorRT enqueue per actual batch.
4. Frame/tensor mapping using `NvDsFrameMeta.batch_id` with validation and synthetic tests.
5. Worker architecture split: `options`, `detector_pipeline`, `batch_contract`, `metadata_writer`, `annotated_video_sink`.
6. Optional GPU-native annotated MP4 branch (`--annotated-output`) using `nvdsosd` + `nvv4l2h264enc` + `qtmux`.
7. Updated Python `NativeDetectorClient` with batch/render options and validation tests.
8. Immutable batch-1 baseline capture under `backend/out/sprint-04/baseline_batch1/`.
9. Sprint 04 make targets:
   - `phase2-sprint-04-feasibility`
   - `phase2-sprint-04-build`
   - `phase2-sprint-04-unit`
   - `phase2-sprint-04-eos`
   - `phase2-sprint-04-batch-parity`
   - `phase2-sprint-04-determinism`
   - `phase2-sprint-04-render`
   - `phase2-sprint-04-hotpath`
   - `phase2-sprint-04-benchmark`
   - `phase2-sprint-04-acceptance`
10. Final review package: `docs/implementation/review_packages/SPRINT-004-CODE-REVIEW-PACKAGE.md`.

## Acceptance

1. Sprint 03 foundation remains green through the entire Sprint 04 work.
2. Feasibility gate proves actual temporal batching with correct `batch_id` mapping and EOS partial flushing.
3. True batch inference: `enqueue_count < processed_frame_count` for `batch-size > 1`.
4. Batch 4 works; batch 8 and 16 each pass or are honestly rejected with evidence.
5. Batch-1 vs selected-batch semantic parity passes under frozen gates.
6. EOS behavior verified for frame counts 1, 3, 4, 5, 15, 16, 17, 50.
7. No `nvtracker` is instantiated in the batch/render pipeline.
8. Annotated MP4 is playable, has H.264 stream, matching duration, and no fake tracker IDs or names.
9. Hot-path contract passes (no full-frame D2H, no full detector tensor D2H, no per-frame cudaMalloc/free, no CPU postprocess).
10. Benchmark report contains real measurements and selects one runtime batch size from data.
11. `git diff --check` passes; no video/model/engine committed.
12. No git commit/push.

## Chosen approach

- **Integrated render (A):** the worker optionally adds `nvdsosd → nvv4l2h264enc → qtmux → filesink` to the same pipeline so metadata and annotated MP4 are produced in one pass.
- Old batch-1 tracker-capable code path remains but is not used for Sprint 04 acceptance.
- `nvdspreprocess` `network-input-shape` will be generated per configured max batch at runtime; TensorRT will run with the actual batch size via `setInputShape`.

## Status

IN_PROGRESS — Build mode approved.
