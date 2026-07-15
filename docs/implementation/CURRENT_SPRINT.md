# MergenVision Phase 2 — Sprint 05

## Sprint 04 closure status

- Detector batching + native render: **PASS**.
- `make phase2-sprint-04-acceptance` exit 0.
- Review package: `docs/implementation/review_packages/SPRINT-004-CODE-REVIEW-PACKAGE.md`.
- NvDCF raw tracker ID assignment: **KNOWN_BROKEN / DEFERRED** (all detections receive `UNTRACKED_OBJECT_ID`).

---

## Objective

Deliver the first complete user-visible native GPU recognition vertical slice:

```text
encoded MP4
  -> NVDEC / nvstreammux temporal batch
  -> RetinaFace batched face detection
  -> CUDA five-point landmark alignment
  -> batched TensorRT GlintR100 embedding
  -> GPU L2 normalization
  -> compact embedding/metadata CPU boundary
  -> gallery cosine matching
  -> nvdsosd / NVENC annotated MP4 with name + similarity + detector confidence
```

Primary user outcome: a single CLI command produces a playable annotated MP4 where each face box shows either a gallery name + cosine similarity + detector score, or `unknown`.

---

## Out of scope

- FastAPI endpoints, PostgreSQL, MinIO, Qdrant.
- Frontend / UI.
- Multi-GPU scheduling beyond host GPU selection.
- Livestream / RTSP / HLS.
- Canonical cross-scene identity reconciliation.
- Tracker batch>1 correctness fix (remains deferred).
- GPU gallery search optimization (CPU matching is sufficient for the first slice).
- Detector re-calibration / model swap.
- Python/OpenCV/NumPy production decode, alignment, or inference fallback.

---

## Deliverables

1. Native worker support for `--mode fast` and `--mode tracked` with explicit `--gallery <path>` and `--threshold`/`--margin` overrides.
2. DeepStream 9.0 recognition path using `nvdspreprocess` SGIE/object mode + custom five-point tensor-preparation library, feeding standard `nvinfer` with the existing GlintR100 TensorRT engine. If the variable face-count contract is not provably correct, fallback to a dedicated `gst-nvdsfacerecognizer` element using the `gst-nvdsvisionencoder-c` structure as reference.
3. Pitched RGBA/NVMM CUDA alignment kernel with explicit frame surface pointer, pitch, batch index, per-face affine matrix, bilinear interpolation, and RGB NCHW output.
4. GPU L2 normalization with finite/zero-norm handling.
5. CPU C++ gallery loader/matcher with top1/top2 cosine + margin decision; unknown remains unknown.
6. Owned custom metadata `mv-face-recognition` attached via copy/release callbacks.
7. OSD label format:
   - fast mode: `{name} | sim:{s} | det:{c}`
   - tracked mode: `T{id} | {name} | sim:{s} | det:{c}`
   - unknown: `unknown | sim:{s} | det:{c}`
8. Native GPU annotated MP4 render (NVENC, qtmux) for fast mode and tracked validation mode.
9. JSONL recognition metadata: frame/pts/bbox/landmarks/detector score/raw track ID/identity ID/name/status/top1/top2/margin.
10. Updated Python layered backend (`app.cli annotate ...`) with mode, batch-size, gallery, threshold/margin, annotated output, and render toggles.
11. Targeted gates/tests for artifact/engine, landmark order, alignment parity, preprocess parity, engine parity, face batch parity, gallery, detector regression, short E2E, tracked E2E, long E2E, hot-path, determinism.
12. Updated `Makefile` Sprint 05 targets and `phase2-sprint-05-acceptance` aggregate command.
13. Updated `docs/implementation/IMPLEMENTATION_DETAILS.md` and `docs/implementation/REFERENCE_DECISION_LOG.md`.
14. Immutable review package: `docs/implementation/review_packages/SPRINT-005-CODE-REVIEW-PACKAGE.md`.

---

## Acceptance

1. `friendsshort_recognized_annotated.mp4` and `Friends_recognized_annotated.mp4` are playable and show real gallery names or honest `unknown`.
2. `recognized_detections.jsonl` and `run_manifest.json` are produced with the fields listed above.
3. Alignment/contact-sheet artifact is generated and inspected.
4. Single native GPU pass; no Python/OpenCV production decode/alignment/inference.
5. No full decoded frame D2H; only compact metadata/normalized embeddings cross to CPU.
6. Output frame count and duration match the input video.
7. Fast mode uses detector batch-size=8, tracker off, render on, and is faster than realtime (target ≥100 FPS effective).
8. Tracked validation mode produces real raw tracker IDs; any remaining `UNTRACKED_OBJECT_ID` causes FAIL.
9. `make phase2-sprint-05-acceptance` exits 0.

---

## Non-goals

- Production biometric calibration claim (label as `demo_uncalibrated_threshold` if needed).
- Tracker/canonical cross-scene reconciliation.
- GPU gallery search optimization.

---

## Status

IN_PROGRESS — Sprint 05 implementation starting.
