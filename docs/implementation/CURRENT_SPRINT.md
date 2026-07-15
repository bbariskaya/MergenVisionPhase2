# MergenVision Phase 2 — Sprint 05

**Binding implementation contract overrides this draft:**
`docs/implementation/plans/SPRINT-005-IMPLEMENTATION-CONTRACT.md`

---

## Sprint 04 closure status

- Detector batching + native render: **PASS**.
- `make phase2-sprint-04-acceptance` exit 0.
- Review package: `docs/implementation/review_packages/SPRINT-004-CODE-REVIEW-PACKAGE.md`.
- NvDCF raw tracker ID assignment: **KNOWN_BROKEN / DEFERRED** (all detections receive `UNTRACKED_OBJECT_ID`).

Sprint 05 starts only after the above closure evidence is final (split verdict, buffer-pool default kept at `max(16, batch*2)`, diagnostic report recorded, aggregation bug fixed with unit test).

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

1. `friendsshort_recognized_annotated.mp4` and `Friends_recognized_annotated.mp4` are playable; faces show real gallery names or honest `unknown`.
2. `recognized_detections.jsonl` and `run_manifest.json` are produced per the JSONL/manifest contract in the binding plan.
3. Alignment/contact-sheet artifact is generated and inspected; alignment parity thresholds were frozen before evaluation.
4. Single native GPU pass; no Python/OpenCV production decode/alignment/inference.
5. No full decoded frame D2H; only compact embeddings/metadata cross to CPU (allowed D2H bytes logged in `run_manifest.json`).
6. Output frame count and duration match the input video; PTS monotonic; clean EOS.
7. Fast mode uses detector batch-size=8, tracker off, render on. Any fixed FPS target is measurement-only, not a pass/fail gate.
8. Tracker-on raw-ID correctness is **non-blocking** for Sprint 05; `phase2-sprint-05-acceptance` does not include the tracker diagnostic. The diagnostic target reports `PASS` or `KNOWN_BROKEN` separately.
9. Recognition decisions must pass frozen semantic gates: at least one expected known identity on friendsshort, not all unknown, not all collapsed to one identity, top1/top2/margin rules, duplicate/wrong-SHA gallery rejection.
10. Batch parity: batch=1 vs batch=8 produce the same detections, identity decisions, and similarity deltas within tolerance.
11. Determinism: short fixture repeated ≥3 times yields identical semantic results (ordered by frame/PTS).
12. Hot-path evidence shows GPU-resident decode, CUDA alignment, TRT GlintR100, CUDA L2 norm, no per-face synchronize, no per-buffer cudaMalloc/free.
13. `make phase2-sprint-05-acceptance` exits 0 **and** the aggregate target invokes the full mandatory acceptance chain (not just a single exit-code check).

---

## Non-goals

- Production biometric calibration claim (label as `demo_uncalibrated_threshold` if needed).
- Tracker/canonical cross-scene reconciliation.
- GPU gallery search optimization.

---

## Status

IN_PROGRESS — Sprint 05 implementation starting.

Sprint 06 binding implementation contract has been staged at `docs/implementation/plans/SPRINT-006-IMPLEMENTATION-CONTRACT.md` and is frozen for activation only after Sprint 05 PASS.
