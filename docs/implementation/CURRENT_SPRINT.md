# MergenVision Phase 2 — Sprint 03

## Objective

Repair and harden the Sprint 01 detector foundation inside the Sprint 02 monorepo layout. Make the repository reproducible on a fresh clone, make the Python → Docker → native DeepStream chain deterministic and correct, and fix CUDA/plugin/subprocess lifecycle issues before any tracker/recognition work.

## Out of scope

- NvDCF/ByteTrack tracker correctness fix (tracker may stay disabled or pass-through only).
- ArcFace/GlintR100 recognition and CUDA face alignment.
- Gallery enrollment/search and canonical reconciliation.
- Annotated video rendering.
- FastAPI endpoints, PostgreSQL, MinIO, Qdrant, SSE/WebSocket.
- Multi-GPU scheduling beyond host GPU selection.
- Model download, engine rebuild, Git history rewrite.
- Parallel GPU NMS redesign; correctness and determinism only.

## Deliverables

1. **Canonical artifact layout** under `backend/artifacts/` with all stale root paths removed.
2. **`make artifacts-check`** — mandatory non-zero exit when required artifacts are missing.
3. **Artifact and stale-path regression tests**.
4. **Native worker/plugin reproducibility**:
   - plugin directory isolated (`backend/native/build/gst-plugins/`),
   - no undefined symbols in native `.so` files,
   - lifecycle tests for create/start/EOS/stop/finalize.
5. **Python control-plane boundary fixes**:
   - `SubprocessNativeWorkerAdapter`/`NativeDetectorClient` GPU assignment consistent,
   - domain constructors free of filesystem side effects,
   - subprocess timeout/cancellation cleanup with `-W error` clean.
   - stricter worker result protocol (exit code vs summary).
6. **CUDA correctness**:
   - async D2H ordering bug fixed,
   - deterministic argsort with anchor-id tie-break,
   - engine tensor contract validation at init.
7. **Hot-path boundary**:
   - JSONL write moved out of tracker pad-probe to a writer queue/thread,
   - no full-frame D2H, no bulk detector-output D2H.
8. **Make targets**:
   - `artifacts-check`, `backend-unit-strict`, `backend-native-build`, `backend-native-linkcheck`, `backend-native-unit`, `backend-detector-parity`, `backend-detector-determinism`, `backend-hotpath`, `backend-cli-smoke`, `backend-video-smoke`, `frontend-test`, `frontend-build`, `phase2-sprint-01-acceptance`, `sprint-02-acceptance`, `phase2-foundation-acceptance`.
9. **Review package**: `docs/implementation/review_packages/SPRINT-003-CODE-REVIEW-PACKAGE.md`.

## Acceptance

1. `make artifacts-check` passes with all required artifacts present and SHA-256 verified.
2. `make backend-unit-strict` passes with `-W error`.
3. `make backend-native-build` and `make backend-native-linkcheck` pass (no undefined symbols, `gst-inspect-1.0 nvdsretinaface` succeeds).
4. `make backend-native-unit` passes.
5. `make backend-detector-parity` matches CPU oracle within tolerance on all processed frames.
6. `make backend-detector-determinism` produces identical normalized output for 20 runs.
7. `make backend-hotpath` proves no full-frame/detector-output D2H and no per-frame cudaMalloc/cudaFree.
8. `make backend-cli-smoke` and `make backend-video-smoke` pass with expected frame/detections counts.
9. `make frontend-test` and `make frontend-build` pass (real build).
10. `make phase2-foundation-acceptance` aggregated command passes end to end.
11. Stale root paths are gone from production/config/test/docs (verified by `backend/tests/integration/test_stale_paths.py`).
12. `git diff --check` passes; no video/model/engine in Git; no raw customer data logged.
13. No git commit/push.

## Status

COMPLETED

### Completed acceptance gates

- [x] `make artifacts-check` passes with all required artifacts present and SHA-256 verified.
- [x] `make backend-unit-strict` passes with `-W error`.
- [x] `make backend-cli-smoke` passes with expected frame/detections counts.
- [x] `make frontend-test` passes.
- [x] `make frontend-build` passes.
- [x] `make backend-native-build`, `backend-native-linkcheck`, `backend-native-unit` pass.
- [x] `make backend-detector-parity` and `make backend-detector-determinism` pass.
- [x] `make backend-hotpath` passes (Nsight Systems D2H contract verified).
- [x] `make backend-video-smoke` passes (Friends.mp4, 6665 frames, 8977 detections).
- [x] `make phase2-foundation-acceptance` aggregated command passes.
- [x] Stale root paths are gone from production/config/test/docs (verified by `backend/tests/integration/test_stale_paths.py`).
- [x] `git diff --check` passes.

### Remaining work

- [ ] Review package `docs/implementation/review_packages/SPRINT-003-CODE-REVIEW-PACKAGE.md` (documentation-only follow-up).
