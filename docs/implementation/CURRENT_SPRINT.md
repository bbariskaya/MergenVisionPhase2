# MergenVision Phase 2 — Sprint 02

## Objective

Restructure the repository into a clean monorepo with backend/frontend separation, move all production Python and native GPU source under `backend/`, and establish a typed Python control-plane → native-worker boundary. No algorithmic changes.

## Out of scope

- RetinaFace/NMS algorithm changes.
- NvDCF tracker fix.
- Recognition, reconciliation, gallery, rendering.
- FastAPI endpoints, PostgreSQL, MinIO, Qdrant, SSE.
- Multi-GPU scheduling.
- Moving large artifact/data folders (`models/`, `engines/`, `DATASET/`, `test_videos/`, `out/`).
- Deleting generated build/output artifacts.

## Deliverables

- `backend/` production tree:
  - `backend/app/domain/native_job.py` — request/progress/result/error models.
  - `backend/app/ports/native_worker.py` — `NativeWorkerPort` protocol.
  - `backend/app/application/services/run_video_detection.py` — port-driven use case.
  - `backend/app/infrastructure/native_worker/subprocess_adapter.py` — concrete subprocess/Docker adapter.
  - `backend/app/cli.py` — thin CLI (`python -m app.cli detect ...`).
- `backend/native/` — native GPU source moved from root `native/`.
- `backend/tests/` — unit, integration, layout, and native tests.
- `backend/scripts/` — production/job scripts.
- `backend/tools/` — developer/benchmark/visualization tools.
- Updated build/test paths (CMake, Makefile, scripts).
- New Make targets:
  - `make backend-unit`
  - `make backend-native-build`
  - `make backend-native-smoke`
  - `make frontend-test`
  - `make sprint-02-acceptance`
- `README.md` and `backend/README.md` with dependency diagrams.
- Review package at `docs/implementation/review_packages/SPRINT-002-CODE-REVIEW-PACKAGE.md`.

## Acceptance

1. All production Python and native GPU source lives under `backend/`.
2. `frontend/` contains only frontend code.
3. Root has no production `.py/.cpp/.cu` source (generated `native/build` left in place and reported).
4. Application → Port → Subprocess adapter call chain works.
5. Native build succeeds in the new path.
6. Existing detector smoke behavior unchanged.
7. Backend unit/layout tests pass.
8. Frontend test/build passes.
9. No machine-specific `/home/user/...` production paths.
10. `git diff --check` passes.
11. No git commit/push.

## Acceptance command

```bash
make sprint-02-acceptance
```

## Status

PASSED — review package at `docs/implementation/review_packages/SPRINT-002-CODE-REVIEW-PACKAGE.md`.
