# Sprint 02 Code Review Package

**Objective:** Restructure the repository into a clean `backend/frontend` monorepo, move all production Python and native GPU source under `backend/`, and establish a typed Python control-plane → native-worker boundary. No algorithmic changes.

**Acceptance command:** `make sprint-02-acceptance`

**Status:** PASSED

## Frozen architectural decision maintained

Hybrid Python/C++ boundary is preserved and made explicit in the file tree:

- **Python control plane:** FastAPI (future), job orchestration, cancellation/progress, storage, reconciliation, aggregation, structured logging.
- **Native C++/CUDA data plane:** GStreamer/DeepStream, NVDEC, NVMM, TensorRT detector, CUDA decode/NMS, compact metadata emission.

## Changes delivered

1. **Monorepo layout**
   - Moved production native source from root `native/` to `backend/native/`.
   - Moved production/test/tool Python from root `tests/`, `scripts/`, `tools/` to `backend/tests/`, `backend/scripts/`, `backend/tools/`.
   - Kept `frontend/` untouched.
    - Consolidated all runtime artifacts under `backend/artifacts/`:
      - `backend/artifacts/engines/` — TensorRT engines (kept the built `bs1.opt*.engine` variants).
      - `backend/artifacts/models/` — ONNX and `.pt` weights.
      - `backend/artifacts/gallery/` — merged the previous root gallery directory and `data/gallery/`.
      - `backend/artifacts/videos/` — moved from the previous root video directory.
      - `backend/artifacts/annotations/` — moved from `data/annotations/`.
    - Moved generated outputs to the canonical `backend/out/` directory.
   - Removed the leftover root `native/build/` directory.

2. **Updated native build paths**
   - `backend/native/CMakeLists.txt` now expects source under `backend/native`.
   - Native worker config defaults updated to `/app/backend/native/configs/...` with a repo-relative fallback `backend/native/configs/...`.
   - `Makefile` Docker mounts use `GST_PLUGIN_PATH=/app/backend/native/build` and worker binary path `/app/backend/native/build/deepstream_face_worker`.

3. **Artifact ignore rules**
   - Updated `.gitignore` so `backend/artifacts/*`, `backend/out/`, `backend/native/build/`, and common large file patterns (`*.engine`, `*.onnx`, `*.pt`) are not tracked.
   - Added `backend/artifacts/.gitkeep` as a directory marker.

4. **Use the built optimized engines**
   - Native worker default engine: `backend/artifacts/engines/retinaface_r50_dynamic.bs1.opt64.max256.fp16.trt1014.engine`.
   - Tool defaults and test harness constants point at the optimized `glintr100.bs1.opt128.max256.fp16.trt1014.engine`.

5. **New Python control-plane boundary**
   - `backend/app/domain/native_job.py` — request, progress, result, and error models.
   - `backend/app/ports/native_worker.py` — `NativeWorkerPort` protocol (async, typed).
   - `backend/app/application/services/run_video_detection.py` — domain use-case that drives the port.
   - `backend/app/infrastructure/native_worker/client.py` — Docker command builder.
   - `backend/app/infrastructure/native_worker/subprocess_adapter.py` — concrete subprocess adapter with timeout and cancellation.
   - `backend/app/cli.py` — thin CLI (`python -m app.cli detect ...`) for manual/ad-hoc runs.

4. **Tests**
   - `backend/tests/unit/test_domain_native_job.py`
   - `backend/tests/unit/test_run_video_detection_service.py`
   - `backend/tests/unit/test_subprocess_adapter.py`
   - `backend/tests/integration/test_layout.py` — verifies production source is under `backend/`, frontend remains separate, and domain layer does not import infrastructure.

5. **Build/orchestration targets**
   - `Makefile` targets:
     - `make backend-unit`
     - `make backend-native-build`
     - `make backend-native-smoke`
     - `make frontend-test`
     - `make sprint-02-acceptance`

6. **Documentation**
   - Root `README.md` with monorepo map and dependency diagram.
   - `backend/README.md` with Python layer descriptions and architecture diagram.

## Acceptance evidence

```bash
$ make sprint-02-acceptance
...
18 passed in 0.58s                         # backend-unit
...
Done. frames=50 detections=25 wall=1.31s error=0  # backend-native-smoke
OK: 50 frames, 25 detections
...
Test Files  9 passed (9)                   # frontend-test
Tests 39 passed (39)
...
tsc -b                                     # frontend typecheck/build
...
git diff --check                           # passed
Sprint 02 acceptance PASSED
```

## Runtime inventory

Host/container environment unchanged from Sprint 01:

```
GPU 0: Quadro RTX 8000 (UUID: GPU-22f04c0c-b6c2-9b75-6f9d-7e08fa7d7537)
nvcc: Cuda compilation tools, release 13.1, V13.1.115
libnvinfer-dispatch-dev 10.14.1.48-1+cuda13.0
gst-inspect-1.0 version 1.24.2
SHA256(engines/retinaface_r50_dynamic.bs1.opt64.max256.fp16.trt1014.engine) = 6563c70086bc08fe7d30b60b36d99a410a2bff36cdbc40da539a43a32cbf0e17
```

## Files touched

- `.gitignore` (updated)
- `Makefile` (replaced)
- `README.md` (new)
- `docs/implementation/CURRENT_SPRINT.md`
- `backend/artifacts/.gitkeep` (new)
- `backend/README.md` (new)
- `backend/pyproject.toml` (new)
- `backend/app/__init__.py` (new)
- `backend/app/domain/__init__.py` (new)
- `backend/app/domain/native_job.py` (new)
- `backend/app/ports/__init__.py` (new)
- `backend/app/ports/native_worker.py` (new)
- `backend/app/application/__init__.py` (new)
- `backend/app/application/services/__init__.py` (new)
- `backend/app/application/services/run_video_detection.py` (new)
- `backend/app/infrastructure/__init__.py` (new)
- `backend/app/infrastructure/native_worker/__init__.py` (new)
- `backend/app/infrastructure/native_worker/client.py` (new)
- `backend/app/infrastructure/native_worker/subprocess_adapter.py` (new)
- `backend/app/cli.py` (new)
- `backend/native/CMakeLists.txt` (moved + updated)
- `backend/native/kernels/*`
- `backend/native/plugins/*`
- `backend/native/parsers/*`
- `backend/native/worker/main.cpp` (moved + paths updated)
- `backend/native/tools/render_annotated_video.cpp` (moved + paths updated)
- `backend/native/tests/*`
- `backend/scripts/*`
- `backend/tests/*`
- `backend/tools/*`

## Non-goals respected

- RetinaFace/NMS algorithm unchanged.
- NvDCF tracker behavior unchanged.
- Recognition, reconciliation, gallery, rendering not touched.
- FastAPI endpoints, PostgreSQL, MinIO, Qdrant, SSE not added.
- Multi-GPU scheduling not added.
- No git commit/push.

## Known warnings under investigation

The GStreamer plugin scanner continues to emit a warning about `libretinaface_parser.so: undefined symbol: cudaPointerGetAttributes`. It remains a plugin-discovery warning; the pipeline completes with `exit_code=0`. This is recorded for a future cleanup sprint and does not affect detector functional behavior.
