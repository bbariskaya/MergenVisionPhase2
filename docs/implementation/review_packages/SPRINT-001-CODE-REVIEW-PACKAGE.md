# Sprint 01 Code Review Package

**Objective:** Make the RetinaFace detector vertical slice mathematically correct, deterministic, reproducible, GPU-only on the hot path, and cleanly callable from Python.

**Acceptance command:** `make phase2-sprint-01-acceptance`

**Status:** PASSED

## Frozen architectural decision maintained

Hybrid Python/C++ boundary is preserved:

- Python control plane: request validation, job orchestration, caller/result parsing, structured logging.
- Native C++/CUDA data plane: GStreamer/DeepStream, NVDEC, NVMM, TensorRT detector, CUDA decode/NMS, compact metadata emission to JSONL.

## Changes delivered

1. **Correct normalized IoU CUDA NMS**
   - `native/kernels/nms.cu` uses continuous-coordinate IoU (no `+1`) and exact sequential NMS.
   - Invalid/zero-area boxes are skipped and do not suppress valid boxes.

2. **Deterministic CUDA argsort**
   - `native/kernels/argsort.cu` sorts `(score desc, original_index asc)` via an explicit comparator.
   - `native/tests/test_nms.cu` verifies the lowest original index wins a 100-way score tie.

3. **Unit tests**
   - `native/tests/test_nms.cu`: IoU basics, single-image NMS, deterministic tie-break.
   - `tests/native/test_native_worker_client.py`: Python caller command construction and result parsing (no Docker).

4. **Detector parity test vs CPU ONNX Runtime oracle**
   - `tests/fixtures/cpu_oracle/cpu_oracle_retinaface.py`: OpenCV decode, ONNX Runtime inference, reference decode/NMS.
   - `tests/native/test_detector_parity.py`: Compares native worker output for frame 0 of `friendsshort_50f.mp4` against the CPU oracle (4 vs 4 detections, 100% matched).

5. **Thin Python native worker caller**
   - `tools/native_detector_client.py`: Builds/constructs Docker commands and parses structured worker metadata.

6. **Reproducible Make targets**
   - `Makefile`: `phase2-sprint-01-build|unit|parity|hotpath|video-smoke|acceptance`.
   - Captures GPU UUID, CUDA/TensorRT/GStreamer versions, and engine SHA-256 into `docs/implementation/review_packages/sprint-01-environment.txt`.

## Acceptance evidence

```bash
$ make phase2-sprint-01-acceptance
...
All NMS tests PASSED
...
Frame 0 parity OK (4 matches)
...
PASS                                            # GPU hot-path contract
OK: 50 frames, 25 detections                    # video-smoke 50f
OK: 1087 frames, 1658 detections                # video-smoke full
Sprint 01 acceptance PASSED
```

## Runtime inventory

See `docs/implementation/review_packages/sprint-01-environment.txt`:

```
GPU 0: Quadro RTX 8000 (UUID: GPU-22f04c0c-b6c2-9b75-6f9d-7e08fa7d7537)
nvcc: Cuda compilation tools, release 13.1, V13.1.115
libnvinfer-dispatch-dev 10.14.1.48-1+cuda13.0
gst-inspect-1.0 version 1.24.2
SHA256(engines/retinaface_r50_dynamic.bs1.opt64.max256.fp16.trt1014.engine) = 6563c70086bc08fe7d30b60b36d99a410a2bff36cdbc40da539a43a32cbf0e17
```

## Files touched

- `native/kernels/nms.cu`
- `native/kernels/argsort.cu`
- `native/kernels/mergenvision_kernels.h`
- `native/tests/test_nms.cu`
- `native/CMakeLists.txt`
- `Makefile` (new)
- `tools/native_detector_client.py` (new)
- `tests/native/test_native_worker_client.py` (new)
- `tests/fixtures/cpu_oracle/cpu_oracle_retinaface.py` (new)
- `tests/native/test_detector_parity.py` (new)
- `docs/implementation/review_packages/sprint-01-environment.txt` (new)
- `docs/implementation/review_packages/SPRINT-001-CODE-REVIEW-PACKAGE.md` (new)

## Non-goals respected

- NvDCF/tracker fix (raw tracker ID still `UNTRACKED`; no attempted tracker fix).
- Python spatial tracker.
- CUDA alignment / GlintR100 native integration.
- Gallery recognition / canonical reconciliation.
- Annotated video rendering.
- FastAPI endpoints, PostgreSQL, MinIO, Qdrant, SSE, frontend.
- Multi-GPU scheduling.

## Known warnings under investigation

The GStreamer plugin scanner emits a warning about `libretinaface_parser.so: undefined symbol: cudaPointerGetAttributes`. This is a plugin discovery warning, not a runtime failure; the worker pipeline completes with `exit_code=0` and is tracked as a cleanup item for Sprint 02.
