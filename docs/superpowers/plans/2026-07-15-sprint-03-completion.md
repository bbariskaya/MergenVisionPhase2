# Sprint 03 Acceptance Completion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or implement tasks inline in this session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Sprint 03 acceptance gates from `docs/implementation/CURRENT_SPRINT.md` actually pass by implementing the missing Makefile targets, fixing the one known native dynamic-symbol issue, and adding the missing detector-determinism test.

**Architecture:** Existing Python control-plane fixes and native build already work. We only need to surface and execute the native/hot-path correctness checks that CURRENT_SPRINT expects: native `.so` link-check, native NMS unit test, detector parity against CPU oracle, detector determinism across repeated runs, GPU hot-path contract via Nsight Systems, and a longer video smoke. The plan is build → harden link → run native unit tests → parity → determinism → hot-path → video smoke → aggregate acceptance.

**Tech Stack:** Docker, DeepStream 9.0, GStreamer, CMake, CUDA 13.x, TensorRT 10.14, Python 3.11, pytest, Nsight Systems, ONNX Runtime (CPU oracle only used directly by parity script).

## Global Constraints

- Do NOT commit or push.
- Do NOT change the public API contract for video jobs.
- Keep tracker enabled (current smoke configuration); tracker correctness is explicitly out-of-scope, but the existing tracker must not be removed.
- No full-frame or bulk detector-output D2H in the hot-path (already enforced by design; prove with `nsys`).
- All new scripts live under `backend/scripts/` or `backend/tests/native/` and must be runnable with the host Python.
- All GPU targets pin to `--gpus device=$(GPU_DEVICE)` and map the repo to `/app` inside the container.
- Deterministic output means face geometry/score must repeat; raw tracker IDs may legitimately differ between NvDCF instantiations, so compare normalized detection lists per frame, not track IDs.

---

## Task 1: Fix undefined CUDA symbol in `libretinaface_parser.so`

**Files:**
- Modify: `backend/native/CMakeLists.txt:54-73`

**Problem:** GStreamer plugin scanner emits `undefined symbol: cudaPointerGetAttributes` when loading `libretinaface_parser.so` because `retinaface_parser.cpp` calls `cudaPointerGetAttributes()` but the shared library does not link `CUDA::cudart`.

- [ ] **Step 1: Add CUDA runtime to parser link**

```cmake
add_library(retinaface_parser SHARED
    parsers/retinaface_parser.cpp
)
target_include_directories(retinaface_parser PRIVATE ${DEEPSTREAM_INCLUDES})
target_link_libraries(retinaface_parser CUDA::cudart)
```

- [ ] **Step 2: Rebuild**

Run: `make backend-native-build`
Expected: `libretinaface_parser.so` builds without errors.

- [ ] **Step 3: Verify symbol is defined**

Run:
```bash
docker run --rm -v "$(pwd):/app" -w /app --entrypoint nm nvcr.io/nvidia/deepstream:9.0-triton-multiarch \
  -D /app/backend/native/build/libretinaface_parser.so | grep cudaPointerGetAttributes
```
Expected: symbol shows `U` resolved if run against a full environment, and the `undefined symbol` GStreamer warning disappears in the next smoke.

---

## Task 2: Add `backend-native-linkcheck` target

**Files:**
- Create: `backend/scripts/native_linkcheck.py`
- Modify: `Makefile`

- [ ] **Step 1: Write link-check script**

The Python script checks every `.so` in `backend/native/build/` that the project owns (`libgstnvdsretinaface.so`, `libretinaface_parser.so`). For each:
1. Run `nm -D <so>` and fail if any non-weak `U` (undefined) symbol remains that is not provided by glibc/GLIBC or by installed DeepStream/NVIDIA libraries.
2. Run `gst-inspect-1.0 nvdsretinaface` and fail if it returns non-zero or prints the old undefined-symbol warning.

Exit 0 if clean, non-zero otherwise.

- [ ] **Step 2: Add Makefile target**

```makefile
backend-native-linkcheck:
	@echo "=== Backend native link check ==="
	python3 backend/scripts/native_linkcheck.py
```

- [ ] **Step 3: Run target**

Run: `make backend-native-linkcheck`
Expected: PASS with no undefined symbols in our `.so` files.

---

## Task 3: Add `backend-native-unit` target

**Files:**
- Modify: `Makefile`

The executable `backend/native/build/test_nms` is already built by `backend-native-build`. We only need a target that runs it inside the DeepStream container.

- [ ] **Step 1: Add Makefile target**

```makefile
backend-native-unit:
	@echo "=== Backend native unit tests ==="
	$(DOCKER_RUN) --entrypoint /app/backend/native/build/test_nms $(CONTAINER)
```

- [ ] **Step 2: Run target**

Run: `make backend-native-unit`
Expected: `All NMS tests PASSED`.

---

## Task 4: Add `backend-detector-parity` target

**Files:**
- Modify: `Makefile`
- Already exists: `backend/tests/native/test_detector_parity.py`

- [ ] **Step 1: Add Makefile target**

```makefile
DETECTOR_PARITY_RUN := /app/backend/out/detector_parity_run

backend-detector-parity: backend-native-build
	@echo "=== Detector parity vs CPU oracle ==="
	$(DOCKER_RUN) --entrypoint $(WORKER) $(CONTAINER) $(TEST_VIDEOS)/friendsshort_50f.mp4 $(DETECTOR_PARITY_RUN) 0
	$(DOCKER_RUN) --entrypoint python3 $(CONTAINER) /app/backend/tests/native/test_detector_parity.py
```

- [ ] **Step 2: Run target**

Run: `make backend-detector-parity`
Expected: script prints `Frame 0 parity OK` and exits 0.

---

## Task 5: Add `backend-detector-determinism` target

**Files:**
- Create: `backend/tests/native/test_detector_determinism.py`
- Modify: `Makefile`

The test runs the native worker three times on `friendsshort_50f.mp4` and compares normalized detections per frame (ignoring track IDs and order). It exits 0 if all runs produce the same geometry/score within tolerance.

- [ ] **Step 1: Create determinism test**

Pseudo-behavior:
1. Delete old run directories.
2. For `i` in `0..2`:
   - Run worker to `backend/out/detector_determinism_run_{i}`.
3. Load each `detections.jsonl`.
4. For each frame, normalize detection list: sort by `(x1, y1, score)`, round coordinates and score to 3 decimals.
5. Assert all three runs produce identical normalized lists for every frame.

- [ ] **Step 2: Add Makefile target**

```makefile
backend-detector-determinism: backend-native-build
	@echo "=== Detector determinism repeated run check ==="
	$(DOCKER_RUN) --entrypoint python3 $(CONTAINER) /app/backend/tests/native/test_detector_determinism.py
```

- [ ] **Step 3: Run target**

Run: `make backend-detector-determinism`
Expected: 3 runs complete and normalized outputs match.

---

## Task 6: Add `backend-hotpath` target

**Files:**
- Modify: `Makefile`
- Already exists: `backend/tests/native/test_gpu_hot_path_contract.py`

- [ ] **Step 1: Add Makefile target**

```makefile
backend-hotpath: backend-native-build
	@echo "=== GPU hot-path contract (Nsight Systems) ==="
	python3 -m pytest backend/tests/native/test_gpu_hot_path_contract.py -v
```

- [ ] **Step 2: Run target**

Run: `make backend-hotpath`
Expected: pytest reports `test_no_full_output_tensor_d2h_and_clean_exit` PASSED and prints `PASS`.

---

## Task 7: Add `backend-video-smoke` target

**Files:**
- Modify: `Makefile`

Use the longer `Friends.mp4` for a more realistic smoke run. Re-use `sanity_check_detections.py`, but allow a higher detection count.

- [ ] **Step 1: Add Makefile target**

```makefile
VIDEO_SMOKE_RUN := /app/backend/out/video_smoke_run

backend-video-smoke: backend-native-build
	@echo "=== Backend video smoke (longer clip) ==="
	$(DOCKER_RUN) --entrypoint $(WORKER) $(CONTAINER) $(TEST_VIDEOS)/Friends.mp4 $(VIDEO_SMOKE_RUN) 0
	$(DOCKER_RUN) --entrypoint python3 $(CONTAINER) /app/backend/scripts/sanity_check_detections.py \
		$(VIDEO_SMOKE_RUN)/detections.jsonl
```

- [ ] **Step 2: Run target**

Run: `make backend-video-smoke`
Expected: worker completes, sanity script exits 0.

---

## Task 8: Refresh aggregate acceptance targets

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Update `phase2-foundation-acceptance`**

Make it depend on the new targets:

```makefile
phase2-foundation-acceptance: artifacts-check backend-unit-strict backend-cli-smoke \
    backend-native-build backend-native-linkcheck backend-native-unit \
    backend-detector-parity backend-detector-determinism backend-hotpath \
    backend-video-smoke sprint-02-acceptance
	@echo "=== git diff --check ==="
	git diff --check
	@echo "Phase2 foundation acceptance PASSED"
```

- [ ] **Step 2: Update `sprint-02-acceptance`**

Keep existing dependencies, or optionally also add `backend-native-linkcheck` so Sprint 02 native smoke runs after link verification:

```makefile
sprint-02-acceptance: backend-unit backend-native-build backend-native-linkcheck backend-native-smoke frontend-test frontend-build
	@echo "=== git diff --check ==="
	git diff --check
	@echo "Sprint 02 acceptance PASSED"
```

- [ ] **Step 3: Run top-level acceptance**

Run: `make phase2-foundation-acceptance`
Expected: all gates pass and `git diff --check` is clean.

---

## Task 9: Final review and report

- [ ] Run `git status --short` and verify no unintended files were created.
- [ ] Run `git diff --check`.
- [ ] Report to the user which targets pass and which still fail (if any), with reasons.
