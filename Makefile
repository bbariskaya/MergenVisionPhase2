REPO := $(shell pwd)
CONTAINER ?= nvcr.io/nvidia/deepstream:9.0-triton-multiarch
BUILD_CONTAINER ?= mergenvision/deepstream-dev:9.0
GPU_DEVICE ?= 0

NATIVE_SRC := /app/backend/native
NATIVE_BUILD := /app/backend/native/build
WORKER := /app/backend/native/build/deepstream_face_worker
GST_PLUGIN_PATH := /app/backend/native/build/gst-plugins
TEST_VIDEOS := /app/backend/artifacts/videos
OUT_DIR := /app/backend/out

DOCKER_RUN := docker run --rm --gpus "device=$(GPU_DEVICE)" \
	-e CUDA_VISIBLE_DEVICES=0 \
	-e GST_PLUGIN_PATH=$(GST_PLUGIN_PATH) \
	-v "$(REPO):/app" \
	-w /app

.PHONY: artifacts-check backend-unit backend-unit-strict backend-native-build backend-native-linkcheck backend-native-unit backend-native-smoke backend-cli-smoke backend-detector-parity backend-detector-frame-identity backend-detector-engine-parity backend-detector-determinism backend-hotpath backend-video-smoke frontend-test frontend-build sprint-02-acceptance phase2-sprint-01-acceptance phase2-foundation-acceptance backend-batch-invariants backend-cli-tracker-reject backend-batch-parity backend-batch-determinism backend-render-parity backend-batch-benchmark phase2-sprint-04-acceptance phase2-sprint-06-native-build phase2-sprint-06-core-unit phase2-sprint-06-sanitizers phase2-sprint-06-plugin-integration phase2-sprint-06-reconciliation-unit phase2-sprint-06-acceptance

artifacts-check:
	@echo "=== Artifact manifest check ==="
	cd backend && python3 scripts/artifacts_check.py

backend-unit:
	@echo "=== Backend unit/integration tests ==="
	cd backend && python3 -m pytest tests/unit tests/integration -q

backend-unit-strict:
	@echo "=== Backend unit/integration tests with -W error ==="
	cd backend && python3 -m pytest tests/unit tests/integration -q -W error

backend-native-build:
	@echo "=== Backend native build ==="
	mkdir -p backend/native/build
	$(DOCKER_RUN) -w $(NATIVE_BUILD) --entrypoint /usr/bin/cmake $(BUILD_CONTAINER) $(NATIVE_SRC) -B $(NATIVE_BUILD)
	$(DOCKER_RUN) -w $(NATIVE_BUILD) --entrypoint /usr/bin/cmake $(BUILD_CONTAINER) --build $(NATIVE_BUILD)

backend-native-linkcheck:
	@echo "=== Backend native link check ==="
	python3 backend/scripts/native_linkcheck.py

backend-native-unit:
	@echo "=== Backend native unit tests ==="
	$(DOCKER_RUN) --entrypoint $(NATIVE_BUILD)/test_nms $(CONTAINER)

backend-native-smoke:
	@echo "=== Backend native detector smoke ==="
	$(DOCKER_RUN) --entrypoint $(WORKER) $(CONTAINER) $(TEST_VIDEOS)/friendsshort_50f.mp4 $(OUT_DIR)/sprint-02-smoke 0
	$(DOCKER_RUN) --entrypoint python3 $(CONTAINER) /app/backend/scripts/sanity_check_detections.py $(OUT_DIR)/sprint-02-smoke/detections.jsonl

backend-detector-pipeline-parity: backend-native-build
	@echo "=== Detector pipeline parity vs CPU oracle ==="
	$(DOCKER_RUN) --entrypoint $(WORKER) $(CONTAINER) $(TEST_VIDEOS)/friendsshort_50f.mp4 $(OUT_DIR)/sprint01_50f_acceptance 0
	python3 backend/tests/native/test_detector_pipeline_parity.py

backend-detector-frame-identity: backend-native-build
	@echo "=== Detector preprocess tensor frame identity gate ==="
	$(DOCKER_RUN) -e MV_DUMP_PREPROC_TENSOR=/app/backend/out/preproc_dump --entrypoint $(WORKER) $(CONTAINER) $(TEST_VIDEOS)/friendsshort_50f.mp4 $(OUT_DIR)/frame_identity_smoke 0
	docker run --rm -v "$(REPO):/app" -w /app --entrypoint chmod $(CONTAINER) -R a+rw /app/backend/out/preproc_dump
	python3 backend/tests/native/test_detector_frame_identity.py

backend-detector-engine-parity: backend-native-build
	@echo "=== Detector engine/tensor parity (ONNX vs TensorRT) ==="
	python3 backend/tests/native/test_detector_engine_parity.py

backend-detector-determinism: backend-native-build
	@echo "=== Detector determinism repeated run check ==="
	python3 backend/tests/native/test_detector_determinism.py

backend-hotpath:
	@echo "=== GPU hot-path contract (Nsight Systems) ==="
	python3 -m pytest backend/tests/native/test_gpu_hot_path_contract.py -v

backend-video-smoke: backend-native-build
	@echo "=== Backend video smoke (longer clip) ==="
	$(DOCKER_RUN) --entrypoint $(WORKER) $(CONTAINER) $(TEST_VIDEOS)/Friends.mp4 $(OUT_DIR)/video_smoke 0
	$(DOCKER_RUN) --entrypoint python3 $(CONTAINER) /app/backend/scripts/sanity_check_detections.py $(OUT_DIR)/video_smoke/detections.jsonl

backend-batch-invariants:
	@echo "=== Backend batch-N source invariants ==="
	cd backend && python3 -m pytest tests/integration/test_batch_invariants.py -q

backend-cli-tracker-reject: backend-native-build
	@echo "=== CLI rejects tracker + batch-size > 1 ==="
	python3 backend/tests/native/test_cli_tracker_batch_reject.py

backend-batch-parity: backend-native-build
	@echo "=== Batch detection parity (1 vs N, tracker off) ==="
	python3 backend/tests/native/test_batch_detection_parity.py

backend-batch-determinism: backend-native-build
	@echo "=== Batch-N determinism ==="
	python3 backend/tests/native/test_batch_determinism.py

backend-render-parity: backend-native-build
	@echo "=== Annotated render parity ==="
	python3 backend/tests/native/test_render_parity.py

backend-batch-benchmark: backend-native-build
	@echo "=== Batch benchmark (Friends.mp4, batch=1 vs batch=16) ==="
	python3 backend/tests/native/test_batch_benchmark.py

backend-cli-smoke:
	@echo "=== Backend CLI smoke ==="
	cd backend && python3 -m app.cli detect \
		--video ../backend/artifacts/videos/friendsshort_50f.mp4 \
		--output ../backend/out/cli-smoke \
		--host-gpu $(GPU_DEVICE)

frontend-test:
	@echo "=== Frontend tests ==="
	cd frontend && npm run test

frontend-build:
	@echo "=== Frontend build ==="
	cd frontend && npm run build

sprint-02-acceptance: backend-unit backend-native-build backend-native-linkcheck backend-native-unit backend-native-smoke frontend-test frontend-build
	@echo "=== git diff --check ==="
	git diff --check
	@echo "Sprint 02 acceptance PASSED"

phase2-sprint-01-acceptance: artifacts-check backend-unit-strict frontend-test frontend-build
	@echo "Phase2 Sprint 01 acceptance PASSED (native parity/hot-path targets maintained separately)"

phase2-foundation-acceptance: artifacts-check backend-unit-strict backend-cli-smoke sprint-02-acceptance \
    backend-detector-frame-identity backend-detector-pipeline-parity backend-detector-engine-parity backend-detector-determinism backend-hotpath backend-video-smoke
	@echo "=== git diff --check ==="
	git diff --check
	@echo "Phase2 foundation acceptance PASSED"

phase2-sprint-04-acceptance: artifacts-check backend-unit-strict backend-native-build backend-native-unit \
    backend-batch-invariants backend-cli-tracker-reject backend-batch-parity backend-batch-determinism \
    backend-render-parity backend-batch-benchmark backend-video-smoke
	@echo "=== git diff --check ==="
	git diff --check
	@echo "Phase2 Sprint 04 acceptance PASSED"

phase2-sprint-06-native-build: backend-native-build

phase2-sprint-06-core-unit: backend-native-build
	@echo "=== Sprint 06 core tracking unit tests ==="
	$(DOCKER_RUN) --entrypoint $(NATIVE_BUILD)/tracking/test_tracking_core $(BUILD_CONTAINER)
	$(DOCKER_RUN) --entrypoint $(NATIVE_BUILD)/tracking/test_evidence_writer $(BUILD_CONTAINER)

phase2-sprint-06-sanitizers: backend-native-build
	@echo "=== Sprint 06 core tracker ASan/UBSan ==="
	$(DOCKER_RUN) --entrypoint $(NATIVE_BUILD)/tracking/test_tracking_core_asan $(BUILD_CONTAINER)

phase2-sprint-06-plugin-integration: backend-native-build
	@echo "=== Sprint 06 mvfacetracker plugin registration ==="
	$(DOCKER_RUN) --entrypoint gst-inspect-1.0 $(BUILD_CONTAINER) mvfacetracker

phase2-sprint-06-reconciliation-unit:
	@echo "=== Sprint 06 Python reconciliation unit tests ==="
	cd backend && python3 -m pytest tests/unit/tracking -q

phase2-sprint-06-acceptance: backend-unit-strict phase2-sprint-06-native-build phase2-sprint-06-core-unit \
    phase2-sprint-06-sanitizers phase2-sprint-06-plugin-integration phase2-sprint-06-reconciliation-unit
	@echo "=== git diff --check ==="
	git diff --check
	@echo "Phase2 Sprint 06 acceptance PASSED"
