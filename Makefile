REPO := $(shell pwd)
CONTAINER ?= nvcr.io/nvidia/deepstream:9.0-triton-multiarch
BUILD_CONTAINER ?= mergenvision/deepstream-dev:9.0
GPU_DEVICE ?= 0

NATIVE_SRC := /app/backend/native
NATIVE_BUILD := /app/backend/native/build
WORKER := /app/backend/native/build/deepstream_face_worker
GST_PLUGIN_PATH := /app/backend/native/build
TEST_VIDEOS := /app/backend/artifacts/videos
OUT_DIR := /app/backend/out

DOCKER_RUN := docker run --rm --gpus "device=$(GPU_DEVICE)" \
	-e CUDA_VISIBLE_DEVICES=0 \
	-e GST_PLUGIN_PATH=$(GST_PLUGIN_PATH) \
	-v "$(REPO):/app" \
	-w /app

.PHONY: backend-unit backend-native-build backend-native-smoke frontend-test frontend-build sprint-02-acceptance

backend-unit:
	@echo "=== Backend unit/integration tests ==="
	cd backend && python3 -m pytest tests/unit tests/integration -q

backend-native-build:
	@echo "=== Backend native build ==="
	mkdir -p backend/native/build
	$(DOCKER_RUN) -w $(NATIVE_BUILD) --entrypoint /usr/bin/cmake $(BUILD_CONTAINER) $(NATIVE_SRC) -B $(NATIVE_BUILD)
	$(DOCKER_RUN) -w $(NATIVE_BUILD) --entrypoint /usr/bin/cmake $(BUILD_CONTAINER) --build $(NATIVE_BUILD)

backend-native-smoke:
	@echo "=== Backend native detector smoke ==="
	$(DOCKER_RUN) --entrypoint $(WORKER) $(CONTAINER) $(TEST_VIDEOS)/friendsshort_50f.mp4 $(OUT_DIR)/sprint-02-smoke 0
	$(DOCKER_RUN) --entrypoint python3 $(CONTAINER) /app/backend/scripts/sanity_check_detections.py $(OUT_DIR)/sprint-02-smoke/detections.jsonl

frontend-test:
	@echo "=== Frontend tests ==="
	cd frontend && npm run test

frontend-build:
	@echo "=== Frontend typecheck/build ==="
	cd frontend && npm run typecheck

sprint-02-acceptance: backend-unit backend-native-build backend-native-smoke frontend-test frontend-build
	@echo "=== git diff --check ==="
	git diff --check
	@echo "Sprint 02 acceptance PASSED"
