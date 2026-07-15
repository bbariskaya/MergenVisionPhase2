# Milestone A — RetinaFace + Tracker Native GPU Vertical Slice

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `friendsshort.mp4` through a pure-GPU DeepStream pipeline (NVDEC → nvinfer RetinaFace → native C++ postprocess → nvtracker) and produce a JSON metadata file containing original-resolution bboxes, 5 keypoints, raw tracker tracklet IDs, frame/PTS metadata, with CPU-oracle parity proof and clean EOS/ teardown.

**Architecture:** A single Dockerized GPU worker loads the inherited RetinaFace engine inside DeepStream `nvinfer` and uses a C++ custom parser (`libnvdsinfer_custom_retinaface.so`) that decodes `loc/conf/landms` tensors on CPU with compact result buffers. The decoded video frames remain in NVMM/GPU memory; only compact metadata (bbox, landmarks, scores, track IDs) crosses to the CPU. A thin Python harness starts the worker container and validates output. CPU oracle (OpenCV + ONNX Runtime) exists only as a parity test fixture, never in the production image hot path.

**Tech Stack:** GStreamer 1.24.2, DeepStream 7.1, TensorRT 10.3, CUDA 12.4, C++17, GCC 11+, Python 3.12 (orchestrator/parity tests), ONNX Runtime (test fixture only).

## Global Constraints

- Production hot path keeps decoded frames in GPU memory; full-frame CPU copy is forbidden.
- Python/NumPy/OpenCV post-processing is **not** a production path; compact metadata only may be consumed by Python.
- CPU oracle (OpenCV + ONNXRuntime) is a test fixture only, excluded from the production Docker image and from throughput benchmarks.
- Runtime `batch-size=1` until tracker-correctness gate is passed; larger batch is a separate benchmark gate.
- Old artifacts copied from `MergenVisionDemo` are never deleted. If they fail a gate, mark `rejected_for_phase2_runtime: true` and build a new engine under a different filename; preserve original SHA-256.
- No MinIO, PostgreSQL, SSE, Qdrant, upload streaming, or recognition in Milestone A.
- All artifact provenance is `inherited_local_artifact_unverified`; commercial use is `blocked_pending_provenance`.

---

## File Structure

- `docker/Dockerfile.worker` — GPU worker image with GStreamer, DeepStream, TensorRT, no OpenCV/ONNXRuntime.
- `docker/docker-compose.worker.yml` — compose file mounting `models/`, `engines/`, `backend/artifacts/videos/` read-only.
- `native/retinaface_parser/` — C++ custom parser for `nvinfer`:
  - `retinaface_parser.h`
  - `retinaface_parser.cpp` — prior generation, variance decode, conf filter, NMS, landmark decode, reverse mapping.
  - `CMakeLists.txt` to build `libnvdsinfer_custom_retinaface.so`.
- `native/worker/` — DeepStream pipeline harness:
  - `main.cpp` — `filesrc → qtdemux → h264parse → nvv4l2decoder → nvstreammux → nvinfer → nvtracker → native sink probe → JSON emitter`.
  - `metadata_writer.cpp/h` — write compact bbox/landmark/track metadata to JSON.
- `configs/` — DeepStream config files:
  - `config_retinaface.txt`
  - `config_tracker.txt`
- `tests/fixtures/cpu_oracle/` — CPU parity oracle. **Test-only.** Not shipped in worker image.
  - `cpu_oracle_retinaface.py`
  - `cpu_oracle_glintr100.py`
- `data/annotations/Phoebe/` — golden annotations for a Phoebe clip/frames, used to prove detector + recognizer parity.
- `tests/integration/` — integration tests that run the dockerized worker and compare output with CPU oracle/expected annotations.

---

## Task 1: Worker Container + Artifact Runtime Inventory

**Files:**
- Create: `docker/Dockerfile.worker`
- Create: `docker/docker-compose.worker.yml`
- Create: `scripts/runtime_inventory.sh`

**Interfaces:**
- Produces: Docker image `mergenvision/worker:milestone-a`.

- [ ] **Step 1: Write Dockerfile.worker**

Base from NVIDIA DeepStream 7.1 devel image, install `sha256sum`, `cmake`, `g++`. Do **not** install OpenCV or ONNX Runtime. Mount read-only host directories for `models/`, `engines/`, `backend/artifacts/videos/`.

```dockerfile
FROM nvcr.io/nvidia/deepstream:7.1-devel
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ make ca-certificates coreutils && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/mergenvision
```

- [ ] **Step 2: Write docker-compose.worker.yml**

```yaml
services:
  worker:
    build:
      context: ..
      dockerfile: docker/Dockerfile.worker
    image: mergenvision/worker:milestone-a
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    volumes:
      - ../backend/artifacts/models:/app/backend/artifacts/models:ro
      - ../backend/artifacts/engines:/app/backend/artifacts/engines:ro
      - ../backend/artifacts/videos:/app/backend/artifacts/videos:ro
      - ../backend/out:/app/backend/out
```

- [ ] **Step 3: Add runtime inventory helper**

`scripts/runtime_inventory.sh` runs inside container and prints:

```bash
#!/bin/bash
set -e
echo "=== GStreamer plugins ==="
gst-inspect-1.0 nvv4l2decoder nvstreammux nvinfer nvtracker fakesink
echo "=== TensorRT ==="
trtexec --version
echo "=== CUDA ==="
nvcc --version
```

Expected: all elements present, TensorRT version 10.x, CUDA 12.4.

- [ ] **Step 4: Build and run inventory**

```bash
docker compose -f docker/docker-compose.worker.yml build
docker compose -f docker/docker-compose.worker.yml run --rm worker bash scripts/runtime_inventory.sh
```

Expected: build succeeds, `nvinfer` element visible.

- [ ] **Step 5: Commit**

```bash
git add docker/ scripts/
git commit -m "chore(worker): add DeepStream/TensortRT worker image"
```

---

## Task 2: TensorRT Deserialize Gate (Gate A)

**Files:**
- Create: `native/worker/deserialize_smoke.cpp`
- Create: `tests/integration/test_deserialize_gate.py`

**Interfaces:**
- Produces: exit code 0 + logs "engine loaded, profile min/opt/max = ...".

- [ ] **Step 1: Write native deserialize smoke program**

```cpp
#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    const char* enginePath = argv[1];
    std::ifstream file(enginePath, std::ios::binary);
    std::vector<char> plan((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    auto logger = nvinfer1::Logger(nvinfer1::ILogger::Severity::kINFO);
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine) return 1;
    std::cout << "engine loaded, nbBindings=" << engine->getNbIOTensors() << "\n";
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        auto dims = engine->getTensorShape(name);
        std::cout << " tensor=" << name << " shape=" << ... << "\n";
    }
    return 0;
}
```

- [ ] **Step 2: Add Python test that runs smoke inside container**

```python
import subprocess
def test_retinaface_deserialize_gate():
    r = subprocess.run([
        "docker","compose","-f","docker/docker-compose.worker.yml",
        "run","--rm","worker",
        "/opt/mergenvision/native/worker/deserialize_smoke",
        "/opt/mergenvision/engines/retinaface_r50_dynamic.engine"
    ], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "loc" in r.stdout and "conf" in r.stdout and "landms" in r.stdout
```

```python
def test_glintr100_deserialize_gate() ...
```

- [ ] **Step 3: Run test**

```bash
pytest tests/integration/test_deserialize_gate.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add native/worker/deserialize_smoke.cpp tests/integration/test_deserialize_gate.py
git commit -m "test(gate-a): add TensorRT deserialize gate for both engines"
```

---

## Task 3: Custom RetinaFace Parser Library

**Files:**
- Create: `native/retinaface_parser/retinaface_parser.h`
- Create: `native/retinaface_parser/retinaface_parser.cpp`
- Create: `native/retinaface_parser/CMakeLists.txt`

**Interfaces:**
- Function signature: `bool NvDsInferParseCustomRetinaFace(...)` matching DeepStream `NvDsInferParseCustomFunc` contract.
- Produces: vector of `NvDsInferParseObjectInfo` with bbox in **model input 640x640** space; reverse mapping done by caller using `scale_x`, `scale_y`.

- [ ] **Step 1: Write header with exported function**

```cpp
#pragma once
#include "nvdsinfer_custom_impl.h"
#include <vector>

extern "C" bool NvDsInferParseCustomRetinaFace(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferParseObjectInfo>& objectList,
    float nmsThreshold = 0.4f,
    float variance0 = 0.1f,
    float variance1 = 0.2f);
```

- [ ] **Step 2: Implement prior generation and decode**

Priors formula from `engine_metadata.json` and old `retinaface_decode.cu` reference:
- feature maps 80, 40, 20 (640 / steps [8,16,32])
- `min_sizes = [[16,32],[64,128],[256,512]]` for the three levels
- For each level, for each cell `(cy,cx)` and each size `s`:
  - `w = s / 640.0f`, `h = s / 640.0f`
  - `cx = (j + 0.5f) * step / 640.0f`, `cy = (i + 0.5f) * step / 640.0f`

Decode:
- `cx = prior.cx + loc[0] * variance0 * prior.w`
- `cy = prior.cy + loc[1] * variance0 * prior.h`
- `w  = prior.w * exp(loc[2] * variance1)`
- `h  = prior.h * exp(loc[3] * variance1)`
- Convert to `(x1,y1,x2,y2)` in **640x640** space.

Landmark decode:
- `landmark_x = prior.cx + landms[k]   * variance0 * prior.w`
- `landmark_y = prior.cy + landms[k+1] * variance0 * prior.h` for k=0..4 in input 640x640 space.

Confidence filter: `conf[1] > 0.5`.

NMS: standard IoU, sort descending by score.

- [ ] **Step 3: Build the .so inside worker image**

```cmake
add_library(nvdsinfer_custom_retinaface SHARED retinaface_parser.cpp)
set_target_properties(nvdsinfer_custom_retinaface PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON)
```

- [ ] **Step 4: Add unit test against CPU oracle**

`tests/unit/test_retinaface_parser.py` loads the same ONNX in ONNX Runtime, applies equivalent decode in Python, and asserts bbox/landmark lists match C++ parser output on synthetic `loc/conf/landms` tensors within 1e-3.

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/test_retinaface_parser.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add native/retinaface_parser/ tests/unit/test_retinaface_parser.py
git commit -m "feat(native): add RetinaFace custom parser with prior decode and NMS"
```

---

## Task 4: DeepStream nvinfer Integration Gate (Gate B)

**Files:**
- Create: `configs/config_retinaface.txt`
- Modify: `docker/Dockerfile.worker` to install parser .so
- Create: `native/worker/nvinfer_integration_probe.cpp` (lightweight probeproving)
- Create: `tests/integration/test_nvinfer_gate.py`

**Interfaces:**
- Config consumed by `nvinfer`.
- Probe checks that output layers named `loc`, `conf`, `landms` exist and batch binding is correct.

- [ ] **Step 1: Write config_retinaface.txt**

```ini
[property]
gpu-id=0
net-scale-factor=1.0
model-color-format=1
mean-file=0
offsets=104.0;117.0;123.0
model-engine-file=/opt/mergenvision/engines/retinaface_r50_dynamic.engine
labelfile-path=/opt/mergenvision/empty_labels.txt
infer-input-dims=3;640;640;0
batch-size=1
network-mode=2
num-detected-classes=1
interval=0
gie-unique-id=1
output-tensor-names=loc;conf;landms
parse-bbox-func-name=NvDsInferParseCustomRetinaFace
custom-lib-path=/opt/mergenvision/native/retinaface_parser/libnvdsinfer_custom_retinaface.so
```

Add `empty_labels.txt` with one line `face`.

- [ ] **Step 2: Build pipeline with probe proving layer names**

`native/worker/nvinfer_integration_probe.cpp` constructs:

```text
filesrc -> qtdemux -> h264parse -> nvv4l2decoder -> nvstreammux -> nvinfer(retinaface) -> fakesink
```

Hook a source pad probe on `nvinfer` source pad, read ` NvDsBatchMeta`, print ` NvDsObjectMeta` count and confirm at EOS that `nvinfer` produced output with expected tensor names via `gst_element_get_static_pad` and `nvinfer` properties. Use `GST_DEBUG=2` and observe no errors and `CUSTOM_LIB` load line.

- [ ] **Step 3: Add Python test**

```python
def test_nvinfer_integration_gate():
    r = subprocess.run([
        "docker","compose","-f","docker/docker-compose.worker.yml",
        "run","--rm","worker",
        "/opt/mergenvision/native/worker/nvinfer_integration_probe",
        "/opt/mergenvision/backend/artifacts/videos/friendsshort.mp4",
        "/opt/mergenvision/configs/config_retinaface.txt"
    ], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "NvDsInferParseCustomRetinaFace" in r.stderr
    assert "tensor=loc" in r.stdout
    assert "tensor=conf" in r.stdout
    assert "tensor=landms" in r.stdout
    assert "EOS clean" in r.stdout
```

- [ ] **Step 4: Run test**

```bash
pytest tests/integration/test_nvinfer_gate.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/ native/worker/nvinfer_integration_probe.cpp tests/integration/test_nvinfer_gate.py
git commit -m "test(gate-b): add DeepStream nvinfer integration gate"
```

---

## Task 5: Output Semantic / Parity Gate (Gate C)

**Files:**
- Create: `native/worker/metadata_writer.cpp`
- Create: `native/worker/metadata_writer.h`
- Create: `tests/fixtures/cpu_oracle/cpu_oracle_retinaface.py`
- Create: `tests/integration/test_parity_gate.py`

**Interfaces:**
- Native JSON schema: list of frames with `frame_index`, `pts_ms`, `objects` list of `{bbox_xyxy, landmarks_5x2, score}`.
- Coordinates returned in **original-resolution** using per-axis reverse mapping:
  - `scale_x = original_width / 640.0`
  - `scale_y = original_height / 640.0`
  - `x_orig = x_640 * scale_x`, `y_orig = y_640 * scale_y`.

- [ ] **Step 1: Write metadata_writer**

In the `nvinfer` source pad probe, for each frame:
- Original width/height from `NvDsFrameMeta->source_frame_width/height`.
- For each object meta produced by custom parser:
  - Reverse map bbox and landmarks.
  - Append to JSON structure for that frame.

Write output to `/app/backend/out/retinaface_parity.json`.

- [ ] **Step 2: Write CPU oracle fixture**

`tests/fixtures/cpu_oracle/cpu_oracle_retinaface.py`:
- Decode a test frame with OpenCV.
- Squish-resize to 640x640, BGR, subtract mean [104,117,123].
- Run ONNX Runtime with `retinaface_r50_dynamic.onnx`.
- Apply same prior/decode/NMS as parser.
- Return original-resolution bbox/landmarks.

- [ ] **Step 3: Add parity test**

Run worker on `friendsshort.mp4`, read output JSON, and compare first-N frames with CPU oracle using `pytest.approx(abs=1.5 px)` and same detections count.

```python
def test_retinaface_output_parity_gate():
    run_worker("friendsshort.mp4")
    native = json.load(open("backend/out/retinaface_parity.json"))
    oracle = cpu_oracle_retinaface.run("backend/artifacts/videos/friendsshort.mp4", max_frames=10)
    assert native[0]["objects"][0]["bbox_xyxy"] == pytest.approx(oracle[0]["bbox_xyxy"], abs=1.5)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/integration/test_parity_gate.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add native/worker/metadata_writer.* tests/fixtures/cpu_oracle/ tests/integration/test_parity_gate.py
git commit -m "test(gate-c): add RetinaFace output semantic and CPU parity gate"
```

---

## Task 6: Preprocess / Reverse Mapping Parity Test

**Files:**
- Create: `tests/integration/test_preprocess_parity.py`

**Interfaces:**
- Compares GPU pipeline preprocess against CPU oracle pixel-level parity, not just config lines.

- [ ] **Step 1: Design the pixel parity gate**

Use a synthetic YUV420/H.264 clip with known solid RGB colors. The CPU oracle decodes the same original frame, performs squish-resize, BGR conversion, and mean subtraction. The GPU path exposes the input after `nvinfer` preprocess if `nvinfer` supports dumping raw tensor; otherwise instrument with a small custom `Gst-nvvideo4linux2` + `nvdspreprocess` element that writes the final preprocessed tensor to CPU for this test only. Alternative: compare final bbox/landmark results; however the user explicitly asked for pixel parity, so we instrument.

Expected output: mean absolute difference < 3.0 per channel across 640x640 input.

- [ ] **Step 2: Add explicit reverse mapping test**

For synthetic 1280x720 input:

```python
scale_x = 1280 / 640.0
scale_y = 720 / 640.0
bbox_640 = [100, 150, 300, 400]
bbox_orig = [100 * scale_x, 150 * scale_y, 300 * scale_x, 400 * scale_y]
landmark_640 = [[110,160], ...]
landmark_orig = [[110*scale_x, 160*scale_y], ...]
```

Assert native output matches these values.

- [ ] **Step 3: Run test**

```bash
pytest tests/integration/test_preprocess_parity.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_preprocess_parity.py
git commit -m "test(gate-c.2): add preprocess pixel and reverse mapping parity"
```

---

## Task 7: Tracker Integration with Batch=1 + PTS Order

**Files:**
- Create: `configs/config_tracker.txt`
- Create: `native/worker/main.cpp`
- Modify: `native/worker/metadata_writer.cpp` to capture tracker IDs

**Interfaces:**
- Tracker output is a raw local tracklet ID from `NvDsObjectMeta->object_id`.
- Metadata JSON now includes `tracker_id` per object.

- [ ] **Step 1: Configure nvtracker**

```ini
[tracker]
tracker-width=640
tracker-height=640
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml
enable-batch-process=1
enable-past-frame=1
```

- [ ] **Step 2: Extend pipeline to nvtracker → fakesink**

```text
... nvinfer -> nvtracker -> fakesink
```

In pad probe after tracker, read `obj_meta->object_id` and write to JSON.

- [ ] **Step 3: Add PTS order test**

Verify that for batch-size=1 the JSON `frame_index` and `pts_ms` are monotonically increasing and that tracker IDs do not change order unexpectedly.

- [ ] **Step 4: Run end-to-end on friendsshort.mp4**

```bash
docker compose -f docker/docker-compose.worker.yml run --rm worker \
  /opt/mergenvision/native/worker/mergenvision_worker \
  -i /opt/mergenvision/backend/artifacts/videos/friendsshort.mp4 \
  -c /opt/mergenvision/configs/config_retinaface.txt \
  -t /opt/mergenvision/configs/config_tracker.txt \
  -o /app/backend/out/milestone_a.json
```

Expected: exit 0, `backend/out/milestone_a.json` produced.

- [ ] **Step 5: Commit**

```bash
git add native/worker/main.cpp configs/config_tracker.txt
git commit -m "feat(worker): add nvtracker to pipeline with batch=1 and PTS order"
```

---

## Task 8: Milestone A Acceptance Gate

**Files:**
- Create: `tests/integration/test_milestone_a_acceptance.py`

**Interfaces:**
- Verifies the single `backend/out/milestone_a.json` meets acceptance contract.

- [ ] **Step 1: Define acceptance test**

```python
def test_milestone_a_acceptance():
    data = json.load(open("backend/out/milestone_a.json"))
    assert data["video"] == "friendsshort.mp4"
    assert data["completed"] is True
    assert data["gpu_scale_x"] == pytest.approx(1280/640, rel=1e-6)
    assert data["gpu_scale_y"] == pytest.approx(720/640, rel=1e-6)
    for frame in data["frames"]:
        assert "frame_index" in frame and "pts_ms" in frame
        assert frame["pts_ms"] >= prev_pts
        for obj in frame["objects"]:
            assert "bbox_xyxy" in obj and len(obj["bbox_xyxy"]) == 4
            assert "landmarks" in obj and len(obj["landmarks"]) == 5
            assert "tracker_id" in obj
            assert "score" in obj
            # original-resolution bounds
            assert 0 <= obj["bbox_xyxy"][0] <= data["original_width"]
```

- [ ] **Step 2: Verify no full-frame CPU copy**

From `GST_DEBUG=*x*:5` or `nvidia-smi`/`nsys` trace, assert no `memcpyDtoH` of size `original_width*original_height*3` and that `nvstreammux`/`nvinfer` buffers carry `memory:NVMM` caps.

- [ ] **Step 3: Verify EOS/teardown**

Test checks container exit code 0 and `GST_MESSAGE_EOS` in logs.

- [ ] **Step 4: Run acceptance**

```bash
pytest tests/integration/test_milestone_a_acceptance.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_milestone_a_acceptance.py
git commit -m "test(acceptance): add Milestone A end-to-end acceptance gate"
```

---

## Task 9: glintr100 Parity Gates (Same Structure)

**Files:**
- Create: `configs/config_glintr100.txt`
- Create: `native/glintr100_alignment/` (CUDA alignment using 5 landmarks)
- Create: `tests/integration/test_glintr100_gates.py`

**Interfaces:**
- `align_face_gpu(landmarks_5x2, image_gpu)` produces 112x112 aligned face.
- `glintr100_infer(aligned_faces_batch)` produces L2-normalized 512-D embedding.

- [ ] **Step 1: Alignment + recognizer deserialize/integration gates**

Same A/B pattern as RetinaFace: deserialize engine inside container, then load in DeepStream `nvinfer` with input 112x112.

- [ ] **Step 2: Landmark order / alignment parity test**

Use a Phoebe annotated frame from `data/annotations/Phoebe/annotations.yaml`:
- Native detector output landmarks.
- GPU alignment crop using canonical 5-point similarity transform and ArcFace template.
- CPU oracle alignment crop using OpenCV + same template.
- Save contact-sheet PNGs, pixel error test `MAE < 5.0`.
- Run both crops through `glintr100.onnx` and assert cosine similarity > 0.999.

- [ ] **Step 3: Embedding parity gate**

Run the same aligned face through the TensorRT engine and ONNX CPU oracle; assert element-wise difference < 1e-3 and cosine similarity > 0.9999 after L2 normalization.

- [ ] **Step 4: Commit**

```bash
git add configs/config_glintr100.txt native/glintr100_alignment/ tests/integration/test_glintr100_gates.py
git commit -m "test(recognition): add glintr100 deserialize, integration and parity gates"
```

---

## Task 10: Phoebe Annotation Folder

**Files:**
- Create: `data/annotations/Phoebe/schema.yaml`
- Create: `data/annotations/Phoebe/README.md`
- Create: `data/annotations/Phoebe/annotations.yaml`

**Interfaces:**
- Golden annotations used to prove detector bbox + 5 KPS and recognizer alignment/embedding parity.

- [ ] **Step 1: Define annotation schema**

```yaml
schema_version: 1.0
identity:
  name: Phoebe Buffay
  source: Friends
  canonical_face_id: phoebe_001
annotations:
  - media_path: backend/artifacts/videos/friendsshort.mp4
    type: video
    frame_index: int
    pts_ms: float
    bbox_xyxy: [float, float, float, float]   # original resolution
    landmarks_5x2: [[x0,y0], [x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    visibility: visible|occluded|ignore
  - media_path: backend/artifacts/gallery/Phoebe/some_photo.jpg
    type: gallery
    bbox_xyxy: [...]
    landmarks_5x2: [...]
```

- [ ] **Step 2: Populate initial Phoebe annotations**

Select 3 representative frames/clips and manually label bbox + 5 landmarks. The file is the source of truth; annotation conforms to the schema.

- [ ] **Step 3: Add test proving detector matches annotations**

```python
def test_phoebe_detector_matches_annotations():
    for ann in load_phoebe_annotations():
        native = run_pipeline_on_frame(ann["media_path"], ann["frame_index"])
        assert match_closest_detection(native, ann)["iou"] >= 0.85
```

- [ ] **Step 4: Add test proving recognizer embedding parity**

Use Phoebe gallery annotations as enrollment; compare native vs CPU oracle embeddings for the same aligned crop.

- [ ] **Step 5: Commit**

```bash
git add data/annotations/Phoebe/
git commit -m "data(phoebe): add Phoebe annotation folder with detector/recognition proof targets"
```

---

## Task 11: Artifact Manifest & Policy Update

**Files:**
- Modify: `docs/model_artifacts/MANIFEST.yaml`
- Create: `docs/model_artifacts/ARTIFACT_POLICY.md`

**Interfaces:**
- Manifest reflects inherited artifacts with unverified provenance and no commercial clearance.

- [ ] **Step 1: Ensure corrected provenance labels**

Each artifact entry contains:

```yaml
provenance_status: inherited_local_artifact_unverified
license_status: unverified
commercial_gate: blocked_pending_provenance
files:
  - path: models/retinaface_r50_dynamic.onnx
    sha256: ...
    rejected_for_phase2_runtime: false
```

No absolute paths; use `source_repository: MergenVisionDemo` and `source_ref: local_read_only_reference` plus SHA maps.

- [ ] **Step 2: Write artifact policy**

Document: old artifacts are never deleted; on failure mark rejected and build new filename; original SHA kept; new engine must not overwrite old.

- [ ] **Step 3: Commit**

```bash
git add docs/model_artifacts/
git commit -m "docs(artifact): correct provenance labels and reject/new-build policy"
```

---

## Task 12: Benchmark Batch Gate (After Acceptance)

**Files:**
- Create: `scripts/benchmark_batch.sh`
- Create: `tests/performance/test_batch_benchmark.py`

**Interfaces:**
- Vary `batch-size` and measure metrics; not part of acceptance, only after Task 8 passes.

- [ ] **Step 1: Add benchmark harness**

Run worker with `batch-size` in `[1,4,8,16]` on `friendsshort.mp4` and report:
- decoded_fps
- detector_fps
- max GPU memory
- tracker ID continuity score
- end-to-end latency

- [ ] **Step 2: Add PTS-order invariant for batched mode**

Even when batch-size > 1, assert every output object carries correct PTS and that frames within a batch are processed in decode order.

- [ ] **Step 3: Commit**

```bash
git add scripts/benchmark_batch.sh tests/performance/test_batch_benchmark.py
git commit -m "perf(batch): add batch benchmark gate after tracker correctness"
```

---

## Spec Coverage Checklist

| Requirement | Task |
|---|---|
| TensorRT deserialize gate | Task 2 |
| DeepStream nvinfer integration gate | Task 4 |
| Output semantic/parity gate | Task 5 |
| Custom parser for loc/conf/landms | Task 3 |
| Native C++ postprocess, no Python/NumPy production path | Tasks 3, 4, 5 |
| Compact metadata only on CPU, no full-frame copy | Tasks 4, 6, 8 |
| Batch=1 until tracker correctness, benchmark later | Tasks 7, 12 |
| Squish resize reverse mapping per axis | Tasks 3, 5, 6 |
| Pixel parity for NVDEC/preprocess | Task 6 |
| Landmark order alignment proof | Tasks 8, 9, 10 |
| CPU oracle test-only | Task 5, 9 (fixtures dir), never Dockerfile |
| Milestone A acceptance on friendsshort.mp4 | Task 8 |
| glintr100 same gates | Task 9 |
| Phoebe annotation + proof | Task 10 |
| Artifact never delete / reject policy | Task 11 |

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-14-milestone-a-retinaface-tracker.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — I execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach do you want?