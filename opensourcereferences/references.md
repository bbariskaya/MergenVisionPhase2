# MergenVision Phase 2 — Approved Reference Registry

**Document type:** Reference-first engineering registry

**Scope:** Uploaded-video, offline face detection/tracking/recognition

**Live stream status:** Deferred; RTSP/webcam implementation is out of scope

**Last reviewed:** 2026-07-15
**Authority:** Planning/implementation evidence; not a requirements document

## 1. Purpose

Bu dosya, MergenVision Phase 2 kodlanmadan önce agent'ın incelemesi gereken resmî dokümanları, upstream source'ları, araştırma makalelerini ve salt-okunur eski proje referanslarını kaydeder.

Bir kaynağın burada listelenmesi:

- dependency'nin otomatik kurulacağı;
- latest branch'in otomatik seçileceği;
- model ağırlığının ticari kullanıma uygun olduğu;
- örnek kodun production-ready olduğu;
- MergenVision requirement'ının değiştirilebileceği

anlamına gelmez.

Her implementation kararı exact requirement, exact installed runtime version, pinned upstream source ve gerçek ölçümle doğrulanmalıdır.

## 2. Source priority

Çelişki halinde:

1. Kullanıcının güncel açık kararı.
2. `requirements/phase2requirements.md`.
3. Onaylanmış MergenVision architecture belgeleri.
4. Exact kurulu sürümün resmî vendor dokümanı.
5. Pinned upstream source, sample ve test.
6. Resmî paper/model card.
7. Eski MergenVision implementasyonları; yalnız lessons learned.
8. Community discussion/blog.

LLM hafızası teknik kaynak değildir.

## 3. Reference-first workflow

Bir reference kullanılmadan önce agent:

1. İlgili requirement ve `AGENTS.md` bölümünü okur.
2. Runtime inventory çıkarır.
3. Reference ID'lerini sprint planına yazar.
4. Exact upstream tag/commit SHA'yı pinler.
5. İlgili source/sample/test dosyalarını gerçekten okur.
6. Preprocess, tensor, coordinate, metadata ownership ve teardown contract'larını doğrular.
7. Code ve model-weight lisanslarını ayrı değerlendirir.
8. “reuse / adapt / reject” kararını `docs/implementation/REFERENCE_DECISION_LOG.md` içine kaydeder.
9. Failing test/reproducer üretir.
10. Gerçek runtime acceptance çalıştırır.

Kaynak yalnız README başlığı okunarak `used` sayılmaz.

## 4. Runtime inventory — kaynak seçmeden önce

İlk native/GPU sprintinde aşağıdaki çıktılar kaydedilir:

```bash
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv
nvcc --version
trtexec --version
gst-launch-1.0 --version
gst-inspect-1.0 appsrc
gst-inspect-1.0 qtdemux
gst-inspect-1.0 nvv4l2decoder
gst-inspect-1.0 nvstreammux
gst-inspect-1.0 nvinfer
gst-inspect-1.0 nvtracker
deepstream-app --version-all
ffprobe -version
docker version
docker compose version
```

Element yoksa property tahmin edilmez. Kurulu DeepStream sürümüyle eşleşmeyen docs/sample kopyalanmaz.

## 5. Core reference index

| ID | Source | Primary use | Status / license note |
|---|---|---|---|
| VID-001 | [NVIDIA DeepStream SDK documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html) | dGPU video analytics architecture, samples, compatibility | Primary official; exact installed version selected |
| VID-002 | [Gst-nvvideo4linux2](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvvideo4linux2.html) | NVDEC encoded-video decode | Primary official |
| VID-003 | [Gst-nvstreammux](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html) | batching, timestamps, source metadata | Primary official; old/new mux behavior version-sensitive |
| VID-004 | [Gst-nvinfer](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html) | TensorRT inference, batching, custom parser | Primary official |
| VID-005 | [Gst-nvtracker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html) | NvDCF/IOU tracker architecture and metadata | Primary official |
| VID-006 | [GStreamer appsrc](https://gstreamer.freedesktop.org/documentation/app/appsrc.html) | API byte stream -> bounded GStreamer input | Primary official |
| VID-007 | [GStreamer qtdemux](https://gstreamer.freedesktop.org/documentation/isomp4/qtdemux.html) | MP4/QuickTime demux, pad/caps behavior | Primary official |
| VID-008 | [GStreamer h264parse](https://gstreamer.freedesktop.org/documentation/videoparsersbad/h264parse.html) / [h265parse](https://gstreamer.freedesktop.org/documentation/videoparsersbad/h265parse.html) | codec elementary stream parsing | Primary official |
| VID-009 | [GStreamer bus](https://gstreamer.freedesktop.org/documentation/gstreamer/gstbus.html) | ERROR/EOS/state/teardown semantics | Primary official |
| API-001 | [Starlette Request](https://www.starlette.io/requests/#body) | streaming request body and disconnect | Primary official |
| API-002 | [FastAPI request files](https://fastapi.tiangolo.com/tutorial/request-files/) | multipart/UploadFile behavior and OpenAPI | Primary official; spool behavior measured |
| API-003 | [OpenAPI Specification](https://spec.openapis.org/oas/latest.html) | frozen endpoint schemas | Primary standard |
| API-004 | [RFC 9457 Problem Details](https://www.rfc-editor.org/rfc/rfc9457.html) | structured HTTP errors | Primary standard |
| GPU-001 | [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | engine build/runtime/context/dynamic shapes | Primary official |
| GPU-002 | [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html) | profiling, CUDA Graphs, streams, timing | Primary official |
| GPU-003 | [CUDA C++ Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | memory transfer, streams, synchronization | Primary official |
| GPU-004 | [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | bounded profiling and CPU/GPU timeline | Primary official |
| MOD-001 | [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) | SCRFD/RetinaFace/ArcFace candidates and published metrics | Code/weights terms separate; published weights marked non-commercial research |
| MOD-002 | [SCRFD paper](https://arxiv.org/abs/2105.04714) | efficient face detector design | Research reference |
| MOD-003 | [RetinaFace paper](https://arxiv.org/abs/1905.00641) | five-landmark face detector reference | Research reference |
| MOD-004 | [ArcFace paper](https://arxiv.org/abs/1801.07698) | recognition embedding/loss reference | Research reference; weights separately licensed |
| MOD-005 | [MagFace paper](https://arxiv.org/abs/2103.06627) and [upstream](https://github.com/IrvingMeng/MagFace) | quality-aware embedding candidate | Research candidate; exact weights/license verified |
| MOD-006 | [Ultralytics upstream](https://github.com/ultralytics/ultralytics) and [licensing](https://www.ultralytics.com/license) | optional YOLO-face export/runtime patterns | AGPL-3.0/enterprise decision required; no generic model is assumed face-trained |
| TRK-001 | [ByteTrack upstream](https://github.com/FoundationVision/ByteTrack) and [paper](https://arxiv.org/abs/2110.06864) | tracker A/B candidate | Upstream pinned and license verified before adaptation |
| TRK-002 | [NvDCF tracker docs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html) | NVIDIA-native tracking baseline | Official DeepStream reference |
| EVAL-001 | [CVAT](https://github.com/cvat-ai/cvat) and [docs](https://docs.cvat.ai/) | video tracks, interpolation, human QA/export | Annotation tooling; pin version/export format |
| EVAL-002 | [TrackEval](https://github.com/JonathonLuiten/TrackEval) | HOTA/IDF1/MOT metric reference | Evaluation-only; pin commit |
| EVAL-003 | [MOTChallenge evaluation](https://motchallenge.net/) | tracking metric definitions and format | Metric reference; face-specific interpretation documented |
| EVAL-004 | [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) | detector benchmark terminology | Dataset terms/protocol respected |
| IO-001 | [FFmpeg ffprobe documentation](https://ffmpeg.org/ffprobe.html) | container/stream metadata probe oracle | Tool/reference path, not decode hot path |
| IO-002 | [FFmpeg formats documentation](https://ffmpeg.org/ffmpeg-formats.html) | MP4/MOV/container behavior | Reference/testing; not production CPU decode mandate |
| DB-001 | [PostgreSQL documentation](https://www.postgresql.org/docs/current/) | durable job/history/locking/retention metadata | Exact deployed major version |
| DB-002 | [SQLAlchemy asyncio](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) | async repository/session lifecycle | Exact installed 2.x version |
| OBJ-001 | [MinIO Python SDK API](https://min.io/docs/minio/linux/developers/python/API.html) | video retention, multipart/object lifecycle | Exact installed client/server versions |
| VEC-001 | [Qdrant documentation](https://qdrant.tech/documentation/) | face-sample vector search and filtering | Derived index; exact client/server version |
| OLD-001 | [`bbariskaya/NVDIAgstreamer`](https://github.com/bbariskaya/NVDIAgstreamer) @ `bc2c5c1bb862649b46cda4fe454726eb62c895a4` | previous DeepStream experiment lessons | Read-only historical reference |
| OLD-002 | [`bbariskaya/MergenVisionDemo`](https://github.com/bbariskaya/MergenVisionDemo) | Phase 1 image API/identity/storage behavior | Read-only; merge contract source only after approved review |

## 6. GStreamer and upload ingestion

### VID-006 — `appsrc`

Inspect before implementation:

- `need-data`, `enough-data`, `seek-data` behavior;
- `stream-type` and seekability;
- `block`, `max-bytes`, `max-buffers`, `max-time`;
- caps negotiation;
- buffer ownership/reference lifetime;
- EOS and downstream error propagation;
- request disconnect/cancel interaction.

Project decision guidance:

- API chunks are not accumulated into one unbounded `bytes` object.
- Bounded backpressure is mandatory.
- `push-buffer` failure is a job failure/cancellation signal, not ignored log text.
- MP4 `moov` metadata position and seek requirements are tested.
- Streaming ingest and retention upload may tee encoded chunks; MinIO completion is not a prerequisite for first decode when the container permits progressive parsing.

Do not copy:

- demo code with unlimited appsrc queue;
- busy-wait pushing;
- no cancellation path;
- raw pointer lifetime not tied to GstBuffer ownership.

### VID-007/008 — demux and parsers

Inspect:

- dynamic pad creation and caps;
- H.264/H.265 stream-format/alignment;
- codec configuration data;
- PTS/DTS/timebase;
- EOS and corrupted input behavior;
- seek requirements.

Container support is not codec support. `mp4`, `mov` and `avi` validation must probe both.

### VID-009 — bus/error handling

Every native worker must map:

- `GST_MESSAGE_ERROR` -> failed job/non-zero worker outcome;
- `GST_MESSAGE_EOS` -> finalize path;
- cancellation -> controlled flush/EOS/teardown;
- timeout -> sanitized failure;
- state transition failure -> readiness/job failure.

Writing output or reaching main-loop exit is not sufficient success evidence.

## 7. NVIDIA DeepStream and NVDEC

### VID-001 — exact-version DeepStream docs

Before code:

- detect installed DeepStream version;
- read matching release notes, migration guide and container docs;
- inspect matching C/C++ samples;
- verify Python bindings status for that release;
- record supported driver/CUDA/TensorRT/GStreamer matrix.

Current online docs may point to a newer release than the host. Never paste newer properties into an older runtime.

### VID-002 — NVDEC

Inspect:

- supported codecs/profiles/bit depths;
- output memory type and color formats;
- decoder surface settings;
- dGPU vs Jetson property differences;
- error concealment/drop behavior;
- frame metadata and timestamp propagation.

Acceptance requires actual `nvv4l2decoder`/NVDEC runtime evidence and absence of full-frame CPU decode in the production route.

### VID-003 — streammux

Inspect:

- old vs new `nvstreammux` behavior;
- batch size and timeout;
- PTS semantics;
- scaling/padding and coordinate mapping;
- source ID/batch ID metadata;
- EOS behavior;
- memory type.

Single uploaded video may still use batching across sampled frames, but batch size is benchmarked. Multi-source examples are not copied blindly.

### VID-004 — inference

Inspect:

- input tensor layout/color/normalization;
- network mode and engine lifecycle;
- custom bbox/landmark parser interface;
- per-batch metadata and `batch_id` offsets;
- object/frame/class metadata ownership;
- secondary inference/object crop limitations;
- raw-output callback cost;
- TensorRT dynamic shape support in the exact plugin release.

Do not perform heavy CUDA allocation, synchronization, JSON writing or network I/O while holding DeepStream metadata locks.

### VID-005/TRK-002 — NvDCF

Inspect:

- low-level tracker library selection;
- feature tensor settings;
- tracker resolution;
- shadow tracking and late activation;
- source/batch metadata;
- state reset at shot boundary/EOS;
- ID allocation and lifetime.

NvDCF raw ID is a tracklet implementation detail. Final MergenVision `trackId` is produced by offline reconciliation.

## 8. TensorRT, CUDA and native hot path

### GPU-001/002 — TensorRT

Inspect and test:

- ONNX parser compatibility;
- explicit/dynamic batch profiles;
- tensor names, shapes and dtypes;
- context-per-worker ownership;
- stream binding;
- device pointer lifetime;
- optimization profile selection;
- FP16 accuracy parity;
- timing cache/engine portability;
- engine/plugin serialization compatibility;
- `trtexec` baseline and application parity.

Engine deserialization alone is not inference success. `trtexec` throughput is not end-to-end video throughput.

### GPU-003 — CUDA

Inspect:

- asynchronous memory-copy requirements;
- pinned host memory;
- stream/event ordering;
- allocator ownership;
- event-fenced buffer reuse;
- avoiding implicit synchronization;
- error checking and device selection;
- resource teardown order.

### GPU-004 — profiling

Nsight/NVTX instrumentation must be bounded and removable. Profile:

- decode;
- detector preprocess/infer/postprocess;
- tracker;
- best-shot selection;
- alignment;
- recognizer;
- CPU boundary;
- persistence.

Final throughput run disables intrusive tracing.

## 9. Detector and recognizer references

### MOD-001 — InsightFace Model Zoo

Use for:

- SCRFD and RetinaFace model definitions/results;
- ArcFace-family model input/output expectations;
- candidate model comparison;
- reference preprocessing/alignment investigation.

Critical license gate:

- Repository code license and downloadable model-weight terms are separate.
- Model Zoo explicitly marks published models for non-commercial research use.
- A successful benchmark does not approve customer deployment.
- Weight source, training-data provenance and commercial permission must be recorded per artifact.

### MOD-002 — SCRFD

Inspect:

- exact KPS variant;
- input resolution and stride heads;
- anchor/center generation;
- landmark order;
- WIDER FACE protocol;
- ONNX export and postprocess parity.

`SCRFD-10G` without KPS is insufficient for canonical five-point alignment unless a separately approved landmark model is added.

### MOD-003 — RetinaFace

Inspect:

- five-landmark output;
- anchor generation;
- coordinate decode;
- NMS;
- multi-scale paper results vs single-resolution deployment;
- preprocessing and color order.

Paper multi-scale accuracy is not automatically the 640x640 runtime result.

### MOD-004 — ArcFace

Inspect:

- training objective vs inference behavior;
- canonical 112x112 five-point alignment;
- embedding dimension;
- normalization and cosine comparison;
- dataset/backbone-specific calibration.

Raw cosine is not calibrated confidence.

### MOD-005 — MagFace

Inspect:

- raw embedding norm as quality signal;
- L2 normalization point;
- backbone and training dataset;
- exact evaluation protocol;
- model weight/license provenance.

MagFace is an A/B candidate, not a guaranteed free accuracy upgrade. If the pipeline normalizes before capturing raw magnitude, quality information is lost.

### MOD-006 — Ultralytics/YOLO-face

Use only if:

- exact checkpoint is genuinely face-trained;
- five landmarks are available or a separately justified alignment stage exists;
- dataset/provenance and commercial license are approved;
- export/TensorRT parser is parity tested;
- WIDER FACE and project golden-set results beat or complement dedicated detectors.

Generic `yolo11n/m/x` object weights are not face detectors. Model size `x` is not automatically optimal merely because VRAM is available.

## 10. Tracking and offline reconciliation

### TRK-001 — ByteTrack

Inspect exact upstream:

- association of high/low confidence detections;
- Kalman state and matching thresholds;
- frame-rate assumptions;
- missing-frame behavior under sampling;
- shot-cut reset;
- license and adapted source files.

ByteTrack was designed as generic multi-object tracking. Face-specific identity continuity still requires embedding-aware offline reconciliation.

### TRK-002 — NvDCF

NvDCF is the NVIDIA-native baseline for local tracklets. Compare it with ByteTrack on the exact same detector outputs and sampled frames.

### Project-specific reconciliation — no upstream is source of truth

MergenVision final identity graph must implement:

```text
raw tracker IDs
 -> contiguous tracklets
 -> per-tracklet quality/best-shot embeddings
 -> known identity evidence + temporal constraints
 -> canonical video track IDs
 -> persistent face IDs
```

Evaluation explicitly measures over-merge and fragmentation. An upstream tracker score cannot replace this end-to-end gate.

## 11. Annotation and evaluation

### EVAL-001 — CVAT

Use for:

- video bbox tracks;
- interpolation;
- identity/occlusion/visibility attributes;
- human review;
- versioned export.

Before adoption:

- pin CVAT version;
- inspect exact native/MOT/COCO export behavior;
- verify frame numbering and coordinate convention;
- implement converter golden tests;
- preserve original video PTS separately when export only contains frame index.

CVAT pre-annotation is not ground truth until reviewed.

### EVAL-002/003 — TrackEval and MOTChallenge

Use HOTA, IDF1, ID switches and fragmentation for **local tracking**. Also add MergenVision-specific canonical reconciliation metrics because scene-cut identity merging is beyond a normal online tracker score.

Pin TrackEval commit and store exact command/config/raw summaries.

### EVAL-004 — WIDER FACE

Use published metrics only to shortlist detector candidates. Final choice uses Friends golden clips and relevant face-size/pose/occlusion buckets. Do not compare models evaluated with different scale/test protocols as if numbers were identical.

## 12. API and job references

### API-001 — Starlette streaming body

Inspect:

- `Request.stream()` one-pass semantics;
- client disconnect detection;
- body size enforcement while streaming;
- cancellation and backpressure;
- interaction with proxy buffering.

Calling `.body()` after stream consumption is not valid. Do not buffer the whole video for convenience.

### API-002 — FastAPI file upload

Inspect `UploadFile` spooling behavior before choosing multipart. A spooled temporary file may be acceptable as explicit seek-required fallback, but it is not the default “zero-copy GPU stream” claim.

### API-003/004 — OpenAPI and Problem Details

Freeze:

- request content types;
- `202` job creation response;
- status/result/cancel contracts;
- stable error code/schema;
- no-face successful response;
- bbox coordinate convention;
- confidence/similarity distinction.

## 13. Probe, storage and persistence

### IO-001/002 — ffprobe/format docs

Use `ffprobe` as validation/test oracle for:

- format/container;
- codec/profile;
- duration;
- timebase/fps;
- width/height;
- streams.

Do not use OpenCV/FFmpeg CPU decode as the production hot path merely because ffprobe is used for metadata.

### DB-001/002 — PostgreSQL and SQLAlchemy

Inspect exact versions for:

- process/job state transitions;
- transaction isolation;
- concurrent job claim/locking;
- async session per task/process;
- cancellation/rollback;
- retention query/index design.

Do not share an async engine/session incorrectly across spawned processes.

### OBJ-001 — MinIO

Inspect:

- streaming/multipart upload;
- abort cleanup;
- checksum/ETag semantics;
- stat/get/range;
- retention deletion;
- idempotent deterministic object keys;
- connection pool/concurrency.

MinIO write must not block native GPU callback. Encoded chunks may be teed through bounded queues.

### VEC-001 — Qdrant

Inspect:

- point IDs and payload indexes;
- cosine/dot normalization behavior;
- `wait` semantics;
- filters for active/model version;
- batch retrieval/upsert;
- pagination/scroll;
- collection aliases/migration.

Do not scroll the full gallery into Python/NumPy for each video. Do not store raw PII or filesystem paths in payload.

## 14. Historical reference — `NVDIAgstreamer`

Pinned reference:

```text
repository: https://github.com/bbariskaya/NVDIAgstreamer
commit: bc2c5c1bb862649b46cda4fe454726eb62c895a4
usage: read-only lessons learned
```

### Reuse as concepts

- GStreamer/DeepStream C++ pipeline;
- `filesrc -> qtdemux -> h264parse -> nvv4l2decoder` hardware decode concept;
- NVMM/nvstreammux/nvinfer/nvtracker composition;
- CUDA five-point alignment direction;
- track best-shot idea;
- compact track/frame metadata.

### Reject/adapt

- file-only H.264/MP4 contract;
- no request-byte-stream ingestion;
- three sequential passes that re-read/re-decode video;
- Python OpenCV resolver/render path;
- every-face/every-frame recognition;
- full Qdrant gallery scroll + NumPy brute force;
- CPU detector postprocess duplication;
- CUDA allocation/synchronization/D2H/JSON I/O inside metadata lock;
- batch parser ignoring `batch_id` offsets;
- unverified detector color/offset/scale config;
- GStreamer ERROR path returning process exit 0;
- local tracker ID treated as final cross-scene identity.

Old output or benchmark cannot be presented as new architecture acceptance.

## 15. Historical reference — Phase 1 image service

Before merge, inspect the approved Phase 1 repository for:

- `known/anonymous/new_anonymous` semantics;
- person/face identity/sample ownership;
- Qdrant point ID/payload contract;
- raw cosine vs calibrated confidence;
- process/event/history contracts;
- MinIO object ownership and retention;
- idempotent cross-store lifecycle;
- API error envelope.

Do not copy:

- known BufferArena/lifecycle bugs;
- per-image bulk identity creation;
- CPU fallbacks;
- benchmark-only shortcuts;
- uncalibrated threshold/confidence behavior.

Phase 2 standalone adapters must be replaceable by Phase 1 ports/contracts at merge time.

## 16. Licensing gate

Every selected component gets a row in a model/dependency license ledger:

| Field | Required |
|---|---|
| project/model | yes |
| source URL | yes |
| pinned tag/commit | yes |
| code license | yes |
| model-weight license | yes |
| training dataset/provenance | yes |
| redistribution allowed | explicit |
| commercial use allowed | explicit |
| attribution/notice | explicit |
| decision owner/date | yes |

Rules:

- Code license does not automatically license weights.
- Research benchmark permission does not equal customer deployment permission.
- Missing/ambiguous commercial permission is a production blocker.
- Model/weight files are not committed to Git.
- `AGPL` or enterprise-license decisions require explicit user/company approval.

## 17. Reference decision log template

Each material decision records:

```markdown
## DEC-XXX — Short title

- Requirement:
- Reference IDs:
- Exact installed versions:
- Upstream URL/tag/commit:
- Files/samples inspected:
- License finding:
- Reused concept:
- Adaptation:
- Rejected behavior:
- Failing test/reproducer:
- Runtime validation:
- Remaining limitation:
```

## 18. Final accountability

Every implementation sprint reports each relevant reference as one of:

- `USED_VERIFIED`: exact docs/source inspected and materially used;
- `REVIEWED_REJECTED`: inspected but rejected with reason;
- `SKIPPED_NOT_RELEVANT`;
- `BLOCKED_VERSION_OR_LICENSE`;
- `NOT_AVAILABLE`.

“Used all references” şeklinde göstermelik toplu iddia yasaktır.
