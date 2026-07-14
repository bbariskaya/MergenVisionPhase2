# MergenVision — Approved References

**Document type:** External reference registry  
**Status:** Planning input; not an implementation specification  
**Last reviewed:** 2026-07-12

---

## 1. Purpose

This file records the external repositories and official documentation that may be consulted while planning and implementing MergenVision.

These references exist to prevent arbitrary design and implementation decisions. They must be inspected before choosing an architecture, model integration pattern, video pipeline, visualization approach, or deployment strategy.

A reference being listed here does **not** mean:

- every dependency must be installed;
- code may be copied without review;
- the latest branch is automatically compatible with the project;
- the referenced project becomes the source of truth for MergenVision requirements;
- an example designed for Jetson, a notebook, or a research demo is production-ready;
- licensing, model terms, runtime compatibility, and security review can be skipped.

The project requirements and explicit user decisions remain the controlling source of truth. References are implementation and research aids.

---

## 2. Mandatory Reference-First Workflow

Before proposing an implementation based on any item in this file, the agent must:

1. Read the relevant MergenVision requirements and current planning documents.
2. Explain why the reference is relevant to the current phase and task.
3. Inspect the exact upstream files, documentation pages, examples, tests, and configuration involved.
4. Record the upstream repository URL, branch or tag, commit SHA, access date, and license.
5. Compare the upstream behavior with the MergenVision requirement instead of copying it blindly.
6. Identify what will be reused as a concept, what may be adapted, and what must not be copied.
7. Verify model preprocessing, output contracts, coordinate systems, thresholds, batching, metadata lifetime, and error handling where relevant.
8. Produce an understandable plan and diagrams before implementation.
9. Wait for explicit approval before writing production code.
10. Report the references used, skipped, or rejected and the reason for each decision.

No implementation may be justified only with phrases such as “best practice,” “standard approach,” or “usually done this way.” A concrete requirement, official source, approved upstream implementation, or measured project evidence must support the decision.

---

## 3. Reference Priority

When sources disagree, use this order:

1. MergenVision client and project requirements.
2. Explicit current user decisions.
3. Official documentation matching the installed software version.
4. Official upstream source code and tests at a pinned commit.
5. Approved historical MergenVision implementations that are known to have worked.
6. Other approved open-source implementations.
7. Community discussions and forum posts as supporting evidence only.

An archived or version-specific document must not override current documentation for a different platform or runtime.

---

## 4. Approved Reference Registry

| ID | Reference | Primary relevance | Phase | Authority | License / usage note |
|---|---|---|---|---|---|
| REF-001 | [Meta Segment Anything](https://github.com/facebookresearch/segment-anything) | Promptable segmentation, mask generation, ONNX export patterns | Future segmentation/object-understanding phases | Official upstream repository | Apache-2.0 repository license; dataset and model terms must be reviewed separately |
| REF-002 | [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO object/person detection, segmentation, pose, tracking interfaces, export patterns | Mainly Phase 2 and later | Official upstream repository | AGPL-3.0 or commercial/enterprise licensing; production use requires an explicit license decision |
| REF-003 | [Roboflow Supervision](https://github.com/roboflow/supervision) | Detection containers, annotation, tracking helpers, metrics, visual QA and dataset/video utilities | Phase 1 QA and Phase 2 tooling | Official upstream repository | MIT; adding it as a runtime dependency still requires justification |
| REF-004 | [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) | High-performance deep-learning framework patterns, training, inference, distributed execution and deployment concepts | Research/reference across phases | Official upstream repository | Apache-2.0; not a default MergenVision runtime dependency |
| REF-005 | [DeepFace](https://github.com/serengil/deepface) | Face verification/recognition orchestration, detector/recognizer combinations, thresholding and alignment behavior | Phase 1 | Official upstream repository | DeepFace code is MIT; licenses of wrapped detectors, recognition models and weights must be checked independently |
| REF-006 | [NVIDIA Jetson Accelerated GStreamer Guide — Jetson Linux r34.1](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/Multimedia/AcceleratedGstreamer.html) | Jetson NVMM, decoder, converter, camera and accelerated GStreamer pipeline examples | Phase 2, Jetson-specific reference | Official NVIDIA archived/versioned documentation | Historical and Jetson-specific; not the primary authority for an x86 dGPU or a newer DeepStream installation |
| REF-007 | [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) | Real-time GPU video analytics, GStreamer pipeline design, decode/inference/tracking/metadata/encode architecture | Phase 2 | Official NVIDIA product entry point | DeepStream is an NVIDIA SDK; exact installed-version documentation, samples, compatibility matrix and terms must be used |

---

## 5. Detailed Usage Notes

### REF-001 — Meta Segment Anything

**URL:** https://github.com/facebookresearch/segment-anything

**Use for:**

- future promptable segmentation requirements;
- mask representation and mask-quality concepts;
- image encoder / prompt encoder / mask decoder separation;
- ONNX export and browser/demo architecture patterns;
- future object isolation before downstream computer-vision processing.

**Do not use for:**

- Phase 1 face recognition;
- replacing RetinaFace or ArcFace;
- identity matching;
- adding segmentation scope before the related phase is approved.

**Required inspection before use:**

- exact model variant and checkpoint terms;
- preprocessing and resizing rules;
- coordinate and prompt transformations;
- mask decoder ONNX export limitations;
- memory and latency requirements;
- upstream tests and example notebooks;
- whether the newer SAM 2 repository is more appropriate for a later video requirement.

**Project status:** Approved future reference; not a Phase 1 dependency.

---

### REF-002 — Ultralytics

**URL:** https://github.com/ultralytics/ultralytics

**Use for:**

- future person and object detection;
- YOLO preprocessing, postprocessing and export behavior;
- image, batch and stream inference interfaces;
- detection, segmentation, pose and tracking result structures;
- benchmark and validation workflow ideas;
- future person-context validation around face detections.

**Do not use for:**

- Phase 1 primary face detection unless separately approved;
- replacing a dedicated face detector without evidence;
- silently introducing an AGPL dependency into a proprietary production service;
- using built-in tracking as an automatic replacement for the approved Phase 2 tracker architecture;
- forcing a new YOLO version simply because it is the newest release.

**Required inspection before use:**

- exact tag and model version;
- preprocessing, letterboxing and coordinate restoration;
- NMS and confidence behavior;
- export path and TensorRT compatibility;
- batch and stream behavior;
- model weight terms;
- AGPL-3.0 versus enterprise licensing implications.

**Project status:** Approved Phase 2/later reference. Any production dependency requires an explicit licensing and architecture decision.

---

### REF-003 — Roboflow Supervision

**URL:** https://github.com/roboflow/supervision

**Use for:**

- standard detection/result containers;
- bounding-box, mask and annotation utilities;
- visual debugging and contact-sheet workflows;
- tracking and line/zone helper concepts;
- evaluation metrics and dataset/video utilities;
- building readable QA artefacts without modifying production inference behavior.

**Do not use for:**

- replacing the project domain model automatically;
- making production code dependent on a large utility package without need;
- treating annotation output as recognition evidence by itself;
- hiding model or coordinate-system bugs behind visualization helpers.

**Required inspection before use:**

- exact coordinate conventions;
- mutation/copy behavior of result containers;
- supported image/video formats;
- tracking helper assumptions;
- performance cost of annotation and conversion utilities;
- whether a small internal utility is clearer than adding the dependency.

**Project status:** Approved QA and visualization reference. Runtime adoption is optional and must be justified.

---

### REF-004 — PaddlePaddle

**URL:** https://github.com/PaddlePaddle/Paddle

**Use for:**

- high-performance inference and deployment architecture ideas;
- training/inference separation;
- single-machine and distributed execution patterns;
- data-loader, batching and accelerator abstractions;
- comparison with other deployment frameworks;
- research into future scale and model-serving options.

**Do not use for:**

- introducing a second deep-learning framework into Phase 1 without a measured need;
- replacing the approved model stack merely because an implementation exists in the Paddle ecosystem;
- mixing framework runtimes in one process without compatibility analysis;
- using framework-level examples as proof of MergenVision correctness.

**Required inspection before use:**

- exact Paddle release and supported CUDA stack;
- relevant subproject rather than only the framework root;
- model format and export compatibility;
- inference API lifetime and batching behavior;
- deployment and licensing requirements of the specific model or subproject.

**Project status:** Approved research reference; not a default production dependency.

---

### REF-005 — DeepFace

**URL:** https://github.com/serengil/deepface

**Use for:**

- Phase 1 face-recognition behavior and terminology;
- enrollment, verification and identification flow comparison;
- detector/recognizer adapter patterns;
- alignment and normalization behavior;
- threshold and distance-metric research;
- test-oracle and prototype comparisons across supported detectors and recognizers;
- understanding how known and unknown outcomes are exposed to callers.

**Do not use for:**

- treating the library as the final high-performance production architecture;
- accepting its default detector, threshold or preprocessing without model-specific parity tests;
- assuming the MIT license of DeepFace covers every wrapped model and weight;
- enabling age, gender, emotion, race or other attributes that are outside MergenVision requirements;
- coupling Phase 1 business logic directly to one convenience library.

**Required inspection before use:**

- the exact detector and recognizer adapter code;
- face extraction, alignment and normalization paths;
- distance metrics and threshold tables;
- multi-face behavior;
- model download and caching behavior;
- underlying model and weight licenses;
- reproducibility against the selected RetinaFace and ArcFace artefacts.

**Project status:** Primary Phase 1 behavior/reference implementation, but not automatically the final runtime dependency.

---

### REF-006 — NVIDIA Jetson Accelerated GStreamer Guide, Jetson Linux r34.1

**URL:** https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/Multimedia/AcceleratedGstreamer.html

**Use for:**

- understanding Jetson-specific accelerated GStreamer concepts;
- NVMM caps and zero-copy pipeline ideas;
- hardware decoder, converter, camera and encoder examples;
- Jetson camera pipelines using `nvarguscamerasrc`;
- comparing Jetson multimedia element behavior with workstation pipelines.

**Do not use for:**

- treating Jetson element names and properties as valid on an x86 dGPU system;
- overriding the documentation for the installed DeepStream/GStreamer version;
- copying an archived r34.1 pipeline without checking current plugin availability;
- assuming Jetson memory and decoder behavior is identical to workstation DeepStream.

**Required inspection before use:**

- target platform: Jetson or x86 dGPU;
- installed plugin names via `gst-inspect-1.0`;
- current JetPack/Jetson Linux version if Jetson is used;
- NVMM caps and color-format requirements;
- decoder/converter/encoder compatibility;
- current NVIDIA documentation for the actual deployment platform.

**Project status:** Approved Phase 2 historical/platform-specific reference. It is not the main authority for the current workstation deployment.

---

### REF-007 — NVIDIA DeepStream SDK

**URL:** https://developer.nvidia.com/deepstream-sdk

**Use for:**

- Phase 2 video and stream architecture;
- GStreamer-based ingestion and pipeline composition;
- GPU-accelerated decode, preprocessing, inference, tracking, metadata, OSD and encode;
- multi-stream and multi-sensor processing;
- NvDCF and other supported tracker patterns;
- C/C++ and Python sample applications;
- TensorRT and Triton integration choices;
- official compatibility, deployment and performance guidance.

**Do not use for:**

- starting Phase 2 before Phase 1 is complete and accepted;
- relying only on the marketing/landing page;
- mixing examples from different DeepStream versions;
- guessing plugin properties instead of checking the installed version;
- treating sample applications as production-ready without ownership, teardown and error-path review;
- introducing video-specific entities into Phase 1 implementation unnecessarily.

**Required inspection before use:**

- exact installed DeepStream version;
- matching developer guide, release notes and migration guide;
- platform and operating-system compatibility matrix;
- installed sample applications and plugin sources;
- `gst-inspect-1.0` output for every selected plugin;
- metadata ownership and lifetime;
- streammux mode and memory type;
- model parser contract;
- tracker configuration;
- error and EOS teardown;
- TensorRT, CUDA, driver and GStreamer compatibility.

**Project status:** Primary official Phase 2 reference. No DeepStream implementation should begin until the Phase 2 plan is explicitly approved.

---

## 6. Reference Use by Project Phase

### Phase 1 — Image-based face recognition

Primary references:

- REF-005 DeepFace — behavior, orchestration, detector/recognizer comparison and threshold research.
- Relevant upstream RetinaFace, InsightFace alignment and ArcFace sources must be added to this file before implementation begins.

Supporting references:

- REF-003 Supervision — visual QA and annotation utilities.
- REF-004 PaddlePaddle — comparison or research only when a concrete need exists.

Not active in Phase 1 implementation:

- REF-001 Segment Anything.
- REF-002 Ultralytics, except a separately approved experiment.
- REF-006 Jetson Accelerated GStreamer.
- REF-007 DeepStream SDK.

### Phase 2 — Video and live-stream processing

Primary references:

- REF-007 NVIDIA DeepStream SDK.
- Installed NVIDIA sample applications and matching versioned documentation.

Supporting references:

- REF-006 Jetson Accelerated GStreamer only for Jetson-specific deployments.
- REF-002 Ultralytics for object/person-context requirements after licensing approval.
- REF-003 Supervision for offline QA, metrics and visualization.
- Historical working MergenVision video repositories, after they are registered separately as internal references.

### Later phases — Object detection and segmentation

- REF-002 Ultralytics for object detection, pose, segmentation and related export patterns.
- REF-001 Segment Anything for promptable segmentation.
- REF-003 Supervision for result handling and visual QA.
- REF-004 PaddlePaddle for comparative framework and deployment research.

---

## 7. Licensing Gate

Before code, model weights, configuration, or architecture are adopted, record:

| Field | Required value |
|---|---|
| Reference ID | `REF-...` |
| Repository / document | Exact source |
| Commit / tag / document version | Pinned value |
| Code license | Verified license |
| Model/weight license | Verified separately |
| Dataset license | Verified separately where relevant |
| Intended use | Research, test oracle, runtime dependency, code adaptation, model deployment, etc. |
| Production compatibility | Approved / rejected / needs legal or client review |
| Attribution obligations | Exact requirement |
| Copyleft impact | None / identified / unresolved |
| Decision owner | User/client/project owner |

Special cautions:

- Ultralytics production adoption requires an explicit AGPL-3.0 or enterprise-license decision.
- DeepFace’s MIT license does not automatically cover the licenses of all wrapped models and weights.
- SAM code licensing and SA-1B dataset licensing are separate concerns.
- NVIDIA SDK and container terms must be checked for the exact DeepStream distribution used.

---

## 8. Evidence Required from an Agent

Whenever an agent uses one of these references, the planning or implementation report must include:

```text
REFERENCE_ID=
REFERENCE_URL=
UPSTREAM_COMMIT_OR_VERSION=
FILES_OR_PAGES_READ=
RELEVANT_BEHAVIOR=
PROJECT_REQUIREMENT_SERVED=
CONCEPT_REUSED=
CODE_ADAPTED=
LICENSE_REVIEW=
DIFFERENCES_FROM_UPSTREAM=
TEST_OR_EVIDENCE=
REJECTED_PARTS_AND_REASON=
```

The report must distinguish:

- **read** — the source was actually opened;
- **used** — a decision or implementation was based on it;
- **considered but rejected** — inspected but not adopted;
- **not relevant** — skipped with a reason.

A repository URL alone is not proof that the reference was studied.

---

## 9. Planning Gate Before Implementation

Before any Phase 1 production code is written, the agent must produce and explain:

1. Requirements traceability.
2. Phase 1 scope and non-goals.
3. Phase 1 component diagram.
4. Phase 1 request/data-flow diagrams.
5. PostgreSQL ER diagram.
6. MinIO object-layout diagram.
7. Qdrant collection and payload design.
8. ML pipeline diagram: RetinaFace → landmarks → alignment → ArcFace → vector search.
9. API contract map.
10. UI screen and navigation map.
11. Failure and compensation flows.
12. Phase 2 extension boundaries without implementing Phase 2.
13. Test strategy and acceptance gates.
14. Reference-to-decision traceability.

The diagrams and plan must be understandable to the user. The agent must explain each major box, arrow, table, and boundary in plain language.

Implementation starts only after explicit approval.

---

## 10. Missing References to Add Before Phase 1 Implementation

The current list is useful but incomplete for the selected face-recognition stack. Before Phase 1 implementation begins, research and register the exact approved sources for:

- RetinaFace model and implementation used by the project;
- InsightFace five-point alignment reference;
- ArcFace model and preprocessing/output contract;
- Qdrant official documentation;
- PostgreSQL official documentation;
- MinIO official documentation and S3 API behavior;
- FastAPI official documentation;
- SQLAlchemy 2 and Alembic official documentation;
- React, TypeScript and Vite official documentation;
- Docker Compose official documentation;
- UUIDv7 / RFC 9562 implementation source;
- any exact model weights, ONNX files or TensorRT engines selected for production.

Each added entry must follow the same structure as this registry.

---

## 11. Prohibited Reference Practices

The following are not allowed:

- copying code from search snippets without opening the upstream source;
- citing a repository while using a different implementation;
- using `main` without recording a commit SHA;
- mixing documentation from incompatible versions;
- importing a library only because an example uses it;
- silently changing the detector or recognizer;
- treating a benchmark from different hardware as a project result;
- reporting a sample run as production acceptance;
- ignoring model, weight or dataset licenses;
- adding Phase 2/video complexity during Phase 1 planning;
- writing code before the user understands and approves the diagrams and plan.

---

## 12. Change Control

When this file is updated, record:

| Date | Change | Reason | Approved by |
|---|---|---|---|
| 2026-07-12 | Initial registry created with seven user-provided references | Establish reference-first planning before the new repository is implemented | User approval pending |
