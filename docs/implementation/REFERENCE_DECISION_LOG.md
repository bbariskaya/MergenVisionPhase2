# Reference Decision Log

This file records reference-first engineering decisions for MergenVision Phase 2.

## Sprint 06 — ByteTrack adaptation for native face tracklets

### References selected

1. Paper: https://arxiv.org/abs/2110.06864
2. Official repository: https://github.com/FoundationVision/ByteTrack
3. Exact pinned reference commit: `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`
4. Relevant upstream C++ source tree:
   https://github.com/FoundationVision/ByteTrack/tree/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/deploy/TensorRT/cpp
5. License: MIT (see `THIRD_PARTY_NOTICES.md`)
6. NVIDIA DeepStream tracker contract:
   https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_gst-nvtracker.html
7. DeepStream custom metadata:
   https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_plugin_metadata.html
8. GStreamer BaseTransform:
   https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html

### Decision: adapt, not copy

- REJECTED: Directly copying the upstream C++ demo application into the repo.
  It depends on OpenCV `cv::Rect`, uses a fixed `dt=1`, uses a process-global
  static track ID generator, hardcodes thresholds, uses `+1` in IoU, and
  contains demo error handling.

- ADAPTED: The ByteTrack state-machine and two-stage association design are
  reimplemented in `backend/native/tracking/` to satisfy the production contract:
  continuous `xyxy` IoU, PTS-aware Kalman, deterministic per-source IDs,
  multi-source support, optional embedding appearance gating, and structured
  error reporting.

- ADAPTED: The `mvfacetracker` GStreamer element is implemented independently
  using GStreamer `GstBaseTransform` and DeepStream metadata APIs. It calls the
  core tracker and emits versioned `MvTrackletMeta` and bus messages. It does
  not reuse upstream GStreamer plugin code.

### Rationale

The upstream demo is a research artifact optimized for MOT17/MOT20 bounding-box
tracking. MergenVision must support offline video face tracking with
recognition-driven appearance gating, deterministic multi-source IDs, PTS-aware
temporal semantics, and separable canonical identity reconciliation. These
requirements are not present in the upstream demo, so an adaptation is required.

### License treatment

- ByteTrack MIT license notice copied to `THIRD_PARTY_NOTICES.md`.
- Source-of-truth and pinned commit recorded in this file.
- No upstream source files committed verbatim.

### Date
2026-07-15

### Author
MergenVision Phase 2 implementation agent

### Status
APPROVED for Sprint 06 implementation.
