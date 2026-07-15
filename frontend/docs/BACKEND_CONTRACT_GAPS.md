# Backend API Contract Gaps

## Scope

This UI sprint **only touches `frontend/`**. The backend API described in
`requirements/phase2requirements.md` is **not yet exposed as HTTP endpoints** in
this repository.

## What exists today

- `native/worker/main.cpp` and supporting CUDA kernels run an offline
  DeepStream pipeline: NVDEC → RetinaFace R50 → NvDCF → alignment →
  GlintR100.
- Current worker output is local files in `out/<video>/`:
  - `detections.jsonl`
  - `tracks.json`
  - `run_manifest.json`
- There is no HTTP server, no job state store, no upload endpoint, no SSE, and
  no result JSON matching the public API contract yet.

## Gaps against the frontend contract

| Contract element | Status | Gap detail |
| ---------------- | ------ | ---------- |
| `POST /videos/recognize` | Not implemented | No HTTP upload, multipart/octet-stream handling, or async job creation. |
| `GET /videos/jobs/{jobId}` | Not implemented | No job state store returning `pending/processing/completed/failed/cancelled` or progress fields. |
| `GET /videos/jobs/{jobId}/result` | Not implemented | No canonical-person aggregation (`faceId`, `trackId`, `status`, `name`, `appearances`, `detections`, `confidence`). |
| `DELETE /videos/jobs/{jobId}` | Not implemented | No cancel endpoint. |
| `GET /faces/{faceId}/appearances` | Not implemented | No face-history/index endpoint. |
| `GET /videos/jobs/{jobId}/annotated.mp4` | Not implemented | There is no annotated render endpoint. |
| SSE progress stream | Not implemented | No server-sent events; frontend uses bounded polling fallback even when backend exists. |

## Frontend strategy

- The UI communicates through a typed `ApiClient`.
- `VITE_USE_MOCK_API=true` (default in local development) routes every call to a
  deterministic in-memory mock adapter.
- When a real backend becomes available, set `VITE_USE_MOCK_API=false` and
  `VITE_API_BASE_URL=http://localhost:<port>`. Only the `ApiClient` transport
  changes; components and fixtures stay isolated.
- Mock fixtures explicitly mark data as `MOCK_DATA` so a production build never
  accidentally serves demo data.

## Mock fixtures

The frontend includes deterministic fixtures for:

- a processing job,
- a completed Friends `friendsshort.mp4` job with `Phoebe`, `Rachel`,
  `Chandler`, and unknown tracks,
- a no-face completed job,
- a failed job,
- a cancelled job.

These fixtures are used only by the mock adapter and in tests.
