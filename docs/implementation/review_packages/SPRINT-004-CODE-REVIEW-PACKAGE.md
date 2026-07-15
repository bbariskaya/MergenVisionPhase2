# Sprint 04 Code Review Package — Detector Batching + Native Annotated Render

## Final verdict

- **Detector batching / render / non-tracker pipeline:** PASS
- **NvDCF tracker correctness for raw `object_id` assignment:** KNOWN_BROKEN / DEFERRED

`make phase2-sprint-04-acceptance` completed with exit 0 on the final run.

---

## Objective

Implement true temporal batched RetinaFace inference and an optional native GPU-annotated MP4 render branch, while keeping the old batch-1 tracker-capable path intact.

---

## Changed source map

| File | Change |
|------|--------|
| `Makefile` | New Sprint 04 targets: `backend-batch-invariants`, `backend-cli-tracker-reject`, `backend-batch-parity`, `backend-batch-determinism`, `backend-render-parity`, `backend-batch-benchmark`, `phase2-sprint-04-acceptance`. |
| `backend/native/plugins/gst-nvdsretinaface/gstnvdsretinaface.cpp` | POD-style C++ lifetime (`Impl*` with explicit new/delete), batched TensorRT enqueue per actual batch, landmark metadata re-attached via owned `NvDsUserMeta` copy/release callbacks. |
| `backend/native/worker/main.cpp` | Worker options parsing (`--batch-size`, `--tracker off|<config>`, `--render`, `--annotated-output`), runtime preprocess/streammux config generation, render branch (`nvdsosd` + `nvv4l2h264enc` + `qtmux` + `filesink` + `nvstreamdemux`), metadata lock/IO split, buffer-pool sizing, `MV_MUX_POOL_SIZE` / `MV_BATCH_PUSH_TIMEOUT_US` env overrides, tracker+batch>1 rejection with `MV_ALLOW_TRACKER_BATCH` experimental override. |
| `backend/tools/benchmark_correctness_matrix.py` | Manifest nested under `result["manifest"]` so `tracker`/`render` string values do not overwrite request booleans; `requested_batch_size`/`requested_tracker`/`requested_render` used for aggregation; `--from-report` re-aggregation with normalization for old/new reports. |
| `backend/tests/integration/test_batch_invariants.py` | New source-level batch-N invariants. |
| `backend/tests/native/test_batch_detection_parity.py` | New parity test (batch 1 vs 2 vs 8, tracker off). |
| `backend/tests/native/test_batch_determinism.py` | New determinism test (5 runs, batch=8). |
| `backend/tests/native/test_render_parity.py` | New render parity test (50 frames, duration, no fake track IDs). |
| `backend/tests/native/test_batch_benchmark.py` | New benchmark gate (Friends.mp4 batch=1 vs batch=16). |
| `backend/tests/native/test_cli_tracker_batch_reject.py` | New CLI rejection gate for tracker+batch>1. |
| `backend/tests/unit/test_benchmark_matrix_summary.py` | New unit test guarding report aggregation semantics. |
| `backend/tools/sprint04_pool_ab.py` | Targeted pool=16 vs pool=128 A/B harness for render path. |
| `backend/tools/sprint04_tracker_diagnostic.py` | Tracker-on diagnostic that reports sentinel/UNTRACKED counts and duplicate-ID frames. |

---

## Runtime evidence

### Final acceptance command

```bash
make phase2-sprint-04-acceptance
```

Result: exit 0, full log at `backend/out/sprint04_acceptance_final.log`.

Key extracts:

```text
Batch detection parity PASSED (50 frames, batches [1, 2, 8])
Batch determinism PASSED (50 frames across 5 runs, batch=8)
Render parity PASSED (frames=50, duration=2.041s)
batch=1:  240.8 FPS (6665 frames, 27.68s)
batch=16: 469.4 FPS (6665 frames, 14.20s)
speedup:  1.95x
Batch benchmark PASSED
Backend video smoke: 6665 frames, 8977 detections OK
Phase2 Sprint 04 acceptance PASSED
```

### Buffer-pool A/B (render path, Friends.mp4, batch=8, tracker=off, render=on)

Command:

```bash
python3 backend/tools/sprint04_pool_ab.py
```

Result: `backend/out/sprint04_pool_ab/pool_ab_report.json`

| Metric | pool=16 | pool=128 |
|--------|---------|----------|
| completed | True | True |
| frames | 6665 | 6665 |
| avg_batch | 7.99 | 7.99 |
| worker wall | 17.48 s | 17.37 s |
| peak GPU mem | 37416 MiB | 37640 MiB |

Throughput regression vs pool=128: **0.61%**. Decision: keep pool=16 (or higher) as the default render-path floor.

`main.cpp` default was updated to `mux_pool_size = std::max(16, opts.batch_size * 2)` for render mode.

### Tracker-on diagnostic

Command:

```bash
python3 backend/tools/sprint04_tracker_diagnostic.py
```

Result: `backend/out/tracker_diagnostic/tracker_diagnostic_report.json`

- Both batch=1 and batch=8 tracker-on runs produced **8977 detections with a single unique `track_id` value**.
- That value is `18446744073709551615` / `0xFFFFFFFFFFFFFFFF`, which is `UNTRACKED_OBJECT_ID`.
- **All 8977 detections are tagged as untracked**, not as a single real tracker ID.
- Duplicate-ID frames: 2023 in both batch sizes.
- Verdict: `TRACKER_DID_NOT_ASSIGN_IDS`.

This proves the failure is at the tracker ID-assignment contract, not a batching-specific collapse. The serialization probe is downstream of `nvtracker` and receives the correct raw `object_id` field; the tracker itself never assigns a real ID.

---

## Known limitations / deferred work

1. **NvDCF tracker correctness is broken and deferred.** The current `tracker_NvDCF_mergen.yml` results in every detected object carrying `UNTRACKED_OBJECT_ID`. Tracker+batch>1 remains behind the `MV_ALLOW_TRACKER_BATCH=1` experimental override.
2. **Canonical identity / reconciliation is out of scope.** Raw tracker IDs are not exposed as final `trackId`.
3. **Audio is not preserved** in the annotated render.
4. **GPU-annotated output is H.264 only** via `nvv4l2h264enc`.

---

## Security / privacy

- No video/model/engine files were committed.
- No secrets or filesystem paths are emitted in public JSONL output.
- `git diff --check` passes.

---

## Questions for reviewer

1. Should the `MV_ALLOW_TRACKER_BATCH` env override be removed entirely, or kept as a gated experiment until tracker contract is fixed?
2. Is the pool=16 default for render mode acceptable given the A/B evidence, or should a higher floor be retained for batch sizes not yet profiled?
3. Should Sprint 05 pull the tracker-fix into its scope, or keep tracker as KNOWN_BROKEN until a dedicated tracker sprint?

---

## Outcome

- Detector/render Sprint 04 acceptance: **PASS**
- Tracker raw-ID correctness: **KNOWN_BROKEN / DEFERRED**
