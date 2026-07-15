# MergenVision Backend

Python control plane and native GPU worker for offline video face analysis.

## Layer map

| Concern | Location |
|---|---|
| API routers (future FastAPI) | `app/api/routers/` |
| Application use cases | `app/application/services/` |
| Domain models | `app/domain/` |
| Driven ports | `app/ports/` |
| Concrete adapters | `app/infrastructure/` |
| Native worker source | `native/` |
| Production scripts | `scripts/` |
| Developer tools | `tools/` |
| Tests | `tests/unit/`, `tests/integration/`, `tests/native/` |

## Calling the native worker

The only supported chain is:

```text
router → RunVideoDetectionService → NativeWorkerPort
       → SubprocessNativeWorkerAdapter → backend/native executable
```

`app/infrastructure/native_worker/subprocess_adapter.py` builds the Docker
command, runs it once per video job, and parses structured events from stdout.
`app/infrastructure/native_worker/client.py` is the low-level command builder
used by the adapter.

The CLI (`app/cli.py`) exercises the same chain without an HTTP layer.

## Error and debug logs

- Fatal/typed errors are returned as `NativeJobError` to the caller.
- Native worker stdout is parsed for structured events and final summary.
- Native worker stderr is forwarded to the Python logger at `DEBUG` level.
- Raw subprocess output is included in errors only for diagnostics.

## Adding a new endpoint

1. Define or reuse domain models in `app/domain/`.
2. Declare a port in `app/ports/` if a new external dependency is needed.
3. Implement the use case in `app/application/services/` using only domain
   objects and ports.
4. Provide or reuse an adapter in `app/infrastructure/`.
5. Wire the router in `app/api/routers/` (when FastAPI is introduced).

## What must never go into C++/CUDA

- Database, object-store or message-queue access.
- Business rules such as identity reconciliation, known/anonymous decisions,
  or retention policy.
- HTTP/API request handling.
- Frame-by-frame Python callbacks.

The native worker is a pure data-plane function: video in, compact metadata out.
