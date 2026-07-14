# CPU Oracle Fixtures (Test-Only)

Files in this directory are **exclusively** for parity tests and local debugging.
They must **never** be imported or executed in the production worker Docker image.

- `cpu_oracle_retinaface.py` — OpenCV decode + ONNX Runtime inference + reference decode/NMS.
- `cpu_oracle_glintr100.py` — OpenCV alignment crop + ONNX Runtime glintr100 inference.

Results from these oracles are the truth baseline for native GPU output parity gates.
They are excluded from throughput benchmarks.
