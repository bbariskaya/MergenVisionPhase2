# Runtime Inventory — Milestone A+B

| Item | Value |
|---|---|
| Host GPUs | 3 × Quadro RTX 8000 |
| Driver | 580.105.08 |
| DeepStream container | `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` |
| DeepStream SDK | 9.0.0 |
| CUDA Driver/Runtime | 13.1 |
| TensorRT container | 10.14 |
| cuDNN | 9.17 |
| GStreamer | 1.24.2 |
| nvcc | V13.1.115 |

## Verified elements

- `appsrc` present
- `qtdemux` present
- `nvv4l2decoder` present
- `nvstreammux` present
- `nvinfer` present
- `nvtracker` present

## Notes

- Existing engines were built against TensorRT 10.3 in a different container.
- DeepStream 9.0 runs TensorRT 10.14, therefore the 10.3 engines are expected to fail deserialization. Rebuild from ONNX under DeepStream 9.0 is required and is allowed as `rejected_for_phase2_runtime` + new build.
