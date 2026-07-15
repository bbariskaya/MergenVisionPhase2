#include "nvdsinfer_custom_impl.h"
#include <cstdio>
#include <cstdlib>

// Diagnostic no-op parser. RetinaFace postprocessing runs entirely on the GPU
// from NvDsInferTensorMeta::out_buf_ptrs_dev; the parser is only retained
// because gst-nvinfer requires a parse-bbox-func-name/custom-lib to be
// configured as a primary detector. The host layer.buffer is intentionally
// not inspected here.
extern "C" bool NvDsInferParseCustomRetinaFace(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    static int call_count = 0;
    if (call_count < 3) {
        for (const auto& layer : outputLayersInfo) {
            if (!layer.layerName || !layer.buffer) continue;
            cudaPointerAttributes attr{};
            cudaError_t err = cudaPointerGetAttributes(&attr, layer.buffer);
            fprintf(stderr,
                "[parser] layer=%s host_ptr=%p cuda_attr_type=%d err=%d\n",
                layer.layerName, layer.buffer,
                (err == cudaSuccess) ? static_cast<int>(attr.type) : -1,
                static_cast<int>(err));
        }
        ++call_count;
    }
    // No object meta from CPU parser; compact GPU-side postprocess attaches it.
    objectList.clear();
    return true;
}
