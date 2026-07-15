#include "nvdsinfer_custom_impl.h"

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
    (void)outputLayersInfo;
    (void)networkInfo;
    (void)detectionParams;
    // No object meta from CPU parser; compact GPU-side postprocess attaches it.
    objectList.clear();
    return true;
}
