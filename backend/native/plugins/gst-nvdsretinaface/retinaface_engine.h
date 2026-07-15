#pragma once

#include <cuda_runtime.h>
#include <string>
#include <memory>
#include <vector>

// Forward declare TensorRT types to keep compile times reasonable.
namespace nvinfer1 {
class ICudaEngine;
class IExecutionContext;
class IRuntime;
}

namespace mergenvision {

class RetinaFaceEngine {
public:
    RetinaFaceEngine(const std::string& engine_path, int gpu_id, cudaStream_t stream);
    ~RetinaFaceEngine();

    bool init();

    // Runs inference for one batch. input_device must live on the same CUDA
    // context/device. Returns device pointers to loc/conf/landms outputs and the
    // number of anchors (expected 16800 for 640x640 input).
    bool infer(void* input_device, int batch_size,
               const float** d_loc, const float** d_conf,
               const float** d_landms, int* num_anchors);

    int maxBatchSize() const { return max_batch_size_; }
    int inputSize() const { return input_size_; }

private:
    bool allocateBuffers();
    void destroy();

    std::string engine_path_;
    int gpu_id_;
    cudaStream_t stream_;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    int max_batch_size_ = 1;
    int input_size_ = 640;
    int num_anchors_ = 16800;

    std::string input_name_;

    void* d_input_ = nullptr;
    void* d_loc_ = nullptr;
    void* d_conf_ = nullptr;
    void* d_landms_ = nullptr;

    size_t input_bytes_max_ = 0;
    size_t loc_bytes_max_ = 0;
    size_t conf_bytes_max_ = 0;
    size_t landms_bytes_max_ = 0;

    std::vector<char> engine_blob_;
};

} // namespace mergenvision
