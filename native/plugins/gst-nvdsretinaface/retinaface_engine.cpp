#include "retinaface_engine.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <string>
#include <iostream>

namespace mergenvision {

static void logError(const std::string& msg) {
    fprintf(stderr, "[RetinaFaceEngine] %s\n", msg.c_str());
}

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

RetinaFaceEngine::RetinaFaceEngine(const std::string& engine_path, int gpu_id, cudaStream_t stream)
    : engine_path_(engine_path), gpu_id_(gpu_id), stream_(stream) {}

RetinaFaceEngine::~RetinaFaceEngine() {
    destroy();
}

void RetinaFaceEngine::destroy() {
    delete context_; context_ = nullptr;
    delete engine_; engine_ = nullptr;
    delete runtime_; runtime_ = nullptr;
    cudaFree(d_loc_); d_loc_ = nullptr;
    cudaFree(d_conf_); d_conf_ = nullptr;
    cudaFree(d_landms_); d_landms_ = nullptr;
}

static int getMaxBatchFromEngine(nvinfer1::ICudaEngine* engine, const char* input_name) {
    int nb_profiles = engine->getNbOptimizationProfiles();
    int max_batch = 1;
    for (int p = 0; p < nb_profiles; ++p) {
        nvinfer1::Dims max_dims = engine->getProfileShape(input_name, p, nvinfer1::OptProfileSelector::kMAX);
        if (max_dims.nbDims > 0 && max_dims.d[0] > max_batch) max_batch = max_dims.d[0];
    }
    if (max_batch < 1) max_batch = 1;
    return max_batch;
}

bool RetinaFaceEngine::init() {
    cudaError_t cuerr = cudaSetDevice(gpu_id_);
    if (cuerr != cudaSuccess) {
        logError(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cuerr));
        return false;
    }

    std::ifstream file(engine_path_, std::ios::binary);
    if (!file) {
        logError("failed to open engine file: " + engine_path_);
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    engine_blob_.resize(size);
    file.read(engine_blob_.data(), size);
    file.close();

    static TrtLogger logger;
    runtime_ = nvinfer1::createInferRuntime(logger);
    if (!runtime_) {
        logError("createInferRuntime failed");
        return false;
    }
    engine_ = runtime_->deserializeCudaEngine(engine_blob_.data(), size);
    if (!engine_) {
        logError("deserializeCudaEngine failed");
        return false;
    }
    context_ = engine_->createExecutionContext();
    if (!context_) {
        logError("createExecutionContext failed");
        return false;
    }

    const char* input_name = nullptr;
    int nb = engine_->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_name = name;
        }
    }
    if (!input_name) {
        logError("no input tensor found in engine");
        return false;
    }
    input_name_ = input_name;

    nvinfer1::Dims input_shape = engine_->getTensorShape(input_name);
    if (input_shape.nbDims != 4) {
        logError("unexpected input shape rank");
        return false;
    }
    max_batch_size_ = getMaxBatchFromEngine(engine_, input_name);
    input_size_ = input_shape.d[2];
    if (input_size_ <= 0) input_size_ = 640;

    const int steps[] = {8, 16, 32};
    num_anchors_ = 0;
    for (int k = 0; k < 3; ++k) {
        int f = (input_size_ + steps[k] - 1) / steps[k];
        num_anchors_ += f * f * 2;
    }

    return allocateBuffers();
}

bool RetinaFaceEngine::allocateBuffers() {
    loc_bytes_max_ = static_cast<size_t>(max_batch_size_) * num_anchors_ * 4 * sizeof(float);
    conf_bytes_max_ = static_cast<size_t>(max_batch_size_) * num_anchors_ * 2 * sizeof(float);
    landms_bytes_max_ = static_cast<size_t>(max_batch_size_) * num_anchors_ * 10 * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&d_loc_, loc_bytes_max_);
    if (err != cudaSuccess) { logError("loc alloc failed"); return false; }
    err = cudaMalloc(&d_conf_, conf_bytes_max_);
    if (err != cudaSuccess) { logError("conf alloc failed"); return false; }
    err = cudaMalloc(&d_landms_, landms_bytes_max_);
    if (err != cudaSuccess) { logError("landms alloc failed"); return false; }

    int nb = engine_->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) continue;
        if (strcmp(name, "loc") == 0) {
            context_->setTensorAddress(name, d_loc_);
        } else if (strcmp(name, "conf") == 0) {
            context_->setTensorAddress(name, d_conf_);
        } else if (strcmp(name, "landms") == 0) {
            context_->setTensorAddress(name, d_landms_);
        } else if (strcmp(name, "landmark") == 0) {
            context_->setTensorAddress(name, d_landms_);
        }
    }
    return true;
}

bool RetinaFaceEngine::infer(void* input_device, int batch_size,
                             const float** d_loc, const float** d_conf,
                             const float** d_landms, int* num_anchors) {
    if (batch_size < 1) batch_size = 1;
    if (batch_size > max_batch_size_) batch_size = max_batch_size_;

    context_->setTensorAddress(input_name_.c_str(), input_device);

    nvinfer1::Dims4 input_dims(batch_size, 3, input_size_, input_size_);
    if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
        logError("setInputShape failed");
        return false;
    }
    if (!context_->allInputDimensionsSpecified()) {
        logError("input dimensions not specified");
        return false;
    }

    if (!context_->enqueueV3(stream_)) {
        logError("enqueueV3 failed");
        return false;
    }

    *d_loc = static_cast<const float*>(d_loc_);
    *d_conf = static_cast<const float*>(d_conf_);
    *d_landms = static_cast<const float*>(d_landms_);
    *num_anchors = num_anchors_;
    return true;
}

} // namespace mergenvision
