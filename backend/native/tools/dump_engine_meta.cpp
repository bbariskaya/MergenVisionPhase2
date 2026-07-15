/* Standalone TensorRT engine metadata inspector.
 *
 * Prints IO tensor names, modes, dtypes, and optimization-profile shapes.
 * Must be compiled and run in a container whose TensorRT version matches the
 * serialized engine.
 */
#include <NvInfer.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

std::string dims_to_string(const nvinfer1::Dims& dims) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i) oss << ",";
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

std::string dtype_to_string(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF: return "float16";
        case nvinfer1::DataType::kINT8: return "int8";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kBOOL: return "bool";
        case nvinfer1::DataType::kUINT8: return "uint8";
        default: return "unknown";
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: dump_engine_meta <engine_file>\n";
        return 1;
    }

    std::ifstream f(argv[1], std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open engine: " << argv[1] << "\n";
        return 1;
    }
    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    Logger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "createInferRuntime failed\n";
        return 1;
    }

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(data.data(), data.size());
    if (!engine) {
        std::cerr << "deserializeCudaEngine failed (TensorRT version mismatch?)\n";
        return 1;
    }

    std::cout << "num_io_tensors: " << engine->getNbIOTensors() << "\n";
    std::cout << "num_optimization_profiles: " << engine->getNbOptimizationProfiles() << "\n";

    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        const auto mode = engine->getTensorIOMode(name);
        const auto dtype = engine->getTensorDataType(name);
        std::cout << "tensor " << i << "\n";
        std::cout << "  name: " << name << "\n";
        std::cout << "  mode: " << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << "\n";
        std::cout << "  dtype: " << dtype_to_string(dtype) << "\n";
        std::cout << "  static_shape: " << dims_to_string(engine->getTensorShape(name)) << "\n";
        for (int p = 0; p < engine->getNbOptimizationProfiles(); ++p) {
            auto mn = engine->getProfileShape(name, p, nvinfer1::OptProfileSelector::kMIN);
            auto opt = engine->getProfileShape(name, p, nvinfer1::OptProfileSelector::kOPT);
            auto mx = engine->getProfileShape(name, p, nvinfer1::OptProfileSelector::kMAX);
            std::cout << "  profile " << p << "\n";
            std::cout << "    min: " << dims_to_string(mn) << "\n";
            std::cout << "    opt: " << dims_to_string(opt) << "\n";
            std::cout << "    max: " << dims_to_string(mx) << "\n";
        }
    }

    return 0;
}
