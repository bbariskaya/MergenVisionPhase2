#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mergenvision {

/* RAII wrapper around a serialized GlintR100 TensorRT engine.
 *
 * The engine is expected to expose:
 *   input  tensor: [N, 3, 112, 112] float32, name "input.1"
 *   output tensor: [N, 512] float32, name "1333"
 * with one optimization profile whose MAX batch is at least 1.
 *
 * The wrapper allocates internal GPU buffers sized to the engine MAX batch.
 * Callers write face crops into input_buffer(), enqueue inference, then read
 * embeddings from output_buffer().  Both buffers are owned by this class.
 */
class GlintR100Engine {
public:
    struct Contract {
        std::string input_name = "input.1";
        std::string output_name = "1333";
        int input_rank = 4;
        int output_rank = 2;
        nvinfer1::DataType input_dtype = nvinfer1::DataType::kFLOAT;
        nvinfer1::DataType output_dtype = nvinfer1::DataType::kFLOAT;
        int image_h = 112;
        int image_w = 112;
        int embedding_dim = 512;
    };

    GlintR100Engine();
    ~GlintR100Engine();

    // Non-copyable; movable but not implemented (keep simple).
    GlintR100Engine(const GlintR100Engine&) = delete;
    GlintR100Engine& operator=(const GlintR100Engine&) = delete;

    /* Deserialize engine and validate tensor contract.
     * gpu_id selects the CUDA device. engine_path must point to a
     * serialized engine compatible with the linked TensorRT runtime.
     * Returns true on success and false on error (message in error).
     */
    bool load(int gpu_id, const std::string& engine_path, std::string* error);

    bool loaded() const { return engine_ != nullptr; }

    int max_batch() const { return max_batch_; }
    int embedding_dim() const { return contract_.embedding_dim; }
    const std::string& engine_path() const { return engine_path_; }

    /* Pointers to the internal GPU buffers sized for max_batch().
     * Layouts are NCHW [N,3,112,112] and [N,512] respectively.
     * These pointers are stable until the engine is destroyed.
     */
    float* input_buffer() const { return d_input_buffer_; }
    float* output_buffer() const { return d_output_buffer_; }

    /* Run inference for exactly count faces (1 <= count <= max_batch()).
     * Input is read from input_buffer(); output is written to output_buffer().
     * Returns true on success.  The caller must synchronize on stream if
     * reading output on the host.  No implicit synchronize is performed.
     */
    bool enqueue(int count, cudaStream_t stream, std::string* error);

private:
    bool validate_contract(std::string* error);
    bool allocate_buffers(std::string* error);

    int gpu_id_ = 0;
    std::string engine_path_;
    Contract contract_;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    int max_batch_ = 0;
    size_t input_buffer_size_ = 0;
    size_t output_buffer_size_ = 0;

    float* d_input_buffer_ = nullptr;
    float* d_output_buffer_ = nullptr;

    // TensorRT logger; warnings+ are emitted to stderr.
    class Logger;
    std::unique_ptr<Logger> logger_;
};

} // namespace mergenvision
