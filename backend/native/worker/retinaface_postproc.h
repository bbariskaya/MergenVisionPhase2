#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace mergenvision {

struct FaceDetection {
    float x1, y1, x2, y2;     // original-resolution
    float landmarks[10];      // original-resolution
    float score;
};

class RetinaFacePostproc {
public:
    RetinaFacePostproc(int input_size, int max_candidates, int device_id, cudaStream_t stream);
    ~RetinaFacePostproc();

    // Processes RetinaFace output tensors directly from device memory.
    std::vector<FaceDetection> processFrame(
        const float* d_loc,
        const float* d_conf,
        const float* d_landms,
        int num_anchors,
        int original_width,
        int original_height,
        float conf_threshold,
        float nms_threshold);

    // Batch-aware postprocess. Returns detections per frame in batch order.
    std::vector<std::vector<FaceDetection>> processBatch(
        const float* d_loc,
        const float* d_conf,
        const float* d_landms,
        int num_anchors,
        int batch,
        const std::vector<std::pair<int, int>>& original_dims,
        float conf_threshold,
        float nms_threshold);

    void setDebug(bool v) { debug_ = v; }
    int deviceId() const { return device_id_; }
    cudaStream_t stream() const { return stream_; }
    int maxBatchAlloc() const { return max_batch_alloc_; }

private:
    void generatePriors();
    void ensureBuffers(int batch);

    int input_size_;
    int max_candidates_;
    int device_id_;
    cudaStream_t stream_;
    bool debug_ = false;

    float* d_priors_ = nullptr;
    float* d_cand_boxes_ = nullptr;
    float* d_cand_scores_ = nullptr;
    float* d_cand_landmarks_ = nullptr;
    int* d_counter_ = nullptr;
    float* d_sorted_scores_ = nullptr;
    int* d_order_ = nullptr;
    uint8_t* d_keep_ = nullptr;
    float* d_out_boxes_ = nullptr;
    float* d_out_landmarks_ = nullptr;
    float* d_out_scores_ = nullptr;
    int* d_out_count_ = nullptr;

    // Pinned host buffers for async D2H. Reusing them means no per-frame
    // cudaMalloc/free and exactly one stream synchronize.
    float* h_out_boxes_ = nullptr;
    float* h_out_landmarks_ = nullptr;
    float* h_out_scores_ = nullptr;
    int* h_out_count_ = nullptr;

    int max_batch_alloc_ = 1;

    int num_priors_ = 0;
};

} // namespace mergenvision
