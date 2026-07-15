#include "retinaface_postproc.h"
#include "mergenvision_kernels.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace mergenvision {

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        abort();
    }
}

RetinaFacePostproc::RetinaFacePostproc(int input_size, int max_candidates, int device_id, cudaStream_t stream)
    : input_size_(input_size), max_candidates_(max_candidates), device_id_(device_id), stream_(stream) {
    generatePriors();
    checkCuda(cudaSetDevice(device_id_), "set device");
    size_t max_size = max_candidates_ * sizeof(float);

    checkCuda(cudaMalloc(&d_priors_, num_priors_ * 4 * sizeof(float)), "priors");
    checkCuda(cudaMalloc(&d_cand_boxes_, max_candidates_ * 4 * sizeof(float)), "cand boxes");
    checkCuda(cudaMalloc(&d_cand_scores_, max_candidates_ * sizeof(float)), "cand scores");
    checkCuda(cudaMalloc(&d_cand_landmarks_, max_candidates_ * 10 * sizeof(float)), "cand landmarks");
    checkCuda(cudaMalloc(&d_counter_, sizeof(int)), "counter");
    checkCuda(cudaMalloc(&d_sorted_scores_, max_candidates_ * sizeof(float)), "sorted scores");
    checkCuda(cudaMalloc(&d_order_, max_candidates_ * sizeof(int)), "order");
    checkCuda(cudaMalloc(&d_keep_, max_candidates_ * sizeof(uint8_t)), "keep");
    checkCuda(cudaMalloc(&d_out_boxes_, max_candidates_ * 4 * sizeof(float)), "out boxes");
    checkCuda(cudaMalloc(&d_out_landmarks_, max_candidates_ * 10 * sizeof(float)), "out landmarks");
    checkCuda(cudaMalloc(&d_out_scores_, max_candidates_ * sizeof(float)), "out scores");
    checkCuda(cudaMalloc(&d_out_count_, sizeof(int)), "out count");

    checkCuda(cudaMallocHost(&h_out_boxes_, max_candidates_ * 4 * sizeof(float)), "host out boxes");
    checkCuda(cudaMallocHost(&h_out_landmarks_, max_candidates_ * 10 * sizeof(float)), "host out landmarks");
    checkCuda(cudaMallocHost(&h_out_scores_, max_candidates_ * sizeof(float)), "host out scores");
    checkCuda(cudaMallocHost(&h_out_count_, sizeof(int)), "host out count");

    std::vector<float> host_priors(num_priors_ * 4);
    int idx = 0;
    const int steps[] = {8, 16, 32};
    const int min_sizes[][2] = {{16, 32}, {64, 128}, {256, 512}};
    for (int k = 0; k < 3; ++k) {
        int step = steps[k];
        int f_h = (input_size_ + step - 1) / step;
        int f_w = (input_size_ + step - 1) / step;
        for (int i = 0; i < f_h; ++i) {
            for (int j = 0; j < f_w; ++j) {
                for (int m = 0; m < 2; ++m) {
                    float s = static_cast<float>(min_sizes[k][m]) / input_size_;
                    host_priors[idx * 4 + 0] = (j + 0.5f) * step / input_size_;
                    host_priors[idx * 4 + 1] = (i + 0.5f) * step / input_size_;
                    host_priors[idx * 4 + 2] = s;
                    host_priors[idx * 4 + 3] = s;
                    ++idx;
                }
            }
        }
    }
    checkCuda(cudaMemcpyAsync(d_priors_, host_priors.data(), num_priors_ * 4 * sizeof(float), cudaMemcpyHostToDevice, stream_), "priors H2D");
}

RetinaFacePostproc::~RetinaFacePostproc() {
    cudaFree(d_priors_);
    cudaFree(d_cand_boxes_);
    cudaFree(d_cand_scores_);
    cudaFree(d_cand_landmarks_);
    cudaFree(d_counter_);
    cudaFree(d_sorted_scores_);
    cudaFree(d_order_);
    cudaFree(d_keep_);
    cudaFree(d_out_boxes_);
    cudaFree(d_out_landmarks_);
    cudaFree(d_out_scores_);
    cudaFree(d_out_count_);

    cudaFreeHost(h_out_boxes_);
    cudaFreeHost(h_out_landmarks_);
    cudaFreeHost(h_out_scores_);
    cudaFreeHost(h_out_count_);
}

void RetinaFacePostproc::generatePriors() {
    const int steps[] = {8, 16, 32};
    const int min_sizes[][2] = {{16, 32}, {64, 128}, {256, 512}};
    int count = 0;
    for (int k = 0; k < 3; ++k) {
        int f = (input_size_ + steps[k] - 1) / steps[k];
        count += f * f * 2;
    }
    num_priors_ = count;
}

std::vector<FaceDetection> RetinaFacePostproc::processFrame(
    const float* d_loc,
    const float* d_conf,
    const float* d_landms,
    int num_anchors,
    int original_width,
    int original_height,
    float conf_threshold,
    float nms_threshold) {

    if (num_anchors != num_priors_) {
        fprintf(stderr, "Anchor mismatch: %d vs %d\n", num_anchors, num_priors_);
        return {};
    }

    // Initialize candidate buffers so invalid entries are scored/computed safely.
    checkCuda(cudaMemsetAsync(d_cand_boxes_, 0, max_candidates_ * 4 * sizeof(float), stream_), "cand boxes memset");
    checkCuda(cudaMemsetAsync(d_cand_scores_, 0, max_candidates_ * sizeof(float), stream_), "cand scores memset");
    checkCuda(cudaMemsetAsync(d_cand_landmarks_, 0, max_candidates_ * 10 * sizeof(float), stream_), "cand landmarks memset");
    checkCuda(cudaMemsetAsync(d_keep_, 0, max_candidates_ * sizeof(uint8_t), stream_), "keep memset");
    checkCuda(cudaMemsetAsync(d_counter_, 0, sizeof(int), stream_), "counter memset");
    checkCuda(cudaMemsetAsync(d_out_count_, 0, sizeof(int), stream_), "out count memset");

    checkCuda((cudaError_t)mergenvision_retinaface_decode_batch(
        d_loc, d_conf, d_landms, d_priors_,
        1, num_priors_, conf_threshold, 0.1f, 0.2f, max_candidates_,
        d_cand_boxes_, d_cand_scores_, d_cand_landmarks_, d_counter_, stream_), "decode");

    // Sort all candidate slots by descending score; invalid entries are zero.
    checkCuda(cudaMemcpyAsync(d_sorted_scores_, d_cand_scores_, max_candidates_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_), "scores copy");
    checkCuda((cudaError_t)mergenvision_argsort_descending(d_sorted_scores_, d_order_, max_candidates_, stream_), "argsort");

    checkCuda((cudaError_t)mergenvision_nms(
        d_cand_boxes_, d_cand_scores_, d_order_, max_candidates_,
        nms_threshold, conf_threshold, d_keep_, stream_), "nms");

    checkCuda((cudaError_t)mergenvision_scale_clip_compact_xy(
        d_cand_boxes_, d_cand_landmarks_, d_cand_scores_,
        d_order_, d_keep_, max_candidates_,
        static_cast<float>(original_width), static_cast<float>(original_height),
        original_width, original_height, conf_threshold,
        d_out_boxes_, d_out_landmarks_, d_out_scores_, d_out_count_, stream_), "scale");

    // Stage 1: copy only the compact output count and synchronize so we know
    // exactly how many detections were produced. This avoids copying the full
    // detector output to the host every frame (production hot-path contract).
    checkCuda(cudaMemcpyAsync(h_out_count_, d_out_count_, sizeof(int), cudaMemcpyDeviceToHost, stream_), "outcount D2H");
    checkCuda(cudaStreamSynchronize(stream_), "count sync");

    int out_count = *h_out_count_;
    if (out_count > max_candidates_) out_count = max_candidates_;

    // Stage 2: copy only the compact metadata that actually survived NMS.
    if (out_count > 0) {
        checkCuda(cudaMemcpyAsync(h_out_boxes_, d_out_boxes_, out_count * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream_), "boxes D2H");
        checkCuda(cudaMemcpyAsync(h_out_landmarks_, d_out_landmarks_, out_count * 10 * sizeof(float), cudaMemcpyDeviceToHost, stream_), "landmarks D2H");
        checkCuda(cudaMemcpyAsync(h_out_scores_, d_out_scores_, out_count * sizeof(float), cudaMemcpyDeviceToHost, stream_), "scores D2H");
        checkCuda(cudaStreamSynchronize(stream_), "metadata sync");
    }
    if (out_count <= 0) {
        return {};
    }
    if (out_count > max_candidates_) out_count = max_candidates_;

    std::vector<FaceDetection> result;
    result.reserve(out_count);
    for (int i = 0; i < out_count; ++i) {
        FaceDetection d;
        d.x1 = h_out_boxes_[i * 4 + 0];
        d.y1 = h_out_boxes_[i * 4 + 1];
        d.x2 = h_out_boxes_[i * 4 + 2];
        d.y2 = h_out_boxes_[i * 4 + 3];
        for (int k = 0; k < 10; ++k) d.landmarks[k] = h_out_landmarks_[i * 10 + k];
        d.score = h_out_scores_[i];
        result.push_back(d);
    }
    return result;
}

} // namespace mergenvision
