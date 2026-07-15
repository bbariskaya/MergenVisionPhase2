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

void RetinaFacePostproc::ensureBuffers(int batch) {
    if (batch <= max_batch_alloc_) return;

    cudaSetDevice(device_id_);
    size_t per = static_cast<size_t>(max_candidates_);

    cudaFree(d_cand_boxes_); d_cand_boxes_ = nullptr;
    cudaFree(d_cand_scores_); d_cand_scores_ = nullptr;
    cudaFree(d_cand_landmarks_); d_cand_landmarks_ = nullptr;
    cudaFree(d_counter_); d_counter_ = nullptr;
    cudaFree(d_sorted_scores_); d_sorted_scores_ = nullptr;
    cudaFree(d_order_); d_order_ = nullptr;
    cudaFree(d_keep_); d_keep_ = nullptr;
    cudaFree(d_out_boxes_); d_out_boxes_ = nullptr;
    cudaFree(d_out_landmarks_); d_out_landmarks_ = nullptr;
    cudaFree(d_out_scores_); d_out_scores_ = nullptr;
    cudaFree(d_out_count_); d_out_count_ = nullptr;

    cudaFreeHost(h_out_boxes_); h_out_boxes_ = nullptr;
    cudaFreeHost(h_out_landmarks_); h_out_landmarks_ = nullptr;
    cudaFreeHost(h_out_scores_); h_out_scores_ = nullptr;
    cudaFreeHost(h_out_count_); h_out_count_ = nullptr;

    checkCuda(cudaMalloc(&d_cand_boxes_, batch * per * 4 * sizeof(float)), "cand boxes");
    checkCuda(cudaMalloc(&d_cand_scores_, batch * per * sizeof(float)), "cand scores");
    checkCuda(cudaMalloc(&d_cand_landmarks_, batch * per * 10 * sizeof(float)), "cand landmarks");
    checkCuda(cudaMalloc(&d_counter_, batch * sizeof(int)), "counter");
    checkCuda(cudaMalloc(&d_sorted_scores_, batch * per * sizeof(float)), "sorted scores");
    checkCuda(cudaMalloc(&d_order_, batch * per * sizeof(int)), "order");
    checkCuda(cudaMalloc(&d_keep_, batch * per * sizeof(uint8_t)), "keep");
    checkCuda(cudaMalloc(&d_out_boxes_, batch * per * 4 * sizeof(float)), "out boxes");
    checkCuda(cudaMalloc(&d_out_landmarks_, batch * per * 10 * sizeof(float)), "out landmarks");
    checkCuda(cudaMalloc(&d_out_scores_, batch * per * sizeof(float)), "out scores");
    checkCuda(cudaMalloc(&d_out_count_, batch * sizeof(int)), "out count");

    checkCuda(cudaMallocHost(&h_out_boxes_, batch * per * 4 * sizeof(float)), "host out boxes");
    checkCuda(cudaMallocHost(&h_out_landmarks_, batch * per * 10 * sizeof(float)), "host out landmarks");
    checkCuda(cudaMallocHost(&h_out_scores_, batch * per * sizeof(float)), "host out scores");
    checkCuda(cudaMallocHost(&h_out_count_, batch * sizeof(int)), "host out count");

    max_batch_alloc_ = batch;
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

std::vector<std::vector<FaceDetection>> RetinaFacePostproc::processBatch(
    const float* d_loc,
    const float* d_conf,
    const float* d_landms,
    int num_anchors,
    int batch,
    const std::vector<std::pair<int, int>>& original_dims,
    float conf_threshold,
    float nms_threshold) {

    std::vector<std::vector<FaceDetection>> per_frame;
    if (batch <= 0) return per_frame;
    per_frame.resize(batch);
    if (original_dims.size() != static_cast<size_t>(batch)) {
        fprintf(stderr, "RetinaFacePostproc::processBatch dim count mismatch: %zu vs %d\n",
                original_dims.size(), batch);
        return per_frame;
    }

    ensureBuffers(batch);

    if (num_anchors != num_priors_) {
        fprintf(stderr, "Anchor mismatch: %d vs %d\n", num_anchors, num_priors_);
        return per_frame;
    }

    const size_t per = static_cast<size_t>(max_candidates_);

    checkCuda(cudaMemsetAsync(d_cand_boxes_, 0, batch * per * 4 * sizeof(float), stream_), "cand boxes memset");
    checkCuda(cudaMemsetAsync(d_cand_scores_, 0, batch * per * sizeof(float), stream_), "cand scores memset");
    checkCuda(cudaMemsetAsync(d_cand_landmarks_, 0, batch * per * 10 * sizeof(float), stream_), "cand landmarks memset");
    checkCuda(cudaMemsetAsync(d_keep_, 0, batch * per * sizeof(uint8_t), stream_), "keep memset");
    checkCuda(cudaMemsetAsync(d_counter_, 0, batch * sizeof(int), stream_), "counter memset");
    checkCuda(cudaMemsetAsync(d_out_count_, 0, batch * sizeof(int), stream_), "out count memset");

    checkCuda((cudaError_t)mergenvision_retinaface_decode_batch(
        d_loc, d_conf, d_landms, d_priors_,
        batch, num_priors_, conf_threshold, 0.1f, 0.2f, max_candidates_,
        d_cand_boxes_, d_cand_scores_, d_cand_landmarks_, d_counter_, stream_), "decode batch");

    for (int b = 0; b < batch; ++b) {
        int offset = b * max_candidates_;
        checkCuda(cudaMemcpyAsync(
            d_sorted_scores_ + offset,
            d_cand_scores_ + offset,
            per * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream_), "scores copy per frame");

        checkCuda((cudaError_t)mergenvision_argsort_descending(
            d_sorted_scores_ + offset,
            d_order_ + offset,
            max_candidates_,
            stream_), "argsort per frame");

        checkCuda((cudaError_t)mergenvision_nms(
            d_cand_boxes_ + offset * 4,
            d_cand_scores_ + offset,
            d_order_ + offset,
            max_candidates_,
            nms_threshold,
            conf_threshold,
            d_keep_ + offset,
            stream_), "nms per frame");

        int ow = original_dims[b].first;
        int oh = original_dims[b].second;
        checkCuda((cudaError_t)mergenvision_scale_clip_compact_xy(
            d_cand_boxes_ + offset * 4,
            d_cand_landmarks_ + offset * 10,
            d_cand_scores_ + offset,
            d_order_ + offset,
            d_keep_ + offset,
            max_candidates_,
            static_cast<float>(ow),
            static_cast<float>(oh),
            ow, oh,
            conf_threshold,
            d_out_boxes_ + offset * 4,
            d_out_landmarks_ + offset * 10,
            d_out_scores_ + offset,
            d_out_count_ + b,
            stream_), "scale per frame");
    }

    checkCuda(cudaMemcpyAsync(h_out_count_, d_out_count_, batch * sizeof(int),
                              cudaMemcpyDeviceToHost, stream_), "counts D2H");
    checkCuda(cudaStreamSynchronize(stream_), "count sync");

    for (int b = 0; b < batch; ++b) {
        int count = h_out_count_[b];
        if (count < 0) count = 0;
        if (count > max_candidates_) count = max_candidates_;
        if (count == 0) continue;

        int offset = b * max_candidates_;
        checkCuda(cudaMemcpyAsync(h_out_boxes_ + offset * 4,
                                  d_out_boxes_ + offset * 4,
                                  count * 4 * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_), "boxes D2H per frame");
        checkCuda(cudaMemcpyAsync(h_out_landmarks_ + offset * 10,
                                  d_out_landmarks_ + offset * 10,
                                  count * 10 * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_), "landmarks D2H per frame");
        checkCuda(cudaMemcpyAsync(h_out_scores_ + offset,
                                  d_out_scores_ + offset,
                                  count * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_), "scores D2H per frame");
    }
    checkCuda(cudaStreamSynchronize(stream_), "metadata sync");

    for (int b = 0; b < batch; ++b) {
        int count = h_out_count_[b];
        if (count < 0) count = 0;
        if (count > max_candidates_) count = max_candidates_;
        int offset = b * max_candidates_;
        per_frame[b].reserve(count);
        for (int i = 0; i < count; ++i) {
            FaceDetection d;
            d.x1 = h_out_boxes_[(offset + i) * 4 + 0];
            d.y1 = h_out_boxes_[(offset + i) * 4 + 1];
            d.x2 = h_out_boxes_[(offset + i) * 4 + 2];
            d.y2 = h_out_boxes_[(offset + i) * 4 + 3];
            for (int k = 0; k < 10; ++k) {
                d.landmarks[k] = h_out_landmarks_[(offset + i) * 10 + k];
            }
            d.score = h_out_scores_[offset + i];
            per_frame[b].push_back(d);
        }
    }
    return per_frame;
}

} // namespace mergenvision
