#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

namespace mergenvision {

// Inclusive-coordinate IoU (matches official InsightFace SCRFD).
__device__ inline float iou(const float* a, const float* b) {
    float x1 = fmaxf(a[0], b[0]);
    float y1 = fmaxf(a[1], b[1]);
    float x2 = fminf(a[2], b[2]);
    float y2 = fminf(a[3], b[3]);
    float w = fmaxf(0.0f, x2 - x1 + 1.0f);
    float h = fmaxf(0.0f, y2 - y1 + 1.0f);
    float inter = w * h;
    float area_a = (a[2] - a[0] + 1.0f) * (a[3] - a[1] + 1.0f);
    float area_b = (b[2] - b[0] + 1.0f) * (b[3] - b[1] + 1.0f);
    float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// Parallel approximate NMS over a score-sorted candidate list.
// Thread i processes sorted position i. Invalid entries (score <= threshold or
// non-positive area) are not kept and do not suppress valid boxes.
__global__ void nms_kernel(
    const float* __restrict__ boxes,  // [N, 4]
    const float* __restrict__ scores, // [N]
    const int* __restrict__ order,    // sorted indices (score descending)
    int n,
    float iou_threshold,
    float score_threshold,
    uint8_t* __restrict__ keep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    keep[idx] = 0;
    int idx_i = order[idx];
    const float* box_i = boxes + idx_i * 4;
    if (scores[idx_i] <= score_threshold) return;
    if (box_i[2] <= box_i[0] || box_i[3] <= box_i[1]) return;

    keep[idx] = 1;
    if (idx == 0) return;

    for (int j = 0; j < idx; ++j) {
        int idx_j = order[j];
        if (!keep[j]) continue; // only consider surviving higher-scored boxes
        if (iou(box_i, boxes + idx_j * 4) > iou_threshold) {
            keep[idx] = 0;
            return;
        }
    }
}

extern "C" int mergenvision_nms(
    const float* d_boxes,
    const float* d_scores,
    const int* d_order,
    int n,
    float iou_threshold,
    float score_threshold,
    uint8_t* d_keep,
    cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    nms_kernel<<<blocks, threads, 0, stream>>>(
        d_boxes, d_scores, d_order, n, iou_threshold, score_threshold, d_keep);
    return cudaGetLastError();
}

} // namespace mergenvision
