#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

namespace mergenvision {

// Standard continuous-coordinate IoU for normalized [0,1] boxes.
__device__ inline float iou_normalized(const float* a, const float* b) {
    float x1 = fmaxf(a[0], b[0]);
    float y1 = fmaxf(a[1], b[1]);
    float x2 = fminf(a[2], b[2]);
    float y2 = fminf(a[3], b[3]);
    float w = fmaxf(0.0f, x2 - x1);
    float h = fmaxf(0.0f, y2 - y1);
    float inter = w * h;
    float area_a = fmaxf(0.0f, a[2] - a[0]) * fmaxf(0.0f, a[3] - a[1]);
    float area_b = fmaxf(0.0f, b[2] - b[0]) * fmaxf(0.0f, b[3] - b[1]);
    float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// Exact sequential NMS over a score-descending candidate list.
// One block per image; only thread 0 performs the O(N^2) scan.
// Invalid/zero-area/low-score boxes are skipped and do not suppress others.
// The caller must provide a stable sort (ties broken by original index).
extern "C" __shared__ int nms_kept_indices[];

__global__ void nms_exact_kernel(
    const float* __restrict__ boxes,  // [N, 4]
    const float* __restrict__ scores, // [N]
    const int* __restrict__ order,    // sorted indices
    int n,
    float iou_threshold,
    float score_threshold,
    uint8_t* __restrict__ keep)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int* kept = nms_kept_indices;
    int kept_count = 0;
    for (int i = 0; i < n; ++i) {
        int idx_i = order[i];
        float score = scores[idx_i];
        if (score <= score_threshold) break; // sorted descending

        const float* box_i = boxes + idx_i * 4;
        if (box_i[2] <= box_i[0] || box_i[3] <= box_i[1]) continue;

        bool suppress = false;
        for (int k = 0; k < kept_count; ++k) {
            int j = kept[k];
            int idx_j = order[j];
            if (iou_normalized(box_i, boxes + idx_j * 4) > iou_threshold) {
                suppress = true;
                break;
            }
        }
        if (!suppress) {
            kept[kept_count] = i;
            ++kept_count;
        }
    }

    for (int i = 0; i < n; ++i) {
        keep[i] = 0;
    }
    for (int k = 0; k < kept_count; ++k) {
        keep[kept[k]] = 1;
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
    size_t smem = static_cast<size_t>(n) * sizeof(int);
    nms_exact_kernel<<<1, 1, smem, stream>>>(
        d_boxes, d_scores, d_order, n, iou_threshold, score_threshold, d_keep);
    return cudaGetLastError();
}

} // namespace mergenvision
