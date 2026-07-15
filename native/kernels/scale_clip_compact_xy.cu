#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

namespace mergenvision {

__global__ void scale_clip_compact_xy_kernel(
    const float* __restrict__ boxes,      // [N, 4] in normalized detector space
    const float* __restrict__ landmarks,  // [N, 10]
    const float* __restrict__ scores,     // [N]
    const int* __restrict__ order,        // argsorted indices, length N
    const uint8_t* __restrict__ keep,     // NMS keep mask, length N
    int n,
    float scale_x,
    float scale_y,
    int img_w,
    int img_h,
    float score_threshold,
    float* __restrict__ out_boxes,        // [K, 4] in original image space
    float* __restrict__ out_landmarks,    // [K, 10]
    float* __restrict__ out_scores,       // [K]
    int* __restrict__ out_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int k = 0;
    for (int i = 0; i < n; ++i) {
        if (!keep[i]) continue;
        int src = order[i];
        float score = scores[src];
        if (score <= score_threshold) break; // sorted descending; rest invalid

        const float* b = boxes + src * 4;
        float x1 = fminf(fmaxf(b[0] * scale_x, 0.0f), static_cast<float>(img_w));
        float y1 = fminf(fmaxf(b[1] * scale_y, 0.0f), static_cast<float>(img_h));
        float x2 = fminf(fmaxf(b[2] * scale_x, 0.0f), static_cast<float>(img_w));
        float y2 = fminf(fmaxf(b[3] * scale_y, 0.0f), static_cast<float>(img_h));
        if (x2 <= x1 || y2 <= y1) continue;

        float* ob = out_boxes + k * 4;
        ob[0] = x1; ob[1] = y1; ob[2] = x2; ob[3] = y2;

        const float* l = landmarks + src * 10;
        float* ol = out_landmarks + k * 10;
        for (int j = 0; j < 5; ++j) {
            float lx = fminf(fmaxf(l[j * 2] * scale_x, 0.0f), static_cast<float>(img_w));
            float ly = fminf(fmaxf(l[j * 2 + 1] * scale_y, 0.0f), static_cast<float>(img_h));
            ol[j * 2] = lx;
            ol[j * 2 + 1] = ly;
        }

        out_scores[k] = score;
        ++k;
    }
    *out_count = k;
}

extern "C" int mergenvision_scale_clip_compact_xy(
    const float* d_boxes,
    const float* d_landmarks,
    const float* d_scores,
    const int* d_order,
    const uint8_t* d_keep,
    int n,
    float scale_x,
    float scale_y,
    int img_w,
    int img_h,
    float score_threshold,
    float* d_out_boxes,
    float* d_out_landmarks,
    float* d_out_scores,
    int* d_out_count,
    cudaStream_t stream)
{
    scale_clip_compact_xy_kernel<<<1, 1, 0, stream>>>(
        d_boxes, d_landmarks, d_scores, d_order, d_keep, n,
        scale_x, scale_y, img_w, img_h, score_threshold,
        d_out_boxes, d_out_landmarks, d_out_scores, d_out_count);
    return cudaGetLastError();
}

} // namespace mergenvision
