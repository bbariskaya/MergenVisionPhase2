#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

namespace mergenvision {

// Decode RetinaFace R50 anchors per image in a batch.
// Outputs are in detector-input (normalized [0,1]) coordinate space.
__global__ void retinaface_decode_batch_kernel(
    const float* loc,      // [B, A, 4]
    const float* conf,     // [B, A, 2]
    const float* landms,   // [B, A, 10]
    const float* priors,   // [A, 4] center-size [cx,cy,w,h], normalized
    int B,
    int A,
    float conf_threshold,
    float variance0,
    float variance1,
    int max_candidates,
    float* out_boxes,      // [B, max_candidates, 4]
    float* out_scores,     // [B, max_candidates]
    float* out_landmarks,  // [B, max_candidates, 10]
    int* counters)         // [B]
{
    int total_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * A;
    if (total_idx >= total) return;

    int b = total_idx / A;
    int a = total_idx % A;

    float score = conf[b * A * 2 + a * 2 + 1];
    if (score < conf_threshold) return;

    int pos = atomicAdd(counters + b, 1);
    if (pos >= max_candidates) return;

    const float* prior = priors + a * 4;
    float pcx = prior[0];
    float pcy = prior[1];
    float pw = prior[2];
    float ph = prior[3];

    const float* l = loc + b * A * 4 + a * 4;
    float cx = pcx + l[0] * variance0 * pw;
    float cy = pcy + l[1] * variance0 * ph;
    float w = pw * expf(l[2] * variance1);
    float h = ph * expf(l[3] * variance1);

    float* box = out_boxes + b * max_candidates * 4 + pos * 4;
    box[0] = fmaxf(cx - 0.5f * w, 0.0f);
    box[1] = fmaxf(cy - 0.5f * h, 0.0f);
    box[2] = fminf(cx + 0.5f * w, 1.0f);
    box[3] = fminf(cy + 0.5f * h, 1.0f);

    out_scores[b * max_candidates + pos] = score;

    const float* lm = landms + b * A * 10 + a * 10;
    float* ol = out_landmarks + b * max_candidates * 10 + pos * 10;
    for (int k = 0; k < 5; ++k) {
        ol[k * 2 + 0] = pcx + lm[k * 2 + 0] * variance0 * pw;
        ol[k * 2 + 1] = pcy + lm[k * 2 + 1] * variance0 * ph;
    }
}

extern "C" int mergenvision_retinaface_decode_batch(
    const float* d_loc,
    const float* d_conf,
    const float* d_landms,
    const float* d_priors,
    int batch,
    int num_anchors,
    float conf_threshold,
    float variance0,
    float variance1,
    int max_candidates,
    float* d_out_boxes,
    float* d_out_scores,
    float* d_out_landmarks,
    int* d_counters,
    cudaStream_t stream)
{
    int total = batch * num_anchors;
    constexpr int threads = 256;
    int blocks = (total + threads - 1) / threads;
    retinaface_decode_batch_kernel<<<blocks, threads, 0, stream>>>(
        d_loc, d_conf, d_landms, d_priors,
        batch, num_anchors, conf_threshold,
        variance0, variance1, max_candidates,
        d_out_boxes, d_out_scores, d_out_landmarks, d_counters);
    return cudaGetLastError();
}

} // namespace mergenvision
