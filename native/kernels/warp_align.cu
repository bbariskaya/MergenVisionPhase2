#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

namespace mergenvision {

// Normalization constants for ArcFace recognizer input.
// ArcFace models expect (x - 127.5) / 127.5; the detector uses 128.0.
constexpr float kMean = 127.5f;
constexpr float kStd = 127.5f;
constexpr float kOutSize = 112.0f;

__device__ inline float clamp(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

// Bilinear sample from an RGB uint8 image with border constant = 0.
__device__ inline void sample_bilinear(
    const uint8_t* src,
    int h,
    int w,
    float y,
    float x,
    float* r,
    float* g,
    float* b)
{
    float fx = floorf(x);
    float fy = floorf(y);
    float dx = x - fx;
    float dy = y - fy;
    int x0 = static_cast<int>(fx);
    int y0 = static_cast<int>(fy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    uint8_t v00r = 0, v00g = 0, v00b = 0;
    uint8_t v01r = 0, v01g = 0, v01b = 0;
    uint8_t v10r = 0, v10g = 0, v10b = 0;
    uint8_t v11r = 0, v11g = 0, v11b = 0;

    auto fetch = [&](int yy, int xx, uint8_t& rr, uint8_t& gg, uint8_t& bb) {
        if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
            const uint8_t* p = src + (yy * w + xx) * 3;
            rr = p[0];
            gg = p[1];
            bb = p[2];
        }
    };

    fetch(y0, x0, v00r, v00g, v00b);
    fetch(y0, x1, v01r, v01g, v01b);
    fetch(y1, x0, v10r, v10g, v10b);
    fetch(y1, x1, v11r, v11g, v11b);

    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    *r = (w00 * v00r + w01 * v01r + w10 * v10r + w11 * v11r - kMean) / kStd;
    *g = (w00 * v00g + w01 * v01g + w10 * v10g + w11 * v11g - kMean) / kStd;
    *b = (w00 * v00b + w01 * v01b + w10 * v10b + w11 * v11b - kMean) / kStd;
}

__global__ void warp_align_kernel(
    const uint8_t* __restrict__ src,  // [H, W, 3]
    int h,
    int w,
    const float* __restrict__ matrices, // [N, 6] -> M = [[a, -b, tx], [b, a, ty]]
    int n,
    float* __restrict__ dst)            // [N, 3, 112, 112]
{
    int total = n * 3 * 112 * 112;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int x = tid % 112;
    int y = (tid / 112) % 112;
    int c = (tid / (112 * 112)) % 3;
    int i = tid / (112 * 112 * 3);

    const float* M = matrices + i * 6;
    float a = M[0];
    float b = M[3];  // M[3] is b because M = [[a,-b,tx],[b,a,ty]]
    float tx = M[2];
    float ty = M[5];
    float det = a * a + b * b;
    float sx, sy;
    if (det > 0.0f) {
        float inv_det = 1.0f / det;
        sx = inv_det * (a * (x - tx) + b * (y - ty));
        sy = inv_det * (-b * (x - tx) + a * (y - ty));
    } else {
        sx = 0.0f;
        sy = 0.0f;
    }

    float r, g, bval;
    sample_bilinear(src, h, w, sy, sx, &r, &g, &bval);

    float val = 0.0f;
    if (c == 0) val = r;
    else if (c == 1) val = g;
    else val = bval;

    int idx = ((i * 3 + c) * 112 + y) * 112 + x;
    dst[idx] = val;
}

extern "C" int mergenvision_warp_align(
    const uint8_t* d_src,
    int h,
    int w,
    const float* d_matrices,
    int n,
    float* d_dst,
    cudaStream_t stream)
{
    constexpr int block = 256;
    int total = n * 3 * 112 * 112;
    int grid = (total + block - 1) / block;
    warp_align_kernel<<<grid, block, 0, stream>>>(
        d_src, h, w, d_matrices, n, d_dst);
    return cudaGetLastError();
}

} // namespace mergenvision
