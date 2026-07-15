/* GPU five-point face alignment from pitched RGBA/NVMM surfaces.
 *
 * Operates on a variable number of faces (n) that may come from one or more
 * frame surfaces. Each face is described by:
 *   - an affine matrix that maps destination (112x112) pixels back to the
 *     source RGBA surface (dst -> src);
 *   - a surface index selecting which d_surface_ptrs base pointer to sample;
 *   - the surface pitch, width, and height.
 *
 * Output is a contiguous NCHW float tensor [n, 3, 112, 112] normalized by
 * subtracting 127.5 and dividing by 127.5 (ArcFace convention).
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

namespace mergenvision {

constexpr float kArcMean = 127.5f;
constexpr float kArcStd = 127.5f;
constexpr int kOutSize = 112;

__device__ inline float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ inline void sample_bilinear_rgba(
    const uint8_t* __restrict__ src,
    int h,
    int w,
    int pitch,
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
            const uint8_t* p = src + yy * pitch + xx * 4;
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

    *r = (w00 * v00r + w01 * v01r + w10 * v10r + w11 * v11r - kArcMean) / kArcStd;
    *g = (w00 * v00g + w01 * v01g + w10 * v10g + w11 * v11g - kArcMean) / kArcStd;
    *b = (w00 * v00b + w01 * v01b + w10 * v10b + w11 * v11b - kArcMean) / kArcStd;
}

// One block per face: blockDim=(16,16,1), gridDim=(n,1,1).
// Each thread strides over 112x112 output pixels so we stay within the 1024
// thread-per-block limit and keep the face index as the explicit grid dimension.
__global__ void warp_align_rgba_pitch_kernel(
    const uint8_t* const* __restrict__ d_surface_ptrs,
    const int* __restrict__ d_surface_indices,
    const int* __restrict__ d_pitches,
    const int* __restrict__ d_widths,
    const int* __restrict__ d_heights,
    const float* __restrict__ d_matrices,
    int n,
    float* __restrict__ d_dst)
{
    int face = blockIdx.x;
    if (face >= n) return;

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    const float* M = d_matrices + face * 6;
    float a = M[0];
    float b = M[3];
    float tx = M[2];
    float ty = M[5];
    float det = a * a + b * b;
    float inv_det = 0.0f;
    bool valid_m = det > 0.0f;
    if (valid_m) inv_det = 1.0f / det;

    int surf_idx = d_surface_indices[face];
    const uint8_t* base = d_surface_ptrs[surf_idx];
    int pitch = d_pitches[face];
    int w = d_widths[face];
    int h = d_heights[face];

    for (int y = tidy; y < kOutSize; y += blockDim.y) {
        for (int x = tidx; x < kOutSize; x += blockDim.x) {
            float sx = 0.0f, sy = 0.0f;
            if (valid_m) {
                sx = inv_det * (a * (x - tx) + b * (y - ty));
                sy = inv_det * (-b * (x - tx) + a * (y - ty));
            }
            float r, g, bval;
            sample_bilinear_rgba(base, h, w, pitch, sy, sx, &r, &g, &bval);
            int base_idx = (face * 3 * kOutSize * kOutSize) + (y * kOutSize + x);
            d_dst[base_idx + 0 * kOutSize * kOutSize] = r;
            d_dst[base_idx + 1 * kOutSize * kOutSize] = g;
            d_dst[base_idx + 2 * kOutSize * kOutSize] = bval;
        }
    }
}

extern "C" cudaError_t mergenvision_warp_align_rgba_pitch(
    const uint8_t* const* d_surface_ptrs,
    const int* d_surface_indices,
    const int* d_pitches,
    const int* d_widths,
    const int* d_heights,
    const float* d_matrices,
    int n,
    float* d_dst,
    cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    dim3 block(16, 16, 1);
    dim3 grid(n, 1, 1);
    warp_align_rgba_pitch_kernel<<<grid, block, 0, stream>>>(
        d_surface_ptrs, d_surface_indices, d_pitches, d_widths, d_heights,
        d_matrices, n, d_dst);
    return cudaGetLastError();
}

} // namespace mergenvision
