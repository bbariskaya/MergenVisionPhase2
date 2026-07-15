#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

namespace mergenvision {

__global__ void l2_normalize_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    float epsilon,
    int* __restrict__ status)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* x = input + row * cols;
    float* y = output + row * cols;

    float norm_sq = 0.0f;
    bool finite = true;
    for (int i = 0; i < cols; ++i) {
        float v = x[i];
        if (!isfinite(v)) {
            finite = false;
        }
        norm_sq += v * v;
    }

    if (!finite) {
        atomicOr(status, 1);  // non-finite input flag
        for (int i = 0; i < cols; ++i) y[i] = 0.0f;
        return;
    }

    if (norm_sq <= epsilon) {
        atomicOr(status, 2);  // zero/near-zero norm flag
        for (int i = 0; i < cols; ++i) y[i] = 0.0f;
        return;
    }

    float inv_norm = rsqrtf(norm_sq);
    for (int i = 0; i < cols; ++i) {
        y[i] = x[i] * inv_norm;
    }
}

extern "C" int mergenvision_l2_normalize(
    const float* d_input,
    float* d_output,
    int rows,
    int cols,
    float epsilon,
    int* d_status,
    cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return 0;

    // The wrapper owns status initialization so stale flags are never reused.
    cudaError_t err = cudaMemsetAsync(d_status, 0, sizeof(int), stream);
    if (err != cudaSuccess) return static_cast<int>(err);

    constexpr int block = 256;
    int grid = (rows + block - 1) / block;
    l2_normalize_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, rows, cols, epsilon, d_status);
    return cudaGetLastError();
}

} // namespace mergenvision
