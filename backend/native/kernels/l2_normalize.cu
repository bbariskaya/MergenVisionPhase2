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

    float norm = sqrtf(norm_sq + epsilon);
    if (norm == 0.0f) {
        atomicOr(status, 2);  // zero-norm flag
        norm = 1.0f;
    }
    if (!finite) {
        atomicOr(status, 1);  // non-finite flag
    }

    for (int i = 0; i < cols; ++i) {
        y[i] = x[i] / norm;
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
    constexpr int block = 256;
    int grid = (rows + block - 1) / block;
    l2_normalize_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, rows, cols, epsilon, d_status);
    return cudaGetLastError();
}

} // namespace mergenvision
