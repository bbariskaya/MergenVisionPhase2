#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

namespace mergenvision {

// Canonical ArcFace 112x112 five-point template (x,y).
__constant__ float kArcFaceTemplate[10] = {
    38.2946f, 51.6963f,
    73.5318f, 51.5014f,
    56.0252f, 71.7366f,
    41.5493f, 92.3655f,
    70.7299f, 92.2041f,
};

__global__ void similarity_transform_kernel(
    const float* __restrict__ landmarks,  // [N, 5, 2]
    float* __restrict__ matrices,         // [N, 2, 3]
    int n,
    int /* size */,
    int* __restrict__ status)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float* src = landmarks + idx * 10;
    float* M = matrices + idx * 6;

    float sx[5], sy[5];
    bool finite = true;
    for (int i = 0; i < 5; ++i) {
        sx[i] = src[2 * i];
        sy[i] = src[2 * i + 1];
        if (!isfinite(sx[i]) || !isfinite(sy[i])) {
            finite = false;
        }
    }

    if (!finite) {
        atomicOr(status, 1);
        for (int i = 0; i < 6; ++i) M[i] = 0.0f;
        return;
    }

    // Template destination landmarks.
    float dx[5], dy[5];
    for (int i = 0; i < 5; ++i) {
        dx[i] = kArcFaceTemplate[2 * i];
        dy[i] = kArcFaceTemplate[2 * i + 1];
    }

    // Compute centroids.
    float mean_sx = 0.0f, mean_sy = 0.0f;
    float mean_dx = 0.0f, mean_dy = 0.0f;
    for (int i = 0; i < 5; ++i) {
        mean_sx += sx[i];
        mean_sy += sy[i];
        mean_dx += dx[i];
        mean_dy += dy[i];
    }
    mean_sx *= 0.2f; mean_sy *= 0.2f;
    mean_dx *= 0.2f; mean_dy *= 0.2f;

    // Centered coordinates.
    float num_a = 0.0f, num_b = 0.0f, denom = 0.0f;
    for (int i = 0; i < 5; ++i) {
        float xs = sx[i] - mean_sx;
        float ys = sy[i] - mean_sy;
        float xd = dx[i] - mean_dx;
        float yd = dy[i] - mean_dy;
        num_a += xs * xd + ys * yd;
        num_b += xs * yd - ys * xd;
        denom += xs * xs + ys * ys;
    }

    if (denom == 0.0f) {
        atomicOr(status, 2);
        for (int i = 0; i < 6; ++i) M[i] = 0.0f;
        return;
    }

    float a = num_a / denom;
    float b = num_b / denom;
    float tx = mean_dx - a * mean_sx + b * mean_sy;
    float ty = mean_dy - b * mean_sx - a * mean_sy;

    // OpenCV warpAffine 2x3 matrix: M * [x, y, 1]^T
    M[0] = a;  M[1] = -b; M[2] = tx;
    M[3] = b;  M[4] = a;  M[5] = ty;
}

extern "C" int mergenvision_similarity_transform(
    const float* d_landmarks,
    float* d_matrices,
    int n,
    int size,
    int* d_status,
    cudaStream_t stream)
{
    constexpr int block = 256;
    int grid = (n + block - 1) / block;
    similarity_transform_kernel<<<grid, block, 0, stream>>>(
        d_landmarks, d_matrices, n, size, d_status);
    return cudaGetLastError();
}

} // namespace mergenvision
