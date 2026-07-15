/* Native unit tests for mergenvision_l2_normalize.
 *
 * Required semantics:
 *   - finite input with norm_sq > epsilon  -> unit-normalized output
 *   - zero / near-zero / non-finite input  -> zero vector and flagged status
 *   - wrapper owns status initialization (clears stale flags)
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mergenvision_kernels.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

#define CHECK_KERN(call) do { \
    int kerr = (call); \
    if (kerr != 0) { \
        fprintf(stderr, "Kernel error %d at %s:%d\n", kerr, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

int main() {
    constexpr int dim = 512;
    constexpr int rows = 4;
    float* h_input = (float*)malloc(rows * dim * sizeof(float));
    float* h_output = (float*)malloc(rows * dim * sizeof(float));

    // Row 0: normal vector (all 1.0 -> norm sqrt(512)).
    for (int i = 0; i < dim; ++i) h_input[0 * dim + i] = 1.0f;

    // Row 1: zero vector.
    for (int i = 0; i < dim; ++i) h_input[1 * dim + i] = 0.0f;

    // Row 2: near-zero vector.
    for (int i = 0; i < dim; ++i) h_input[2 * dim + i] = (i == 0) ? 1e-8f : 0.0f;

    // Row 3: NaN vector.
    for (int i = 0; i < dim; ++i) h_input[3 * dim + i] = (float)NAN;

    float *d_input, *d_output;
    int* d_status;
    CHECK_CUDA(cudaMalloc(&d_input, rows * dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, rows * dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_status, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, rows * dim * sizeof(float), cudaMemcpyHostToDevice));

    // Pre-seed status with stale flags; wrapper must clear it.
    int stale = 0xFF;
    CHECK_CUDA(cudaMemcpy(d_status, &stale, sizeof(int), cudaMemcpyHostToDevice));

    CHECK_KERN(mergenvision_l2_normalize(d_input, d_output, rows, dim, 1e-12f,
                                         d_status, 0));
    CHECK_CUDA(cudaStreamSynchronize(0));

    CHECK_CUDA(cudaMemcpy(h_output, d_output, rows * dim * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int status = 0;
    CHECK_CUDA(cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost));

    // Row 0: must be unit-normalized.
    float expected0 = 1.0f / sqrtf(512.0f);
    for (int i = 0; i < dim; ++i) {
        CHECK(fabsf(h_output[0 * dim + i] - expected0) < 1e-5f);
    }

    // Row 1: zero -> zero vector, status bit for zero norm.
    for (int i = 0; i < dim; ++i) CHECK(h_output[1 * dim + i] == 0.0f);
    CHECK((status & 2) != 0);

    // Row 2: near-zero -> zero vector, status bit for zero norm.
    for (int i = 0; i < dim; ++i) CHECK(h_output[2 * dim + i] == 0.0f);
    CHECK((status & 2) != 0);

    // Row 3: NaN -> zero vector, status bit for non-finite.
    for (int i = 0; i < dim; ++i) CHECK(h_output[3 * dim + i] == 0.0f);
    CHECK((status & 1) != 0);

    // Reuse the same d_status for a clean vector; stale flags must not remain.
    for (int i = 0; i < dim; ++i) h_input[i] = 0.5f;
    CHECK_CUDA(cudaMemcpy(d_input, h_input, dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_KERN(mergenvision_l2_normalize(d_input, d_output, 1, dim, 1e-12f,
                                         d_status, 0));
    CHECK_CUDA(cudaStreamSynchronize(0));
    CHECK_CUDA(cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(status == 0);

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_status);
    free(h_input); free(h_output);

    if (g_failures) {
        fprintf(stderr, "l2_normalize tests FAILED: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("l2_normalize tests PASSED\n");
    return 0;
}
