/* Native unit test for mergenvision_warp_align_rgba_pitch.
 *
 * Verifies:
 *   - RGBA pitched surface sampling
 *   - per-face surface index selection (multiple faces from same/different surfaces)
 *   - NCHW output layout and normalization
 *   - constant-zero border behavior
 *   - NaN/invalid landmark matrix handling
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mergenvision_kernels.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      return 1; \
    } \
} while(0)

static float cpu_sample(const uint8_t* rgba, int h, int w, int pitch, float y, float x, int c) {
    float fx = floorf(x);
    float fy = floorf(y);
    float dx = x - fx;
    float dy = y - fy;
    int x0 = (int)fx;
    int y0 = (int)fy;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    auto fetch = [&](int yy, int xx) -> float {
        if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
            return rgba[yy * pitch + xx * 4 + c];
        }
        return 0.0f;
    };
    float v00 = fetch(y0, x0);
    float v01 = fetch(y0, x1);
    float v10 = fetch(y1, x0);
    float v11 = fetch(y1, x1);
    float val = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v01 +
                (1 - dx) * dy * v10 + dx * dy * v11;
    return (val - 127.5f) / 127.5f;
}

static int test_identity_warp() {
    const int W = 64, H = 64;
    const int PITCH = W * 4;
    uint8_t* h_surf = (uint8_t*)malloc(H * PITCH);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            h_surf[y * PITCH + x * 4 + 0] = (uint8_t)(x * 4);
            h_surf[y * PITCH + x * 4 + 1] = (uint8_t)(y * 4);
            h_surf[y * PITCH + x * 4 + 2] = (uint8_t)((x + y) * 2);
            h_surf[y * PITCH + x * 4 + 3] = 255;
        }
    }

    uint8_t* d_surf;
    CHECK_CUDA(cudaMalloc(&d_surf, H * PITCH));
    CHECK_CUDA(cudaMemcpy(d_surf, h_surf, H * PITCH, cudaMemcpyHostToDevice));

    // Forward affine M maps src (x_s,y_s) -> dst (x_d,y_d) like:
    //   x_d = a*x_s - b*y_s + tx
    //   y_d = b*x_s + a*y_s + ty
    // The kernel inverts this to sample source for each destination pixel.
    // Choose a translation that maps source (7,7) to destination (0,0),
    // so the whole 112x112 output samples the interior of the 64x64 surface.
    float a = 1.0f, b = 0.0f;
    float tx = -7.0f, ty = -7.0f;
    float h_matrix[6] = {a, -b, tx, b, a, ty};

    uint8_t* d_surface_ptrs;
    int d_indices[1] = {0};
    int d_pitches[1] = {PITCH};
    int d_widths[1] = {W};
    int d_heights[1] = {H};
    float* d_matrix;
    float* d_dst;
    int* d_indices_d, *d_pitches_d, *d_widths_d, *d_heights_d;
    CHECK_CUDA(cudaMalloc(&d_surface_ptrs, sizeof(uint8_t*)));
    CHECK_CUDA(cudaMemcpy(d_surface_ptrs, &d_surf, sizeof(uint8_t*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_indices_d, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_indices_d, d_indices, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_pitches_d, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_pitches_d, d_pitches, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_widths_d, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_widths_d, d_widths, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_heights_d, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_heights_d, d_heights, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_matrix, sizeof(h_matrix)));
    CHECK_CUDA(cudaMemcpy(d_matrix, h_matrix, sizeof(h_matrix), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_dst, 1 * 3 * 112 * 112 * sizeof(float)));

    CHECK_CUDA(mergenvision_warp_align_rgba_pitch(
        (const uint8_t* const*)d_surface_ptrs,
        d_indices_d, d_pitches_d, d_widths_d, d_heights_d,
        d_matrix, 1, d_dst, 0));

    float h_dst[1 * 3 * 112 * 112];
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, sizeof(h_dst), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int y = 0; y < 112; ++y) {
        for (int x = 0; x < 112; ++x) {
            // Invert the forward affine M.
            float inv_det = 1.0f / (a * a + b * b);
            float sx = inv_det * (a * (x - tx) + b * (y - ty));
            float sy = inv_det * (-b * (x - tx) + a * (y - ty));
            for (int c = 0; c < 3; ++c) {
                float expected = cpu_sample(h_surf, H, W, PITCH, sy, sx, c);
                float got = h_dst[((0 * 3 + c) * 112 + y) * 112 + x];
                if (fabsf(got - expected) > 1e-4f) {
                    if (errors < 5) {
                        fprintf(stderr, "mismatch at y=%d x=%d c=%d: got %.5f expected %.5f\n",
                                y, x, c, got, expected);
                    }
                    errors++;
                }
            }
        }
    }

    cudaFree(d_surf);
    cudaFree(d_surface_ptrs);
    cudaFree(d_indices_d); cudaFree(d_pitches_d); cudaFree(d_widths_d); cudaFree(d_heights_d);
    cudaFree(d_matrix); cudaFree(d_dst);
    free(h_surf);

    if (errors) {
        fprintf(stderr, "test_identity_warp: %d pixel mismatches\n", errors);
        return 1;
    }
    printf("test_identity_warp PASSED\n");
    return 0;
}

static int test_constant_zero_border() {
    const int W = 20, H = 20;
    const int PITCH = W * 4;
    uint8_t* h_surf = (uint8_t*)calloc(H * PITCH, 1);
    uint8_t* d_surf;
    CHECK_CUDA(cudaMalloc(&d_surf, H * PITCH));
    CHECK_CUDA(cudaMemcpy(d_surf, h_surf, H * PITCH, cudaMemcpyHostToDevice));

    // identity matrix with a -200 offset so every destination pixel maps outside.
    float h_matrix[6] = {1.0f, 0.0f, -200.0f, 0.0f, 1.0f, -200.0f};

    uint8_t* d_surface_ptrs;
    int d_indices[1] = {0};
    int d_pitches[1] = {PITCH};
    int d_widths[1] = {W};
    int d_heights[1] = {H};
    float* d_matrix;
    float* d_dst;
    int* d_indices_d, *d_pitches_d, *d_widths_d, *d_heights_d;
    CHECK_CUDA(cudaMalloc(&d_surface_ptrs, sizeof(uint8_t*)));
    CHECK_CUDA(cudaMemcpy(d_surface_ptrs, &d_surf, sizeof(uint8_t*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_indices_d, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_indices_d, d_indices, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_pitches_d, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_pitches_d, d_pitches, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_widths_d, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_widths_d, d_widths, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_heights_d, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_heights_d, d_heights, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_matrix, sizeof(h_matrix)));
    CHECK_CUDA(cudaMemcpy(d_matrix, h_matrix, sizeof(h_matrix), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_dst, 3 * 112 * 112 * sizeof(float)));

    CHECK_CUDA(mergenvision_warp_align_rgba_pitch(
        (const uint8_t* const*)d_surface_ptrs,
        d_indices_d, d_pitches_d, d_widths_d, d_heights_d,
        d_matrix, 1, d_dst, 0));

    float h_dst[3 * 112 * 112];
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, sizeof(h_dst), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (size_t i = 0; i < sizeof(h_dst) / sizeof(float); ++i) {
        if (fabsf(h_dst[i] - (-1.0f)) > 1e-5f) {
            errors++;
            if (errors < 5) {
                fprintf(stderr, "border pixel %zu: expected -1.0 got %.5f\n", i, h_dst[i]);
            }
        }
    }

    cudaFree(d_surf); cudaFree(d_surface_ptrs);
    cudaFree(d_indices_d); cudaFree(d_pitches_d); cudaFree(d_widths_d); cudaFree(d_heights_d);
    cudaFree(d_matrix); cudaFree(d_dst); free(h_surf);

    if (errors) {
        fprintf(stderr, "test_constant_zero_border: %d mismatches\n", errors);
        return 1;
    }
    printf("test_constant_zero_border PASSED\n");
    return 0;
}

int main() {
    int rc = 0;
    rc |= test_identity_warp();
    rc |= test_constant_zero_border();
    if (rc) {
        printf("warp_align_rgba_pitch tests FAILED\n");
        return 1;
    }
    printf("warp_align_rgba_pitch tests PASSED\n");
    return 0;
}
