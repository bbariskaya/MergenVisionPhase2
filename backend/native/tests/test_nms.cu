#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

#include "mergenvision_kernels.h"

static int g_failures = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    }
}

static float compute_iou_normalized(const float* a, const float* b) {
    float x1 = fmaxf(a[0], b[0]);
    float y1 = fmaxf(a[1], b[1]);
    float x2 = fminf(a[2], b[2]);
    float y2 = fminf(a[3], b[3]);
    float inter = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float area_a = fmaxf(0.0f, a[2] - a[0]) * fmaxf(0.0f, a[3] - a[1]);
    float area_b = fmaxf(0.0f, b[2] - b[0]) * fmaxf(0.0f, b[3] - b[1]);
    float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

static void test_iou_basics() {
    float a[4] = {0.1f, 0.1f, 0.5f, 0.5f};
    float b[4] = {0.3f, 0.3f, 0.7f, 0.7f};
    float iou = compute_iou_normalized(a, b);
    float expected = (0.5f - 0.3f) * (0.5f - 0.3f)
        / ((0.4f * 0.4f) + (0.4f * 0.4f) - (0.2f * 0.2f));
    check(std::fabs(iou - expected) < 1e-5f, "partial IoU should match expected");

    float same[4] = {0.2f, 0.2f, 0.6f, 0.6f};
    check(std::fabs(compute_iou_normalized(same, same) - 1.0f) < 1e-5f, "identical boxes IoU == 1");

    float c[4] = {0.0f, 0.0f, 0.5f, 0.5f};
    float d[4] = {0.6f, 0.6f, 1.0f, 1.0f};
    check(std::fabs(compute_iou_normalized(c, d)) < 1e-5f, "disjoint boxes IoU == 0");
}

static void run_nms(int n, const float* boxes, const float* scores, float thr,
                    std::vector<uint8_t>* keep,
                    std::vector<int>* out_order = nullptr) {
    keep->assign(n, 0);
    float *d_boxes = nullptr, *d_scores = nullptr;
    int *d_order = nullptr;
    uint8_t* d_keep = nullptr;
    cudaMalloc(&d_boxes, n * 4 * sizeof(float));
    cudaMalloc(&d_scores, n * sizeof(float));
    cudaMalloc(&d_order, n * sizeof(int));
    cudaMalloc(&d_keep, n);

    cudaMemcpy(d_boxes, boxes, n * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, scores, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(0);

    mergenvision_argsort_descending(d_scores, d_order, n, 0);
    cudaStreamSynchronize(0);

    if (out_order) {
        out_order->resize(n);
        cudaMemcpy(out_order->data(), d_order, n * sizeof(int), cudaMemcpyDeviceToHost);
    }

    mergenvision_nms(d_boxes, d_scores, d_order, n, thr, 0.0f, d_keep, 0);
    cudaStreamSynchronize(0);

    cudaMemcpy(keep->data(), d_keep, n, cudaMemcpyDeviceToHost);

    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_order);
    cudaFree(d_keep);
}

static void test_nms_basic() {
    const int n = 4;
    float boxes[n * 4] = {
        0.0f, 0.0f, 0.5f, 0.5f,  // 0: high score, keep
        0.1f, 0.1f, 0.6f, 0.6f,  // 1: overlaps 0 heavily, suppress
        0.7f, 0.7f, 1.0f, 1.0f,  // 2: disjoint, keep
        0.0f, 0.0f, 0.0f, 0.0f,  // 3: zero area, skip
    };
    float scores[n] = {0.9f, 0.85f, 0.7f, 0.6f};
    std::vector<uint8_t> keep;
    std::vector<int> order;
    run_nms(n, boxes, scores, 0.4f, &keep, &order);

    // keep positions are in CUDA argsort order.
    int kept0 = keep[std::find(order.begin(), order.end(), 0) - order.begin()];
    int kept1 = keep[std::find(order.begin(), order.end(), 1) - order.begin()];
    int kept2 = keep[std::find(order.begin(), order.end(), 2) - order.begin()];
    int kept3 = keep[std::find(order.begin(), order.end(), 3) - order.begin()];

    check(kept0 == 1, "box 0 should be kept");
    check(kept1 == 0, "box 1 should be suppressed by higher-score box 0");
    check(kept2 == 1, "box 2 should be kept as it is disjoint");
    check(kept3 == 0, "box 3 (zero area) should be skipped");
}

static void test_nms_tiebreak_determinism() {
    const int n = 100;
    std::vector<float> boxes(n * 4);
    std::vector<float> scores(n);
    for (int i = 0; i < n; ++i) {
        boxes[i * 4 + 0] = 0.1f;
        boxes[i * 4 + 1] = 0.1f;
        boxes[i * 4 + 2] = 0.2f;
        boxes[i * 4 + 3] = 0.2f;
        scores[i] = 0.5f;
    }
    std::vector<uint8_t> keep;
    std::vector<int> order;
    run_nms(n, boxes.data(), scores.data(), 0.4f, &keep, &order);

    // CUDA argsort is (score desc, original index asc); position 0 should be index 0.
    check(order[0] == 0, "deterministic tie-break should place lowest index first");

    int kept_count = 0;
    int kept_index = -1;
    for (int i = 0; i < n; ++i) {
        if (keep[i]) {
            ++kept_count;
            kept_index = order[i];
        }
    }
    check(kept_count == 1, "all-equal overlapping boxes should collapse to one");
    check(kept_index == 0, "lowest original index should win tie-break");
}

int main() {
    test_iou_basics();
    test_nms_basic();
    test_nms_tiebreak_determinism();

    if (g_failures == 0) {
        printf("All NMS tests PASSED\n");
        return 0;
    }
    fprintf(stderr, "NMS tests FAILED: %d failures\n", g_failures);
    return 1;
}
