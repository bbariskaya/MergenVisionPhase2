#include "linear_assignment.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mv::tracking {

namespace {

constexpr float kInf = 1e9f;
constexpr float kEps = 1e-7f;

inline float get(const std::vector<float>& m, std::size_t cols,
                 std::size_t row, std::size_t col) {
    return m[row * cols + col];
}

} // namespace

LinearAssignmentResult solve_linear_assignment(const LinearAssignmentInput& input) {
    LinearAssignmentResult result;
    if (input.rows == 0 || input.cols == 0) {
        return result;
    }

    const std::size_t n = std::max(input.rows, input.cols);

    // Build a square cost matrix padded with kInf for dummy rows/columns.
    std::vector<float> cost(n * n, kInf);
    for (std::size_t i = 0; i < input.rows; ++i) {
        for (std::size_t j = 0; j < input.cols; ++j) {
            float c = get(input.costs, input.cols, i, j);
            // Deterministic tie-break with negligible numerical effect (< kEps).
            if (c < kInf * 0.5f) {
                c += static_cast<float>(i * input.cols + j) * kEps / static_cast<float>(input.rows * input.cols + 1);
            }
            cost[i * n + j] = c;
        }
    }

    // Hungarian algorithm (Kuhn-Munkres) for minimum cost assignment, O(n^3).
    std::vector<float> u(n + 1, 0.0f);
    std::vector<float> v(n + 1, 0.0f);
    std::vector<std::size_t> p(n + 1, 0);
    std::vector<std::size_t> way(n + 1, 0);

    for (std::size_t i = 1; i <= n; ++i) {
        p[0] = i;
        std::size_t j0 = 0;
        std::vector<float> minv(n + 1, kInf);
        std::vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            std::size_t i0 = p[j0];
            std::size_t j1 = 0;
            float delta = kInf;
            for (std::size_t j = 1; j <= n; ++j) {
                if (used[j]) continue;
                float cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                // Deterministic tie-break: pick smallest column index on equal slack.
                if (minv[j] < delta || (std::fabs(minv[j] - delta) < 1e-12f && j < j1)) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (std::size_t j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            std::size_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    // p[j] = row assigned to column j.
    std::vector<char> row_matched(input.rows, false);
    std::vector<char> col_matched(input.cols, false);
    for (std::size_t j = 1; j <= n; ++j) {
        std::size_t row = p[j] - 1; // p is 1-based, 0 means dummy/unassigned.
        std::size_t col = j - 1;
        if (row < input.rows && col < input.cols) {
            float original_cost = get(input.costs, input.cols, row, col);
            if (original_cost < kInf * 0.5f) {
                result.matches.emplace_back(row, col);
                row_matched[row] = true;
                col_matched[col] = true;
            }
        }
    }

    result.matched_rows = 0;
    for (char m : row_matched) if (m) ++result.matched_rows;
    result.matched_cols = 0;
    for (char m : col_matched) if (m) ++result.matched_cols;
    return result;
}

} // namespace mv::tracking
