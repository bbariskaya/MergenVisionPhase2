#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace mv::tracking {

struct LinearAssignmentInput {
    std::size_t rows = 0;
    std::size_t cols = 0;
    // cost(r, c) in row-major order. Empty input has zero rows/cols.
    std::vector<float> costs;
};

struct LinearAssignmentResult {
    // Each pair is (row_index, col_index). Only finite-cost matches are returned.
    std::vector<std::pair<std::size_t, std::size_t>> matches;
    // Number of rows that were matched.
    std::size_t matched_rows = 0;
    // Number of cols that were matched.
    std::size_t matched_cols = 0;
};

// Solve the rectangular assignment problem using the Hungarian/Munkres algorithm.
// The implementation is deterministic: equal-cost zeros are resolved by preferring
// the smaller column index, and rows/columns are processed in fixed order.
LinearAssignmentResult solve_linear_assignment(const LinearAssignmentInput& input);

} // namespace mv::tracking
