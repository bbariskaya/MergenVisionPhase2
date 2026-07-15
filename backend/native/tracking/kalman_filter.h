#pragma once

#include "tracker_types.h"

#include <array>
#include <cstddef>

namespace mv::tracking {

// Constant-velocity Kalman filter in the ByteTrack state space:
//   [cx, cy, aspect_ratio, height, vx, vy, va, vh]
class KalmanFilter {
public:
    static constexpr std::size_t kStateDim = 8;
    static constexpr std::size_t kMeasDim = 4;

    struct State {
        std::array<float, kStateDim> mean{};
        std::array<float, kStateDim * kStateDim> covariance{};
    };

    KalmanFilter();

    State predict(float dt) const;
    State update(const State& prior, const std::array<float, kMeasDim>& measurement) const;

    void init(const RectF& bbox);
    void set_state(const State& state) { state_ = state; }

    const State& state() const { return state_; }

private:
    static std::array<float, kStateDim * kStateDim> eye(std::size_t n, float value);
    static void matmul_NNM(const float* a, const float* b, float* out,
                           std::size_t n, std::size_t m);
    static void add_diag(float* mat, std::size_t n, float value);

    State state_;

    // Tuning defaults. May be made configurable later.
    float std_weight_position_ = 1.0f / 20.0f;
    float std_weight_velocity_ = 1.0f / 160.0f;
    float meas_noise_weight_ = 1.0f / 20.0f;
};

// Helpers to convert between bbox and measurement vector.
std::array<float, KalmanFilter::kMeasDim> bbox_to_measurement(const RectF& r) noexcept;
RectF state_to_bbox(const std::array<float, KalmanFilter::kStateDim>& mean) noexcept;

} // namespace mv::tracking
