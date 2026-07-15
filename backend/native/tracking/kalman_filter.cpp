#include "kalman_filter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace mv::tracking {

namespace {

inline float* at(std::array<float, KalmanFilter::kStateDim * KalmanFilter::kStateDim>& m,
                 std::size_t row, std::size_t col) {
    return &m[row * KalmanFilter::kStateDim + col];
}
inline const float* at(const std::array<float, KalmanFilter::kStateDim * KalmanFilter::kStateDim>& m,
                       std::size_t row, std::size_t col) {
    return &m[row * KalmanFilter::kStateDim + col];
}

void matmul_888(const float* a, const float* b, float* out) {
    std::memset(out, 0, sizeof(float) * 8 * 8);
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t k = 0; k < 8; ++k) {
            float aik = a[i * 8 + k];
            for (std::size_t j = 0; j < 8; ++j) {
                out[i * 8 + j] += aik * b[k * 8 + j];
            }
        }
    }
}

// Generic row-major matrix multiplication C = A * B with A(n,k), B(k,m).
void matmul_generic(const float* a, const float* b, float* c,
                    std::size_t n, std::size_t k, std::size_t m) {
    std::memset(c, 0, sizeof(float) * n * m);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            float aik = a[i * k + p];
            for (std::size_t j = 0; j < m; ++j) {
                c[i * m + j] += aik * b[p * m + j];
            }
        }
    }
}

void transpose_8x8(const float* in, float* out) {
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 8; ++j) {
            out[j * 8 + i] = in[i * 8 + j];
        }
    }
}

void transpose_NxM(const float* in, float* out, std::size_t n, std::size_t m) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            out[j * n + i] = in[i * m + j];
        }
    }
}

// Invert a small square matrix in-place using Gauss-Jordan with partial pivoting.
// Returns false if singular.
bool invert_nxn(float* a, std::size_t n) {
    std::vector<std::size_t> pivot(n);
    for (std::size_t i = 0; i < n; ++i) pivot[i] = i;

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t best_row = col;
        float best_val = std::fabs(a[col * n + col]);
        for (std::size_t r = col + 1; r < n; ++r) {
            float v = std::fabs(a[r * n + col]);
            if (v > best_val) {
                best_val = v;
                best_row = r;
            }
        }
        if (best_val < 1e-12f) return false;
        if (best_row != col) {
            for (std::size_t k = 0; k < n; ++k) {
                std::swap(a[col * n + k], a[best_row * n + k]);
            }
            std::swap(pivot[col], pivot[best_row]);
        }

        float piv = a[col * n + col];
        for (std::size_t k = 0; k < n; ++k) {
            a[col * n + k] /= piv;
        }

        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) continue;
            float factor = a[r * n + col];
            if (std::fabs(factor) > 0.0f) {
                for (std::size_t k = 0; k < n; ++k) {
                    a[r * n + k] -= factor * a[col * n + k];
                }
            }
        }
    }

    // Undo row swaps on the inverse (which is now in a).
    for (std::size_t i = n; i-- > 0;) {
        if (pivot[i] != i) {
            for (std::size_t k = 0; k < n; ++k) {
                std::swap(a[k * n + i], a[k * n + pivot[i]]);
            }
        }
    }
    return true;
}

} // namespace

KalmanFilter::KalmanFilter() {
    std::memset(&state_, 0, sizeof(state_));
}

std::array<float, KalmanFilter::kStateDim * KalmanFilter::kStateDim>
KalmanFilter::eye(std::size_t n, float value) {
    std::array<float, kStateDim * kStateDim> m{};
    for (std::size_t i = 0; i < n; ++i) {
        m[i * n + i] = value;
    }
    return m;
}

void KalmanFilter::matmul_NNM(const float* a, const float* b, float* out,
                              std::size_t n, std::size_t m) {
    std::memset(out, 0, sizeof(float) * n * m);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            float aik = a[i * n + k];
            for (std::size_t j = 0; j < m; ++j) {
                out[i * m + j] += aik * b[k * m + j];
            }
        }
    }
}

void KalmanFilter::add_diag(float* mat, std::size_t n, float value) {
    for (std::size_t i = 0; i < n; ++i) mat[i * n + i] += value;
}

void KalmanFilter::init(const RectF& bbox) {
    state_.mean[0] = (bbox.x1 + bbox.x2) * 0.5f;
    state_.mean[1] = (bbox.y1 + bbox.y2) * 0.5f;
    state_.mean[2] = bbox.width() / std::max(bbox.height(), 1e-6f);
    state_.mean[3] = bbox.height();
    std::fill(state_.mean.begin() + 4, state_.mean.end(), 0.0f);

    std::fill(state_.covariance.begin(), state_.covariance.end(), 0.0f);
    float pos = std_weight_position_;
    float vel = std_weight_velocity_;
    float vars[8] = {
        2.0f * pos, 2.0f * pos,
        1e-2f, 2.0f * pos,
        10.0f * vel, 10.0f * vel,
        1e-5f, 10.0f * vel
    };
    for (std::size_t i = 0; i < kStateDim; ++i) {
        state_.covariance[i * kStateDim + i] = vars[i] * vars[i];
    }
}

KalmanFilter::State KalmanFilter::predict(float dt) const {
    State next;
    // Build motion matrix F.
    float F[64];
    std::memset(F, 0, sizeof(F));
    for (std::size_t i = 0; i < 8; ++i) F[i * 8 + i] = 1.0f;
    for (std::size_t i = 0; i < 4; ++i) F[i * 8 + (i + 4)] = dt;

    // x' = F x
    for (std::size_t i = 0; i < 8; ++i) {
        next.mean[i] = 0.0f;
        for (std::size_t j = 0; j < 8; ++j) {
            next.mean[i] += F[i * 8 + j] * state_.mean[j];
        }
    }

    // P' = F P F^T + Q
    float FP[64];
    matmul_888(F, state_.covariance.data(), FP);
    float FT[64];
    transpose_8x8(F, FT);
    matmul_888(FP, FT, next.covariance.data());

    float pos_var = std_weight_position_ * std_weight_position_;
    float vel_var = std_weight_velocity_ * std_weight_velocity_;
    float q_diag[8] = {
        dt * dt * vel_var, dt * dt * vel_var,
        dt * dt * vel_var, dt * dt * vel_var,
        vel_var, vel_var, vel_var, vel_var
    };
    for (std::size_t i = 0; i < 8; ++i) {
        next.covariance[i * 8 + i] += q_diag[i];
    }

    return next;
}

KalmanFilter::State KalmanFilter::update(const State& prior,
                                         const std::array<float, kMeasDim>& z) const {
    State next = prior;

    // Innovation y = z - H x
    std::array<float, kMeasDim> y = {z[0] - prior.mean[0], z[1] - prior.mean[1],
                                     z[2] - prior.mean[2], z[3] - prior.mean[3]};

    // S = H P H^T + R
    // H is 4x8, picks first 4 state components.
    std::array<float, kMeasDim * kMeasDim> S{};
    for (std::size_t i = 0; i < kMeasDim; ++i) {
        for (std::size_t j = 0; j < kMeasDim; ++j) {
            S[i * kMeasDim + j] = prior.covariance[i * kStateDim + j];
        }
    }
    float h_mean = std::sqrt(prior.mean[2] * prior.mean[3]); // proxy scale
    float r_scale = meas_noise_weight_ * h_mean;
    for (std::size_t i = 0; i < kMeasDim; ++i) {
        S[i * kMeasDim + i] += r_scale * r_scale;
    }

    // K = P H^T S^{-1}. P H^T is the first 4 columns of P (8x4).
    float PHt[8 * 4];
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            PHt[i * 4 + j] = prior.covariance[i * kStateDim + j];
        }
    }

    float Sinv[16];
    std::memcpy(Sinv, S.data(), sizeof(Sinv));
    if (!invert_nxn(Sinv, 4)) {
        // If S is singular, skip update.
        return prior;
    }

    float K[8 * 4];
    matmul_generic(PHt, Sinv, K, 8, 4, 4);

    // x = x + K y
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            next.mean[i] += K[i * 4 + j] * y[j];
        }
    }

    // P = (I - K H) P
    float KH[64];
    std::memset(KH, 0, sizeof(KH));
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            KH[i * 8 + j] = K[i * 4 + j];
        }
    }
    float I_KH[64];
    std::memset(I_KH, 0, sizeof(I_KH));
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 8; ++j) {
            I_KH[i * 8 + j] = (i == j ? 1.0f : 0.0f) - KH[i * 8 + j];
        }
    }
    float tmp[64];
    matmul_888(I_KH, prior.covariance.data(), tmp);
    std::memcpy(next.covariance.data(), tmp, sizeof(tmp));

    return next;
}

std::array<float, KalmanFilter::kMeasDim> bbox_to_measurement(const RectF& r) noexcept {
    float cx = (r.x1 + r.x2) * 0.5f;
    float cy = (r.y1 + r.y2) * 0.5f;
    float h = std::max(r.height(), 1e-6f);
    float a = r.width() / h;
    return {cx, cy, a, h};
}

RectF state_to_bbox(const std::array<float, KalmanFilter::kStateDim>& mean) noexcept {
    float cx = mean[0];
    float cy = mean[1];
    float a = mean[2];
    float h = mean[3];
    float w = a * h;
    float x1 = cx - w * 0.5f;
    float x2 = cx + w * 0.5f;
    float y1 = cy - h * 0.5f;
    float y2 = cy + h * 0.5f;
    return {x1, y1, x2, y2};
}

} // namespace mv::tracking
