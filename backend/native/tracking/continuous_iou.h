#pragma once

#include "tracker_types.h"

namespace mv::tracking {

constexpr float kIoUInfCost = 1e9f;

float intersection_area(const RectF& a, const RectF& b) noexcept;
float continuous_iou(const RectF& a, const RectF& b) noexcept;

} // namespace mv::tracking
