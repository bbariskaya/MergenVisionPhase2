#include "continuous_iou.h"

#include <algorithm>
#include <cmath>

namespace mv::tracking {

float intersection_area(const RectF& a, const RectF& b) noexcept {
    if (!a.valid() || !b.valid()) return 0.0f;

    float iw = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    float ih = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    return iw * ih;
}

float continuous_iou(const RectF& a, const RectF& b) noexcept {
    if (!a.valid() || !b.valid()) return 0.0f;

    float inter = intersection_area(a, b);
    float uni = a.area() + b.area() - inter;
    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}

} // namespace mv::tracking
