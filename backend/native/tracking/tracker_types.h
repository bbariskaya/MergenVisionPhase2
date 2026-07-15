#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace mv::tracking {

using SourceId = std::uint32_t;
using FrameNumber = std::uint64_t;
using TimestampNs = std::uint64_t;
using TrackletId = std::uint64_t;
using DetectionId = std::uint64_t;

constexpr TrackletId kUnassignedTrackletId = 0;
constexpr TrackletId kUntrackedObjectId = 0;
constexpr TrackletId kInvalidTrackletId = std::numeric_limits<TrackletId>::max();

struct RectF {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;

    bool valid() const noexcept {
        return std::isfinite(x1) && std::isfinite(y1) &&
               std::isfinite(x2) && std::isfinite(y2) &&
               x1 < x2 && y1 < y2;
    }

    float width() const noexcept { return x2 - x1; }
    float height() const noexcept { return y2 - y1; }
    float area() const noexcept { return width() * height(); }
};

struct FrameKey {
    SourceId source_id = 0;
    FrameNumber frame_number = 0;
    TimestampNs pts_ns = 0;
};

enum class DetectionRole {
    Public,
    TrackingOnly
};

struct Detection {
    DetectionId detection_id = 0;
    RectF bbox;
    float detector_score = 0.0f;

    DetectionRole role = DetectionRole::Public;

    bool embedding_valid = false;
    float embedding_quality = 0.0f;
    std::array<float, 512> embedding{};

    // Stale iteration order breaker for deterministic input sorting.
    std::uint32_t sequence_index = 0;
};

enum class TrackState {
    Tentative,
    Tracked,
    Lost,
    Removed
};

enum class TerminationReason {
    LostTimeout,
    SourceEos,
    StreamReset,
    TimestampGap,
    IdentityConflict,
    PipelineShutdown
};

struct Assignment {
    DetectionId detection_id = 0;
    TrackletId tracklet_id = 0;
    TrackState state = TrackState::Tracked;
    bool newly_created = false;
};

struct EndedTracklet {
    TrackletId tracklet_id = 0;
    SourceId source_id = 0;

    FrameNumber start_frame = 0;
    FrameNumber end_frame = 0;

    TimestampNs start_pts_ns = 0;
    TimestampNs end_pts_ns = 0;

    std::size_t detection_count = 0;
    TerminationReason reason = TerminationReason::LostTimeout;
};

struct UpdateResult {
    std::vector<Assignment> assignments;
    std::vector<EndedTracklet> ended_tracklets;
};

struct TrackerConfig {
    float detector_emit_threshold = 0.10f;
    float track_low_threshold = 0.10f;
    float track_high_threshold = 0.50f;
    float new_track_threshold = 0.45f;

    float first_match_cost_threshold = 1.0f;
    float second_match_cost_threshold = 1.0f;

    float min_iou_gate = 0.10f;
    float min_embedding_gate = 0.20f;

    float iou_weight = 1.0f;
    float embedding_weight = 0.0f;

    float min_embedding_quality = 0.0f;

    TimestampNs lost_timeout_ns = 2'000'000'000ULL;      // 2 s
    TimestampNs maximum_timestamp_gap_ns = 10'000'000'000ULL; // 10 s
    TimestampNs nominal_frame_period_ns = 33'333'333ULL; // 30 fps default

    std::size_t max_active_tracks = 10'000;

    // Number of matched high-score frames required to confirm a tentative track.
    std::size_t activation_hits = 1;

    std::uint64_t reserved_object_id_bits = 16;
};

class ByteTracker;
class MultiSourceTracker;

} // namespace mv::tracking
