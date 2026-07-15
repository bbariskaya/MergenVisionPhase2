#pragma once

#include "kalman_filter.h"
#include "linear_assignment.h"
#include "tracker_types.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace mv::tracking {

struct InternalTrack;

class ByteTracker {
public:
    explicit ByteTracker(SourceId source_id, TrackerConfig config);
    ~ByteTracker();

    ByteTracker(const ByteTracker&) = delete;
    ByteTracker& operator=(const ByteTracker&) = delete;
    ByteTracker(ByteTracker&&) = default;
    ByteTracker& operator=(ByteTracker&&) = default;

    UpdateResult update(const FrameKey& frame, std::span<const Detection> detections);
    std::vector<EndedTracklet> flush(TerminationReason reason);

    SourceId source_id() const { return source_id_; }

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;

    SourceId source_id_;
    TrackerConfig config_;

    LinearAssignmentInput build_cost_matrix_(
        const std::vector<InternalTrack*>& track_pool,
        const std::vector<const Detection*>& detections,
        bool use_appearance,
        float dt) const;

    void update_appearance_(InternalTrack& track, const Detection& det);
};

} // namespace mv::tracking
