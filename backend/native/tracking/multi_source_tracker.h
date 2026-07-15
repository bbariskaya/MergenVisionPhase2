#pragma once

#include "byte_tracker.h"
#include "tracker_types.h"

#include <span>
#include <unordered_map>
#include <vector>

namespace mv::tracking {

class MultiSourceTracker {
public:
    explicit MultiSourceTracker(TrackerConfig config);

    UpdateResult update(const FrameKey& frame, std::span<const Detection> detections);

    std::vector<EndedTracklet> flush_source(SourceId source_id, TerminationReason reason);
    std::vector<EndedTracklet> flush_all(TerminationReason reason);

private:
    std::unordered_map<SourceId, ByteTracker> trackers_;
    TrackerConfig config_;
};

} // namespace mv::tracking
