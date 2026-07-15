#include "multi_source_tracker.h"

namespace mv::tracking {

MultiSourceTracker::MultiSourceTracker(TrackerConfig config)
    : config_(config) {}

UpdateResult MultiSourceTracker::update(const FrameKey& frame,
                                        std::span<const Detection> detections) {
    auto it = trackers_.find(frame.source_id);
    if (it == trackers_.end()) {
        it = trackers_.emplace(frame.source_id, ByteTracker(frame.source_id, config_)).first;
    }
    return it->second.update(frame, detections);
}

std::vector<EndedTracklet> MultiSourceTracker::flush_source(SourceId source_id,
                                                            TerminationReason reason) {
    auto it = trackers_.find(source_id);
    if (it == trackers_.end()) return {};
    return it->second.flush(reason);
}

std::vector<EndedTracklet> MultiSourceTracker::flush_all(TerminationReason reason) {
    std::vector<EndedTracklet> result;
    for (auto& kv : trackers_) {
        auto ended = kv.second.flush(reason);
        result.insert(result.end(), ended.begin(), ended.end());
    }
    return result;
}

} // namespace mv::tracking
