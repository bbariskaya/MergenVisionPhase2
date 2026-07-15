#include "byte_tracker.h"

#include "continuous_iou.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mv::tracking {

namespace {

constexpr float kCostInf = kIoUInfCost;

inline float l2_norm(const std::array<float, 512>& v) {
    float s = 0.0f;
    for (float x : v) s += x * x;
    return std::sqrt(s);
}

inline void l2_normalize(std::array<float, 512>& v) {
    float n = l2_norm(v);
    if (n < 1e-12f) return;
    for (float& x : v) x /= n;
}

inline float l2_normalized_dot(const std::array<float, 512>& a,
                               const std::array<float, 512>& b) {
    float s = 0.0f;
    for (std::size_t i = 0; i < 512; ++i) s += a[i] * b[i];
    return std::max(-1.0f, std::min(1.0f, s));
}

} // namespace

struct InternalTrack {
    TrackletId id = 0;
    TrackState state = TrackState::Tentative;
    KalmanFilter kalman;

    FrameNumber start_frame = 0;
    TimestampNs start_pts_ns = 0;
    FrameNumber last_matched_frame = 0;
    TimestampNs last_matched_pts_ns = 0;

    std::size_t detection_count = 0;
    std::size_t hits = 0;

    std::array<float, 512> appearance_mean{};
    float appearance_weight = 0.0f;
    bool appearance_valid = false;

    bool ended_reported = false;
};

struct ByteTracker::Impl {
    bool initialized = false;
    FrameNumber last_frame_number = 0;
    TimestampNs last_pts_ns = 0;
    TrackletId next_tracklet_id = 1;
    std::vector<std::unique_ptr<InternalTrack>> tracks;
};

ByteTracker::ByteTracker(SourceId source_id, TrackerConfig config)
    : source_id_(source_id), config_(config), impl_(std::make_shared<Impl>()) {}

ByteTracker::~ByteTracker() = default;

namespace {

void validate_monotonic(bool initialized, FrameNumber last_frame, TimestampNs last_pts,
                        const FrameKey& frame) {
    if (!initialized) return;
    if (frame.frame_number <= last_frame) {
        throw std::invalid_argument("frame_number is not monotonic");
    }
    if (frame.pts_ns <= last_pts) {
        throw std::invalid_argument("pts_ns is not monotonic");
    }
}

float compute_dt(bool initialized, TimestampNs last_pts,
                 TimestampNs current_pts, TimestampNs nominal_period) {
    if (!initialized || nominal_period == 0) return 1.0f;
    float delta = static_cast<float>(current_pts - last_pts);
    return delta / static_cast<float>(nominal_period);
}

} // namespace

LinearAssignmentInput ByteTracker::build_cost_matrix_(
    const std::vector<InternalTrack*>& track_pool,
    const std::vector<const Detection*>& detections,
    bool use_appearance,
    float dt) const {

    LinearAssignmentInput input;
    input.rows = track_pool.size();
    input.cols = detections.size();
    input.costs.resize(input.rows * input.cols, kCostInf);

    for (std::size_t i = 0; i < track_pool.size(); ++i) {
        const auto* track = track_pool[i];
        RectF pred_box = state_to_bbox(track->kalman.predict(dt).mean);
        for (std::size_t j = 0; j < detections.size(); ++j) {
            const auto* det = detections[j];
            float iou = continuous_iou(pred_box, det->bbox);
            if (iou <= 0.0f) continue;

            bool track_emb = track->appearance_valid;
            bool detection_emb = det->embedding_valid &&
                                 det->embedding_quality >= config_.min_embedding_quality;
            if (iou < config_.min_iou_gate && !(track_emb && detection_emb)) {
                continue;
            }

            float cost = config_.iou_weight * (1.0f - iou);
            if (use_appearance && track_emb && detection_emb) {
                float cos = l2_normalized_dot(track->appearance_mean, det->embedding);
                if (cos < config_.min_embedding_gate) continue;
                cost += config_.embedding_weight * (1.0f - cos);
            }

            float threshold = use_appearance ? config_.first_match_cost_threshold
                                             : config_.second_match_cost_threshold;
            if (cost > threshold) continue;
            input.costs[i * input.cols + j] = cost;
        }
    }
    return input;
}

void ByteTracker::update_appearance_(InternalTrack& track, const Detection& det) {
    if (!det.embedding_valid || det.embedding_quality < config_.min_embedding_quality) {
        return;
    }
    std::array<float, 512> norm = det.embedding;
    l2_normalize(norm);
    float q = det.embedding_quality;
    if (track.appearance_weight == 0.0f) {
        track.appearance_mean = norm;
        track.appearance_weight = q;
    } else {
        float w = track.appearance_weight;
        for (std::size_t i = 0; i < 512; ++i) {
            track.appearance_mean[i] = (track.appearance_mean[i] * w + norm[i] * q) / (w + q);
        }
        track.appearance_weight += q;
    }
    l2_normalize(track.appearance_mean);
    track.appearance_valid = true;
}

UpdateResult ByteTracker::update(const FrameKey& frame,
                                 std::span<const Detection> detections) {
    UpdateResult result;

    if (impl_->initialized &&
        (frame.pts_ns - impl_->last_pts_ns) > config_.maximum_timestamp_gap_ns) {
        auto gap_ended = flush(TerminationReason::TimestampGap);
        result.ended_tracklets.insert(result.ended_tracklets.end(),
                                      gap_ended.begin(), gap_ended.end());
    }

    validate_monotonic(impl_->initialized, impl_->last_frame_number,
                       impl_->last_pts_ns, frame);

    const bool first_frame_after_gap = !impl_->initialized;
    float dt = compute_dt(impl_->initialized, impl_->last_pts_ns,
                          frame.pts_ns, config_.nominal_frame_period_ns);
    if (dt <= 0.0f) dt = 1.0f;

    if (!impl_->initialized) {
        impl_->initialized = true;
    }

    // Update timing bookkeeping after using the old values for prediction.
    impl_->last_frame_number = frame.frame_number;
    impl_->last_pts_ns = frame.pts_ns;

    std::vector<const Detection*> sorted_dets;
    sorted_dets.reserve(detections.size());
    for (const auto& d : detections) sorted_dets.push_back(&d);
    std::stable_sort(sorted_dets.begin(), sorted_dets.end(),
        [](const Detection* a, const Detection* b) {
            if (a->bbox.x1 != b->bbox.x1) return a->bbox.x1 < b->bbox.x1;
            if (a->bbox.y1 != b->bbox.y1) return a->bbox.y1 < b->bbox.y1;
            if (a->bbox.x2 != b->bbox.x2) return a->bbox.x2 < b->bbox.x2;
            if (a->bbox.y2 != b->bbox.y2) return a->bbox.y2 < b->bbox.y2;
            if (a->detector_score != b->detector_score) return a->detector_score > b->detector_score;
            return a->sequence_index < b->sequence_index;
        });

    std::vector<const Detection*> high_dets;
    std::vector<const Detection*> low_dets;
    for (auto* d : sorted_dets) {
        if (d->detector_score >= config_.track_high_threshold) {
            high_dets.push_back(d);
        } else if (d->detector_score >= config_.track_low_threshold) {
            low_dets.push_back(d);
        }
    }

    struct PredictedTrack {
        InternalTrack* track;
        KalmanFilter::State pred_state;
        RectF pred_box;
    };
    std::vector<PredictedTrack> predicted;
    predicted.reserve(impl_->tracks.size());
    for (auto& t : impl_->tracks) {
        if (t->state == TrackState::Removed || t->ended_reported) continue;
        KalmanFilter::State pred = t->kalman.predict(dt);
        predicted.push_back({t.get(), pred, state_to_bbox(pred.mean)});
    }

    std::vector<InternalTrack*> first_pool;
    for (auto& pt : predicted) {
        if (pt.track->state == TrackState::Tracked || pt.track->state == TrackState::Lost) {
            first_pool.push_back(pt.track);
        }
    }
    std::sort(first_pool.begin(), first_pool.end(),
              [](InternalTrack* a, InternalTrack* b) { return a->id < b->id; });

    auto first_input = build_cost_matrix_(first_pool, high_dets, true, dt);
    auto first_result = solve_linear_assignment(first_input);

    std::vector<char> high_matched(high_dets.size(), false);
    std::vector<char> first_track_matched(first_pool.size(), false);
    for (const auto& m : first_result.matches) {
        InternalTrack* track = first_pool[m.first];
        const Detection* det = high_dets[m.second];
        auto it = std::find_if(predicted.begin(), predicted.end(),
                               [track](const PredictedTrack& p) { return p.track == track; });
        track->kalman.set_state(track->kalman.update(it->pred_state, bbox_to_measurement(det->bbox)));
        track->state = TrackState::Tracked;
        track->hits++;
        track->detection_count++;
        track->last_matched_frame = frame.frame_number;
        track->last_matched_pts_ns = frame.pts_ns;
        update_appearance_(*track, *det);
        high_matched[m.second] = true;
        first_track_matched[m.first] = true;
        result.assignments.push_back({det->detection_id, track->id, track->state, false});
    }

    std::vector<InternalTrack*> second_pool;
    for (std::size_t i = 0; i < first_pool.size(); ++i) {
        if (!first_track_matched[i] && first_pool[i]->state == TrackState::Tracked) {
            second_pool.push_back(first_pool[i]);
        }
    }
    std::sort(second_pool.begin(), second_pool.end(),
              [](InternalTrack* a, InternalTrack* b) { return a->id < b->id; });

    auto second_input = build_cost_matrix_(second_pool, low_dets, false, dt);
    auto second_result = solve_linear_assignment(second_input);

    std::vector<char> low_matched(low_dets.size(), false);
    std::vector<char> second_track_matched(second_pool.size(), false);
    for (const auto& m : second_result.matches) {
        InternalTrack* track = second_pool[m.first];
        const Detection* det = low_dets[m.second];
        auto it = std::find_if(predicted.begin(), predicted.end(),
                               [track](const PredictedTrack& p) { return p.track == track; });
        track->kalman.set_state(track->kalman.update(it->pred_state, bbox_to_measurement(det->bbox)));
        track->state = TrackState::Tracked;
        track->hits++;
        track->detection_count++;
        track->last_matched_frame = frame.frame_number;
        track->last_matched_pts_ns = frame.pts_ns;
        update_appearance_(*track, *det);
        low_matched[m.second] = true;
        second_track_matched[m.first] = true;
        result.assignments.push_back({det->detection_id, track->id, track->state, false});
    }

    std::vector<InternalTrack*> tentative_pool;
    for (auto& pt : predicted) {
        if (pt.track->state == TrackState::Tentative) {
            tentative_pool.push_back(pt.track);
        }
    }
    std::sort(tentative_pool.begin(), tentative_pool.end(),
              [](InternalTrack* a, InternalTrack* b) { return a->id < b->id; });

    std::vector<const Detection*> remaining_high;
    std::vector<std::size_t> remaining_high_idx;
    for (std::size_t i = 0; i < high_dets.size(); ++i) {
        if (!high_matched[i]) {
            remaining_high.push_back(high_dets[i]);
            remaining_high_idx.push_back(i);
        }
    }

    auto tent_input = build_cost_matrix_(tentative_pool, remaining_high, true, dt);
    auto tent_result = solve_linear_assignment(tent_input);
    for (const auto& m : tent_result.matches) {
        InternalTrack* track = tentative_pool[m.first];
        const Detection* det = remaining_high[m.second];
        auto it = std::find_if(predicted.begin(), predicted.end(),
                               [track](const PredictedTrack& p) { return p.track == track; });
        track->kalman.set_state(track->kalman.update(it->pred_state, bbox_to_measurement(det->bbox)));
        track->hits++;
        track->detection_count++;
        track->last_matched_frame = frame.frame_number;
        track->last_matched_pts_ns = frame.pts_ns;
        update_appearance_(*track, *det);
        bool newly_confirmed = track->hits >= config_.activation_hits;
        if (newly_confirmed) track->state = TrackState::Tracked;
        result.assignments.push_back({det->detection_id, track->id, track->state, false});
        std::size_t orig_idx = remaining_high_idx[m.second];
        high_matched[orig_idx] = true;
    }

    for (std::size_t i = 0; i < high_dets.size(); ++i) {
        if (high_matched[i]) continue;
        const Detection* det = high_dets[i];
        if (det->detector_score < config_.new_track_threshold) continue;
        if (impl_->next_tracklet_id > config_.max_active_tracks) continue;

        auto track = std::make_unique<InternalTrack>();
        track->id = impl_->next_tracklet_id++;
        track->state = TrackState::Tentative;
        track->kalman.init(det->bbox);
        track->start_frame = frame.frame_number;
        track->start_pts_ns = frame.pts_ns;
        track->last_matched_frame = frame.frame_number;
        track->last_matched_pts_ns = frame.pts_ns;
        track->hits = 1;
        track->detection_count = 1;
        update_appearance_(*track, *det);
        bool newly_confirmed = track->hits >= config_.activation_hits;
        if (newly_confirmed) track->state = TrackState::Tracked;

        TrackletId tid = track->id;
        TrackState state = track->state;
        impl_->tracks.push_back(std::move(track));
        result.assignments.push_back({det->detection_id, tid, state, true});
    }

    for (auto& pt : predicted) {
        auto* track = pt.track;
        if (track->state == TrackState::Removed || track->ended_reported) continue;

        bool was_matched = false;
        for (const auto& a : result.assignments) {
            if (a.tracklet_id == track->id) { was_matched = true; break; }
        }
        if (was_matched) continue;

        track->kalman.set_state(pt.pred_state);

        if (track->state == TrackState::Tentative) {
            track->state = TrackState::Removed;
        } else if (track->state == TrackState::Tracked) {
            track->state = TrackState::Lost;
        }

        TimestampNs age = frame.pts_ns - track->last_matched_pts_ns;
        if (track->state == TrackState::Lost && age > config_.lost_timeout_ns) {
            track->state = TrackState::Removed;
        }
    }

    for (auto& t : impl_->tracks) {
        if (t->state == TrackState::Removed && !t->ended_reported) {
            t->ended_reported = true;
            EndedTracklet e;
            e.tracklet_id = t->id;
            e.source_id = source_id_;
            e.start_frame = t->start_frame;
            e.end_frame = t->last_matched_frame;
            e.start_pts_ns = t->start_pts_ns;
            e.end_pts_ns = t->last_matched_pts_ns;
            e.detection_count = t->detection_count;
            e.reason = first_frame_after_gap ? TerminationReason::TimestampGap
                                             : TerminationReason::LostTimeout;
            result.ended_tracklets.push_back(e);
        }
    }

    return result;
}

std::vector<EndedTracklet> ByteTracker::flush(TerminationReason reason) {
    std::vector<EndedTracklet> result;
    for (auto& t : impl_->tracks) {
        if (t->state == TrackState::Removed && t->ended_reported) continue;
        if (t->state != TrackState::Removed) {
            t->state = TrackState::Removed;
        }
        t->ended_reported = true;
        EndedTracklet e;
        e.tracklet_id = t->id;
        e.source_id = source_id_;
        e.start_frame = t->start_frame;
        e.end_frame = t->last_matched_frame;
        e.start_pts_ns = t->start_pts_ns;
        e.end_pts_ns = t->last_matched_pts_ns;
        e.detection_count = t->detection_count;
        e.reason = reason;
        result.push_back(e);
    }
    return result;
}

} // namespace mv::tracking
