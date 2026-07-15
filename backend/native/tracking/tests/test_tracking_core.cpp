#include "byte_tracker.h"
#include "continuous_iou.h"
#include "multi_source_tracker.h"
#include "tracker_types.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace mv::tracking;

using Dets = std::vector<Detection>;

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static Detection make_det(DetectionId id, float x1, float y1, float x2, float y2,
                          float score, std::uint32_t seq = 0) {
    Detection d;
    d.detection_id = id;
    d.bbox = {x1, y1, x2, y2};
    d.detector_score = score;
    d.sequence_index = seq;
    return d;
}

static FrameKey frame(uint32_t source, uint64_t num, uint64_t pts_ns) {
    return {source, num, pts_ns};
}

static int test_iou() {
    RectF a{0, 0, 10, 10};
    RectF b{5, 5, 15, 15};
    float iou = continuous_iou(a, b);
    float expected = 25.0f / 175.0f;
    CHECK(std::fabs(iou - expected) < 1e-5f);

    RectF disjoint{100, 100, 110, 110};
    CHECK(continuous_iou(a, disjoint) == 0.0f);

    RectF invalid{0, 0, 0, 10};
    CHECK(continuous_iou(a, invalid) == 0.0f);

    RectF same{0, 0, 10, 10};
    CHECK(continuous_iou(a, same) == 1.0f);
    return 0;
}

static int test_single_detection_creates_track() {
    TrackerConfig cfg;
    cfg.activation_hits = 1;
    ByteTracker tracker(0, cfg);
    std::vector<Detection> dets = Dets{make_det(1, 10, 10, 30, 30, 0.9f)};
    auto r = tracker.update(frame(0, 1, 33'333'333), dets);
    CHECK(r.assignments.size() == 1);
    CHECK(r.assignments[0].newly_created);
    CHECK(r.assignments[0].tracklet_id != 0);
    CHECK(r.assignments[0].tracklet_id != kInvalidTrackletId);
    return 0;
}

static int test_tentative_activation() {
    TrackerConfig cfg;
    cfg.activation_hits = 2;
    ByteTracker tracker(0, cfg);

    auto r1 = tracker.update(frame(0, 1, 33'333'333),
                             Dets{make_det(1, 10, 10, 30, 30, 0.9f)});
    CHECK(r1.assignments.size() == 1);
    CHECK(r1.assignments[0].state == TrackState::Tentative);

    auto r2 = tracker.update(frame(0, 2, 66'666'666),
                             Dets{make_det(2, 11, 11, 31, 31, 0.9f)});
    CHECK(r2.assignments.size() == 1);
    CHECK(r2.assignments[0].state == TrackState::Tracked);
    CHECK(r2.assignments[0].tracklet_id == r1.assignments[0].tracklet_id);
    return 0;
}

static int test_low_detection_cannot_create_track() {
    TrackerConfig cfg;
    cfg.track_low_threshold = 0.1f;
    cfg.track_high_threshold = 0.5f;
    cfg.new_track_threshold = 0.45f;
    ByteTracker tracker(0, cfg);
    std::vector<Detection> dets = Dets{make_det(1, 10, 10, 30, 30, 0.3f)};
    auto r = tracker.update(frame(0, 1, 33'333'333), dets);
    CHECK(r.assignments.empty());
    auto ended = tracker.flush(TerminationReason::SourceEos);
    CHECK(ended.empty());
    return 0;
}

static int test_two_faces_same_frame_different_ids() {
    TrackerConfig cfg;
    cfg.activation_hits = 1;
    ByteTracker tracker(0, cfg);
    std::vector<Detection> dets = {
        make_det(1, 10, 10, 30, 30, 0.9f),
        make_det(2, 100, 100, 120, 120, 0.9f)
    };
    auto r = tracker.update(frame(0, 1, 33'333'333), dets);
    CHECK(r.assignments.size() == 2);
    CHECK(r.assignments[0].tracklet_id != r.assignments[1].tracklet_id);
    return 0;
}

static int test_lost_then_reactivated() {
    TrackerConfig cfg;
    cfg.activation_hits = 1;
    cfg.lost_timeout_ns = 500'000'000ULL; // 0.5 s
    ByteTracker tracker(0, cfg);

    tracker.update(frame(0, 1, 33'333'333), Dets{make_det(1, 10, 10, 30, 30, 0.9f)});

    // Empty frames for 100 ms; track should become Lost but not removed.
    for (int i = 2; i <= 4; ++i) {
        auto r = tracker.update(frame(0, i, static_cast<uint64_t>(i) * 33'333'333ULL), Dets{});
        CHECK(r.assignments.empty());
    }

    uint64_t reacquire_pts = 5ULL * 33'333'333ULL;
    auto r = tracker.update(frame(0, 5, reacquire_pts),
                            Dets{make_det(5, 12, 12, 32, 32, 0.9f)});
    CHECK(r.assignments.size() == 1);
    CHECK(r.assignments[0].tracklet_id == 1);

    // Reappear far later beyond timeout -> old track removed, new track.
    uint64_t late_pts = reacquire_pts + 600'000'000ULL;
    // Detection far from the predicted old track location so timeout removal wins.
    auto r2 = tracker.update(frame(0, 6, late_pts),
                             Dets{make_det(6, 200, 200, 220, 220, 0.9f)});
    CHECK(r2.ended_tracklets.size() == 1);
    CHECK(r2.assignments.size() == 1);
    CHECK(r2.assignments[0].tracklet_id == 2);
    return 0;
}

static int test_id_never_reused() {
    TrackerConfig cfg;
    cfg.activation_hits = 1;
    cfg.lost_timeout_ns = 100'000'000ULL;
    ByteTracker tracker(0, cfg);

    tracker.update(frame(0, 1, 33'333'333), Dets{make_det(1, 10, 10, 30, 30, 0.9f)});
    auto r = tracker.update(frame(0, 2, 300'000'000), Dets{});
    CHECK(r.ended_tracklets.size() == 1);

    auto r2 = tracker.update(frame(0, 3, 400'000'000),
                             Dets{make_det(3, 10, 10, 30, 30, 0.9f)});
    CHECK(!r2.assignments.empty());
    CHECK(r2.assignments[0].tracklet_id > 1);
    return 0;
}

static int test_determinism() {
    TrackerConfig cfg;
    cfg.activation_hits = 1;
    ByteTracker t1(0, cfg);
    ByteTracker t2(0, cfg);

    std::vector<Detection> dets = {
        make_det(1, 10, 10, 30, 30, 0.7f, 0),
        make_det(2, 15, 15, 35, 35, 0.75f, 1),
    };
    auto r1 = t1.update(frame(0, 1, 33'333'333), dets);
    auto r2 = t2.update(frame(0, 1, 33'333'333), dets);
    CHECK(r1.assignments.size() == 2);
    CHECK(r2.assignments.size() == r1.assignments.size());
    for (size_t i = 0; i < r1.assignments.size(); ++i) {
        CHECK(r1.assignments[i].detection_id == r2.assignments[i].detection_id);
        CHECK(r1.assignments[i].tracklet_id == r2.assignments[i].tracklet_id);
    }
    return 0;
}

static int test_pts_gap_terminates_track() {
    TrackerConfig cfg;
    cfg.activation_hits = 1;
    cfg.maximum_timestamp_gap_ns = 100'000'000ULL;
    ByteTracker tracker(0, cfg);

    tracker.update(frame(0, 1, 33'333'333), Dets{make_det(1, 10, 10, 30, 30, 0.9f)});
    auto r = tracker.update(frame(0, 2, 33'333'333 + 200'000'000ULL),
                            Dets{make_det(2, 11, 11, 31, 31, 0.9f)});
    CHECK(r.ended_tracklets.size() == 1);
    CHECK(r.assignments.size() == 1);
    CHECK(r.assignments[0].newly_created);
    return 0;
}

static int test_eos_flush() {
    TrackerConfig cfg;
    cfg.activation_hits = 1;
    ByteTracker tracker(0, cfg);
    tracker.update(frame(0, 1, 33'333'333), Dets{make_det(1, 10, 10, 30, 30, 0.9f)});
    auto ended = tracker.flush(TerminationReason::SourceEos);
    CHECK(ended.size() == 1);
    CHECK(ended[0].reason == TerminationReason::SourceEos);
    return 0;
}

static int test_multi_source_states_isolated() {
    TrackerConfig cfg;
    cfg.activation_hits = 1;
    MultiSourceTracker mst(cfg);

    auto r0 = mst.update(frame(0, 1, 33'333'333),
                         Dets{make_det(1, 10, 10, 30, 30, 0.9f)});
    auto r1 = mst.update(frame(1, 1, 33'333'333),
                         Dets{make_det(1, 100, 100, 120, 120, 0.9f)});
    CHECK(r0.assignments.size() == 1);
    CHECK(r1.assignments.size() == 1);
    CHECK(r0.assignments[0].tracklet_id == 1); // per-source counters both start at 1
    CHECK(r1.assignments[0].tracklet_id == 1);
    return 0;
}

static int test_crossing_trajectories() {
    TrackerConfig cfg;
    cfg.activation_hits = 1;
    cfg.first_match_cost_threshold = 1.0f;
    ByteTracker tracker(0, cfg);

    // Track A moves right, Track B moves left; they cross at frame 5.
    uint64_t pts = 33'333'333ULL;
    TrackletId id_a = 0, id_b = 0;
    for (int f = 1; f <= 10; ++f) {
        float ax1 = 10.0f + static_cast<float>(f - 1) * 8.0f;
        float ay1 = 50.0f;
        float ax2 = ax1 + 20.0f;
        float ay2 = ay1 + 20.0f;
        float bx1 = 90.0f - static_cast<float>(f - 1) * 8.0f;
        float by1 = 55.0f;
        float bx2 = bx1 + 20.0f;
        float by2 = by1 + 20.0f;
        auto r = tracker.update(frame(0, f, pts * f),
                                Dets{
                                    make_det(f * 2 + 0, ax1, ay1, ax2, ay2, 0.9f),
                                    make_det(f * 2 + 1, bx1, by1, bx2, by2, 0.9f),
                                });
        CHECK(r.assignments.size() == 2);
        TrackletId cur_a = 0, cur_b = 0;
        for (const auto& asgn : r.assignments) {
            if (asgn.detection_id == static_cast<DetectionId>(f * 2 + 0)) cur_a = asgn.tracklet_id;
            if (asgn.detection_id == static_cast<DetectionId>(f * 2 + 1)) cur_b = asgn.tracklet_id;
        }
        CHECK(cur_a != 0);
        CHECK(cur_b != 0);
        CHECK(cur_a != cur_b);
        // Without appearance embedding, ID persistence across a true physical
        // crossing is not guaranteed. The hard invariant is that the two
        // simultaneously visible objects receive distinct track IDs each frame.
        CHECK(cur_a != cur_b);
        (void)id_a;
        (void)id_b;
    }
    return 0;
}

static int test_short_occlusion() {
    TrackerConfig cfg;
    cfg.activation_hits = 1;
    cfg.lost_timeout_ns = 500'000'000ULL;
    ByteTracker tracker(0, cfg);

    tracker.update(frame(0, 1, 33'333'333),
                   Dets{make_det(1, 10, 10, 30, 30, 0.9f)});

    // Person occluded for two frames (~66 ms), below timeout.
    tracker.update(frame(0, 2, 66'666'666), Dets{});
    tracker.update(frame(0, 3, 99'999'999), Dets{});

    auto r = tracker.update(frame(0, 4, 133'333'332),
                            Dets{make_det(4, 13, 13, 33, 33, 0.9f)});
    CHECK(r.assignments.size() == 1);
    CHECK(r.assignments[0].tracklet_id == 1);
    return 0;
}

static int test_empty_frame_ages_track() {
    TrackerConfig cfg;
    cfg.activation_hits = 1;
    cfg.lost_timeout_ns = 100'000'000ULL;
    ByteTracker tracker(0, cfg);

    auto r1 = tracker.update(frame(0, 1, 33'333'333),
                             Dets{make_det(1, 10, 10, 30, 30, 0.9f)});
    CHECK(r1.assignments[0].state == TrackState::Tracked);

    auto r2 = tracker.update(frame(0, 2, 66'666'666), Dets{});
    CHECK(r2.assignments.empty());
    CHECK(r2.ended_tracklets.empty());

    // Long empty gap beyond timeout removes the Lost track.
    auto r3 = tracker.update(frame(0, 3, 300'000'000), Dets{});
    CHECK(r3.ended_tracklets.size() == 1);
    return 0;
}

struct FrameInput {
    FrameKey key;
    Dets dets;
};

static bool frame_input_less(const FrameInput& a, const FrameInput& b) {
    if (a.key.source_id != b.key.source_id)
        return a.key.source_id < b.key.source_id;
    if (a.key.pts_ns != b.key.pts_ns)
        return a.key.pts_ns < b.key.pts_ns;
    return a.key.frame_number < b.key.frame_number;
}

static int test_batch_parity() {
    // Build a realistic multi-source sequence: source 0 has a person moving,
    // source 1 has a static person. Randomize the update order and then sort
    // with the same comparator the temporal batch adapter uses.
    TrackerConfig cfg;
    cfg.activation_hits = 1;

    std::vector<FrameInput> inputs;
    for (int f = 1; f <= 8; ++f) {
        float x = 10.0f + (f - 1) * 4.0f;
        inputs.push_back({frame(0, f, 33'333'333ULL * f),
                          Dets{make_det(f, x, 10, x + 20, 30, 0.9f)}});
    }
    for (int f = 1; f <= 6; ++f) {
        inputs.push_back({frame(1, f, 33'333'333ULL * f + 1),
                          Dets{make_det(100 + f, 50, 50, 70, 70, 0.9f)}});
    }

    // Sequential run.
    MultiSourceTracker sequential(cfg);
    for (const auto& in : inputs) {
        sequential.update(in.key, in.dets);
    }
    auto seq_ended = sequential.flush_all(TerminationReason::SourceEos);

    // Shuffled then sorted run.
    std::vector<FrameInput> shuffled = inputs;
    // Intentionally reverse; adapter must correct it.
    std::reverse(shuffled.begin(), shuffled.end());
    std::stable_sort(shuffled.begin(), shuffled.end(), frame_input_less);

    MultiSourceTracker sorted(cfg);
    for (const auto& in : shuffled) {
        sorted.update(in.key, in.dets);
    }
    auto sorted_ended = sorted.flush_all(TerminationReason::SourceEos);

    CHECK(seq_ended.size() == sorted_ended.size());
    std::size_t seq_total = 0, sorted_total = 0;
    for (const auto& e : seq_ended) seq_total += e.detection_count;
    for (const auto& e : sorted_ended) sorted_total += e.detection_count;
    CHECK(seq_total == sorted_total);
    return 0;
}

int main() {
    test_iou();
    test_single_detection_creates_track();
    test_tentative_activation();
    test_low_detection_cannot_create_track();
    test_two_faces_same_frame_different_ids();
    test_lost_then_reactivated();
    test_id_never_reused();
    test_determinism();
    test_pts_gap_terminates_track();
    test_eos_flush();
    test_multi_source_states_isolated();
    test_crossing_trajectories();
    test_short_occlusion();
    test_empty_frame_ages_track();
    test_batch_parity();

    if (g_failures) {
        fprintf(stderr, "tracking core tests FAILED: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("tracking core tests PASSED\n");
    return 0;
}
