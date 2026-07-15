#include "evidence_writer.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace mv::tracking;

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static std::string make_temp_dir() {
    std::string path = std::filesystem::temp_directory_path().string()
                       + "/mv_evidence_test_XXXXXX";
    std::vector<char> v(path.begin(), path.end());
    v.push_back('\0');
    if (!mkdtemp(v.data())) return "";
    return std::string(v.data());
}

static bool file_exists(const std::string& path) {
    return std::filesystem::exists(path);
}

static std::size_t count_lines(const std::string& path) {
    std::ifstream f(path);
    std::size_t n = 0;
    std::string line;
    while (std::getline(f, line)) ++n;
    return n;
}

static std::size_t file_size_bytes(const std::string& path) {
    return static_cast<std::size_t>(std::filesystem::file_size(path));
}

int main() {
    std::string dir = make_temp_dir();
    CHECK(!dir.empty());

    EvidenceWriter writer;
    CHECK(writer.open(dir));

    DetectionRecord d{};
    d.detection_id = 42;
    d.source_id = 0;
    d.frame = 7;
    d.pts_ns = 233'333'333ULL;
    d.bbox = {10.0f, 10.0f, 74.0f, 74.0f};
    d.detector_score = 0.85f;
    d.role = DetectionRole::Public;
    writer.write_detection(d);
    writer.write_detection(d);

    EmbeddingRecord e{};
    e.detection_id = 42;
    e.data.fill(0.01f);
    std::uint64_t ref = writer.write_embedding(e);
    CHECK(ref == 0);

    TrackletRecord tr{};
    tr.tracklet_id = "s0_tl_0007";
    tr.source_id = 0;
    tr.start_frame = 1;
    tr.end_frame = 10;
    tr.start_pts_ns = 33'333'333ULL;
    tr.end_pts_ns = 333'333'333ULL;
    tr.termination_reason = "lost_timeout";
    tr.detection_count = 10;
    tr.embedding_refs = {0};
    tr.best_shot_candidates = {BestShotCandidate{5, 166'666'666ULL, 42, 0.91f, 0}};
    writer.write_tracklet(tr);

    RunManifest m{};
    m.job_id = "job_abc";
    m.run_id = "run_xyz";
    m.media_path = "s3://bucket/video.mp4";
    m.detector_model_hash = "sha256:det";
    m.recognizer_model_hash = "sha256:rec";
    m.gallery_hash = "sha256:gal";
    m.preprocess_hash = "sha256:pre";
    m.tracker_config = TrackerConfig{};
    m.started_at_ns = 1'000'000'000ULL;
    m.finalized_at_ns = 2'000'000'000ULL;
    writer.write_manifest(m);

    writer.close();

    CHECK(file_exists(dir + "/detections.jsonl"));
    CHECK(file_exists(dir + "/tracklets.jsonl"));
    CHECK(file_exists(dir + "/embeddings.f32"));
    CHECK(file_exists(dir + "/embedding_index.jsonl"));
    CHECK(file_exists(dir + "/run_manifest.json"));

    CHECK(count_lines(dir + "/detections.jsonl") == 2);
    CHECK(count_lines(dir + "/tracklets.jsonl") == 1);
    CHECK(count_lines(dir + "/embedding_index.jsonl") == 1);

    // One 512-D float32 embedding => 2048 bytes.
    CHECK(file_size_bytes(dir + "/embeddings.f32") == sizeof(float) * 512);

    std::filesystem::remove_all(dir);

    if (g_failures) {
        fprintf(stderr, "evidence writer tests FAILED: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("evidence writer tests PASSED\n");
    return 0;
}
