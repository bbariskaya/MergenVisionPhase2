#pragma once

#include "tracker_types.h"

#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace mv::tracking {

// Normalized 512-D embedding storage. This is an internal biometric artifact
// and is not logged in plain text by the writer.
struct EmbeddingRecord {
    DetectionId detection_id = 0;
    std::array<float, 512> data{};
};

// Detection metadata sufficient for public/audit records and downstream Python
// reconciliation. The writer serializes one JSON object per line.
struct DetectionRecord {
    DetectionId detection_id = 0;
    SourceId source_id = 0;
    FrameNumber frame = 0;
    TimestampNs pts_ns = 0;
    RectF bbox{};
    float detector_score = 0.0f;
    DetectionRole role = DetectionRole::Public;

    // Optional: index into embeddings.f32 / embedding_index.jsonl.
    std::optional<std::uint64_t> embedding_ref;

    // Optional recognition-derived fields. When absent the writer omits them.
    std::optional<float> top1_similarity;
    std::optional<float> top2_similarity;
    std::optional<float> similarity_margin;
};

struct BestShotCandidate {
    FrameNumber frame = 0;
    TimestampNs pts_ns = 0;
    DetectionId detection_id = 0;
    float quality = 0.0f;
    std::uint64_t embedding_ref = 0;
};

struct TrackletRecord {
    std::string tracklet_id;
    SourceId source_id = 0;
    FrameNumber start_frame = 0;
    FrameNumber end_frame = 0;
    TimestampNs start_pts_ns = 0;
    TimestampNs end_pts_ns = 0;
    std::string termination_reason;
    std::size_t detection_count = 0;
    std::vector<std::uint64_t> embedding_refs;
    std::vector<BestShotCandidate> best_shot_candidates;
};

struct RunManifest {
    std::string schema_version = "2.0";
    std::string job_id;
    std::string run_id;
    std::string media_path;           // object-store key, not local FS path
    std::string detector_model_hash;
    std::string recognizer_model_hash;
    std::string gallery_hash;
    std::string preprocess_hash;
    TrackerConfig tracker_config{};
    std::uint64_t started_at_ns = 0;
    std::uint64_t finalized_at_ns = 0;
};

class EvidenceWriter {
public:
    // Opens files under output_dir. Returns false if any required file cannot be
    // opened or if output_dir is empty.
    bool open(const std::string& output_dir);

    // Flushes and closes all streams.
    void close();

    // Append one detection line. Thread-safe.
    void write_detection(const DetectionRecord& record);

    // Append one embedding vector and its index entry. Thread-safe.
    std::uint64_t write_embedding(const EmbeddingRecord& record);

    // Append one tracklet summary line. Thread-safe.
    void write_tracklet(const TrackletRecord& record);

    // Overwrite run_manifest.json. Thread-safe.
    void write_manifest(const RunManifest& manifest);

private:
    std::string escape_json_string(const std::string& s) const;
    std::string detection_to_json(const DetectionRecord& r) const;
    std::string tracklet_to_json(const TrackletRecord& r) const;
    std::string manifest_to_json(const RunManifest& m) const;

    std::mutex mutex_;
    std::ofstream detections_;
    std::ofstream tracklets_;
    std::ofstream embedding_index_;
    std::ofstream manifest_;
    std::ofstream embeddings_f32_;
    std::uint64_t next_embedding_ref_ = 0;
    std::uint64_t embeddings_byte_offset_ = 0;
    bool open_ = false;
};

} // namespace mv::tracking
