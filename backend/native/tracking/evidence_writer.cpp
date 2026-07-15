#include "evidence_writer.h"

#include <array>
#include <chrono>
#include <iomanip>
#include <ios>
#include <limits>
#include <sstream>

namespace mv::tracking {

namespace {

constexpr std::size_t kEmbeddingDim = 512;

std::string to_string_or_null(float value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream oss;
    oss << std::setprecision(9) << value;
    return oss.str();
}

std::string rect_to_json(const RectF& r) {
    std::ostringstream oss;
    oss << std::setprecision(9);
    oss << "{\"x1\":" << r.x1 << ",\"y1\":" << r.y1
        << ",\"x2\":" << r.x2 << ",\"y2\":" << r.y2 << "}";
    return oss.str();
}

std::string role_to_string(DetectionRole role) {
    return role == DetectionRole::TrackingOnly ? "tracking_only" : "public";
}

std::string reason_to_string(TerminationReason reason) {
    switch (reason) {
    case TerminationReason::LostTimeout: return "lost_timeout";
    case TerminationReason::SourceEos: return "source_eos";
    case TerminationReason::StreamReset: return "stream_reset";
    case TerminationReason::TimestampGap: return "timestamp_gap";
    case TerminationReason::IdentityConflict: return "identity_conflict";
    case TerminationReason::PipelineShutdown: return "pipeline_shutdown";
    }
    return "unknown";
}

}  // namespace

bool EvidenceWriter::open(const std::string& output_dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (output_dir.empty()) return false;

    const std::string det_path = output_dir + "/detections.jsonl";
    const std::string trk_path = output_dir + "/tracklets.jsonl";
    const std::string emb_idx_path = output_dir + "/embedding_index.jsonl";
    const std::string manifest_path = output_dir + "/run_manifest.json";
    const std::string emb_bin_path = output_dir + "/embeddings.f32";

    detections_.open(det_path, std::ios::out | std::ios::trunc);
    tracklets_.open(trk_path, std::ios::out | std::ios::trunc);
    embedding_index_.open(emb_idx_path, std::ios::out | std::ios::trunc);
    manifest_.open(manifest_path, std::ios::out | std::ios::trunc);
    embeddings_f32_.open(emb_bin_path, std::ios::out | std::ios::binary | std::ios::trunc);

    if (!detections_ || !tracklets_ || !embedding_index_ || !manifest_ || !embeddings_f32_) {
        detections_.close();
        tracklets_.close();
        embedding_index_.close();
        manifest_.close();
        embeddings_f32_.close();
        return false;
    }

    open_ = true;
    next_embedding_ref_ = 0;
    embeddings_byte_offset_ = 0;
    return true;
}

void EvidenceWriter::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    detections_.close();
    tracklets_.close();
    embedding_index_.close();
    manifest_.close();
    embeddings_f32_.close();
    open_ = false;
}

void EvidenceWriter::write_detection(const DetectionRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!open_) return;
    detections_ << detection_to_json(record) << "\n";
}

std::uint64_t EvidenceWriter::write_embedding(const EmbeddingRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!open_ || !embeddings_f32_) return std::numeric_limits<std::uint64_t>::max();

    const std::uint64_t ref = next_embedding_ref_++;
    const std::uint64_t byte_offset = embeddings_byte_offset_;

    embeddings_f32_.write(reinterpret_cast<const char*>(record.data.data()),
                          record.data.size() * sizeof(float));
    embeddings_byte_offset_ += record.data.size() * sizeof(float);

    embedding_index_ << "{\"embedding_ref\":" << ref
                     << ",\"byte_offset\":" << byte_offset
                     << ",\"detection_id\":\"d" << record.detection_id << "\"}\n";
    return ref;
}

void EvidenceWriter::write_tracklet(const TrackletRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!open_) return;
    tracklets_ << tracklet_to_json(record) << "\n";
}

void EvidenceWriter::write_manifest(const RunManifest& manifest) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!open_ || !manifest_) return;
    manifest_.seekp(0, std::ios::beg);
    manifest_ << manifest_to_json(manifest) << "\n";
    manifest_.flush();
}

std::string EvidenceWriter::escape_json_string(const std::string& s) const {
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
        case '"': oss << "\\\""; break;
        case '\\': oss << "\\\\"; break;
        case '\b': oss << "\\b"; break;
        case '\f': oss << "\\f"; break;
        case '\n': oss << "\\n"; break;
        case '\r': oss << "\\r"; break;
        case '\t': oss << "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(static_cast<unsigned char>(c)) << std::dec;
            } else {
                oss << c;
            }
        }
    }
    return oss.str();
}

std::string EvidenceWriter::detection_to_json(const DetectionRecord& r) const {
    std::ostringstream oss;
    oss << std::setprecision(9);
    oss << "{\"detection_id\":\"d" << r.detection_id << "\"";
    oss << ",\"source_id\":" << r.source_id;
    oss << ",\"frame\":" << r.frame;
    oss << ",\"pts_ns\":" << r.pts_ns;
    oss << ",\"bbox\":" << rect_to_json(r.bbox);
    oss << ",\"detector_score\":" << r.detector_score;
    oss << ",\"role\":\"" << role_to_string(r.role) << "\"";
    if (r.embedding_ref.has_value()) {
        oss << ",\"embedding_ref\":" << *r.embedding_ref;
    } else {
        oss << ",\"embedding_ref\":null";
    }
    if (r.top1_similarity.has_value()) {
        oss << ",\"top1_similarity\":" << to_string_or_null(*r.top1_similarity);
    }
    if (r.top2_similarity.has_value()) {
        oss << ",\"top2_similarity\":" << to_string_or_null(*r.top2_similarity);
    }
    if (r.similarity_margin.has_value()) {
        oss << ",\"similarity_margin\":" << to_string_or_null(*r.similarity_margin);
    }
    oss << "}";
    return oss.str();
}

std::string EvidenceWriter::tracklet_to_json(const TrackletRecord& r) const {
    std::ostringstream oss;
    oss << "{\"schema_version\":\"2.0\"";
    oss << ",\"tracklet_id\":\"" << escape_json_string(r.tracklet_id) << "\"";
    oss << ",\"source_id\":" << r.source_id;
    oss << ",\"start_frame\":" << r.start_frame;
    oss << ",\"end_frame\":" << r.end_frame;
    oss << ",\"start_pts_ns\":" << r.start_pts_ns;
    oss << ",\"end_pts_ns\":" << r.end_pts_ns;
    oss << ",\"termination_reason\":\"" << escape_json_string(r.termination_reason) << "\"";
    oss << ",\"detection_count\":" << r.detection_count;
    oss << ",\"embedding_refs\":[";
    for (std::size_t i = 0; i < r.embedding_refs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << r.embedding_refs[i];
    }
    oss << "],\"best_shot_candidates\":[";
    for (std::size_t i = 0; i < r.best_shot_candidates.size(); ++i) {
        if (i > 0) oss << ",";
        const auto& c = r.best_shot_candidates[i];
        oss << "{\"frame\":" << c.frame
            << ",\"pts_ns\":" << c.pts_ns
            << ",\"detection_id\":\"" << c.detection_id << "\""
            << ",\"quality\":" << std::setprecision(9) << c.quality
            << ",\"embedding_ref\":" << c.embedding_ref << "}";
    }
    oss << "]}";
    return oss.str();
}

std::string EvidenceWriter::manifest_to_json(const RunManifest& m) const {
    const auto cfg_to_json = [](const TrackerConfig& cfg) -> std::string {
        std::ostringstream oss;
        oss << std::setprecision(9);
        oss << "{\"detector_emit_threshold\":" << cfg.detector_emit_threshold;
        oss << ",\"track_low_threshold\":" << cfg.track_low_threshold;
        oss << ",\"track_high_threshold\":" << cfg.track_high_threshold;
        oss << ",\"new_track_threshold\":" << cfg.new_track_threshold;
        oss << ",\"first_match_cost_threshold\":" << cfg.first_match_cost_threshold;
        oss << ",\"second_match_cost_threshold\":" << cfg.second_match_cost_threshold;
        oss << ",\"min_iou_gate\":" << cfg.min_iou_gate;
        oss << ",\"min_embedding_gate\":" << cfg.min_embedding_gate;
        oss << ",\"iou_weight\":" << cfg.iou_weight;
        oss << ",\"embedding_weight\":" << cfg.embedding_weight;
        oss << ",\"min_embedding_quality\":" << cfg.min_embedding_quality;
        oss << ",\"lost_timeout_ns\":" << cfg.lost_timeout_ns;
        oss << ",\"maximum_timestamp_gap_ns\":" << cfg.maximum_timestamp_gap_ns;
        oss << ",\"max_active_tracks\":" << cfg.max_active_tracks;
        oss << ",\"activation_hits\":" << cfg.activation_hits;
        oss << "}";
        return oss.str();
    };

    std::ostringstream oss;
    oss << "{\"schema_version\":\"" << escape_json_string(m.schema_version) << "\"";
    oss << ",\"job_id\":\"" << escape_json_string(m.job_id) << "\"";
    oss << ",\"run_id\":\"" << escape_json_string(m.run_id) << "\"";
    oss << ",\"media_path\":\"" << escape_json_string(m.media_path) << "\"";
    oss << ",\"detector_model_hash\":\"" << escape_json_string(m.detector_model_hash) << "\"";
    oss << ",\"recognizer_model_hash\":\"" << escape_json_string(m.recognizer_model_hash) << "\"";
    oss << ",\"gallery_hash\":\"" << escape_json_string(m.gallery_hash) << "\"";
    oss << ",\"preprocess_hash\":\"" << escape_json_string(m.preprocess_hash) << "\"";
    oss << ",\"tracker_config\":" << cfg_to_json(m.tracker_config);
    oss << ",\"started_at_ns\":" << m.started_at_ns;
    oss << ",\"finalized_at_ns\":" << m.finalized_at_ns;
    oss << "}";
    return oss.str();
}

} // namespace mv::tracking
