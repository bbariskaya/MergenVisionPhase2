/* GStreamer element: mvfacetracker
 *
 * DeepStream-native face tracklet producer. Receives batched RGBA/NVMM
 * buffers with NvDsBatchMeta/NvDsObjectMeta, runs the MergenVision
 * ByteTrack-like core tracker, and attaches versioned MvTrackletMeta.
 */
#include "gstmvfacetracker.h"

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#include "gstnvdsmeta.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "tracklet_meta.h"
#include "tracker_types.h"
#include "multi_source_tracker.h"
#include "evidence_writer.h"

#ifndef PACKAGE
#define PACKAGE "mvfacetracker"
#endif

#define GST_TYPE_MV_FACE_TRACKER (gst_mv_face_tracker_get_type())
#define GST_MV_FACE_TRACKER(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_MV_FACE_TRACKER, GstMvFaceTracker))

struct GstMvFaceTracker {
    GstBaseTransform base_transform;

    std::uint64_t nominal_frame_period_ns = 33333333ULL;  // ~30 fps default
    gint gpu_id = 0;
    gchar* evidence_dir = nullptr;

    mv::tracking::TrackerConfig tracker_config;
    mv::tracking::MultiSourceTracker* tracker = nullptr;
    mv::tracking::EvidenceWriter* evidence_writer = nullptr;

    struct SourceLedger {
        bool initialized = false;
        std::uint64_t last_frame_number = 0;
        std::uint64_t last_pts_ns = 0;
        std::uint64_t processed_frame_count = 0;
    };
    std::unordered_map<mv::tracking::SourceId, SourceLedger> ledgers;
};

struct GstMvFaceTrackerClass {
    GstBaseTransformClass base_transform_class;
};

G_DEFINE_TYPE(GstMvFaceTracker, gst_mv_face_tracker, GST_TYPE_BASE_TRANSFORM)

GST_ELEMENT_REGISTER_DEFINE(mvfacetracker, "mvfacetracker",
    GST_RANK_PRIMARY, GST_TYPE_MV_FACE_TRACKER)

enum {
    PROP_0,
    PROP_NOMINAL_FRAME_PERIOD_NS,
    PROP_GPU_ID,
    PROP_EVIDENCE_DIR,
};

namespace {

inline mv::tracking::RectF bbox_to_rectf(const NvBbox_Coords& coords) {
    return {coords.left,
            coords.top,
            coords.left + coords.width,
            coords.top + coords.height};
}

inline mv::tracking::TrackletId encode_tracklet_id(
    mv::tracking::SourceId source_id, mv::tracking::TrackletId local_id) {
    return (static_cast<std::uint64_t>(source_id) << 32u) | local_id;
}

inline std::string format_tracklet_id(
    mv::tracking::SourceId source_id, mv::tracking::TrackletId local_id) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "s%u_tl_%04llu",
                  source_id, static_cast<unsigned long long>(local_id));
    return std::string(buf);
}

struct AdapterDetection {
    NvDsObjectMeta* obj_meta = nullptr;
    std::uint32_t sequence_index = 0;
    mv::tracking::DetectionId detection_id = 0;
    mv::tracking::RectF bbox{};
    float detector_score = 0.0f;
    bool tracking_only = false;
};

bool adapter_detection_less(const AdapterDetection& a, const AdapterDetection& b) {
    if (a.tracking_only != b.tracking_only) return a.tracking_only < b.tracking_only;
    if (a.bbox.x1 != b.bbox.x1) return a.bbox.x1 < b.bbox.x1;
    if (a.bbox.y1 != b.bbox.y1) return a.bbox.y1 < b.bbox.y1;
    if (a.bbox.x2 != b.bbox.x2) return a.bbox.x2 < b.bbox.x2;
    if (a.bbox.y2 != b.bbox.y2) return a.bbox.y2 < b.bbox.y2;
    if (a.detector_score != b.detector_score) return a.detector_score > b.detector_score;
    return a.sequence_index < b.sequence_index;
}

mv::tracking::TrackerConfig default_tracker_config() {
    mv::tracking::TrackerConfig cfg{};
    cfg.detector_emit_threshold = 0.10f;
    cfg.track_low_threshold = 0.10f;
    cfg.track_high_threshold = 0.30f;
    cfg.new_track_threshold = 0.30f;
    cfg.first_match_cost_threshold = 0.80f;
    cfg.second_match_cost_threshold = 0.90f;
    cfg.min_iou_gate = 0.10f;
    cfg.min_embedding_gate = 0.20f;
    cfg.iou_weight = 1.0f;
    cfg.embedding_weight = 0.0f;
    cfg.min_embedding_quality = 0.30f;
    cfg.lost_timeout_ns = 3000000000ULL;      // 3 s
    cfg.maximum_timestamp_gap_ns = 10000000000ULL; // 10 s
    cfg.max_active_tracks = 1024;
    return cfg;
}

}  // namespace

static void gst_mv_face_tracker_set_property(GObject* object, guint prop_id,
                                             const GValue* value,
                                             GParamSpec* pspec) {
    auto* self = GST_MV_FACE_TRACKER(object);
    switch (prop_id) {
    case PROP_NOMINAL_FRAME_PERIOD_NS:
        self->nominal_frame_period_ns =
            static_cast<std::uint64_t>(g_value_get_uint64(value));
        break;
    case PROP_GPU_ID:
        self->gpu_id = g_value_get_int(value);
        break;
    case PROP_EVIDENCE_DIR:
        g_free(self->evidence_dir);
        self->evidence_dir = g_value_dup_string(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_mv_face_tracker_get_property(GObject* object, guint prop_id,
                                             GValue* value, GParamSpec* pspec) {
    auto* self = GST_MV_FACE_TRACKER(object);
    switch (prop_id) {
    case PROP_NOMINAL_FRAME_PERIOD_NS:
        g_value_set_uint64(value, self->nominal_frame_period_ns);
        break;
    case PROP_GPU_ID:
        g_value_set_int(value, self->gpu_id);
        break;
    case PROP_EVIDENCE_DIR:
        g_value_set_string(value, self->evidence_dir ? self->evidence_dir : "");
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_mv_face_tracker_finalize(GObject* object) {
    auto* self = GST_MV_FACE_TRACKER(object);
    delete self->tracker;
    self->tracker = nullptr;
    delete self->evidence_writer;
    self->evidence_writer = nullptr;
    g_free(self->evidence_dir);
    self->evidence_dir = nullptr;
    G_OBJECT_CLASS(gst_mv_face_tracker_parent_class)->finalize(object);
}

static GstCaps* gst_mv_face_tracker_transform_caps(GstBaseTransform* btrans,
                                                   GstPadDirection direction,
                                                   GstCaps* caps,
                                                   GstCaps* filter) {
    (void)btrans; (void)direction; (void)filter;
    return gst_caps_ref(caps);
}

static gboolean gst_mv_face_tracker_start(GstBaseTransform* btrans) {
    auto* self = GST_MV_FACE_TRACKER(btrans);
    self->tracker_config.nominal_frame_period_ns = self->nominal_frame_period_ns;
    self->tracker = new mv::tracking::MultiSourceTracker(self->tracker_config);
    self->ledgers.clear();

    if (self->evidence_dir && self->evidence_dir[0]) {
        delete self->evidence_writer;
        self->evidence_writer = new mv::tracking::EvidenceWriter();
        if (!self->evidence_writer->open(self->evidence_dir)) {
            GST_ELEMENT_ERROR(self, RESOURCE, OPEN_READ,
                ("Cannot open evidence directory: %s", self->evidence_dir),
                (NULL));
            delete self->evidence_writer;
            self->evidence_writer = nullptr;
            return FALSE;
        }
    }
    return TRUE;
}

static void process_ended_tracklets(
    GstElement* element,
    mv::tracking::EvidenceWriter* writer,
    const std::vector<mv::tracking::EndedTracklet>& ended) {
    for (const auto& tl : ended) {
        GstStructure* s = gst_structure_new(
            "mv-tracklet-ended",
            "source_id", G_TYPE_UINT, static_cast<guint>(tl.source_id),
            "tracklet_id", G_TYPE_UINT64,
                encode_tracklet_id(tl.source_id, tl.tracklet_id),
            "start_frame", G_TYPE_UINT64, tl.start_frame,
            "end_frame", G_TYPE_UINT64, tl.end_frame,
            "start_pts_ns", G_TYPE_UINT64, tl.start_pts_ns,
            "end_pts_ns", G_TYPE_UINT64, tl.end_pts_ns,
            "detection_count", G_TYPE_UINT64,
                static_cast<guint64>(tl.detection_count),
            "termination_reason", G_TYPE_INT, static_cast<int>(tl.reason),
            nullptr);
        gst_element_post_message(element,
            gst_message_new_application(GST_OBJECT(element), s));

        if (writer) {
            mv::tracking::TrackletRecord rec{};
            rec.tracklet_id = format_tracklet_id(tl.source_id, tl.tracklet_id);
            rec.source_id = tl.source_id;
            rec.start_frame = tl.start_frame;
            rec.end_frame = tl.end_frame;
            rec.start_pts_ns = tl.start_pts_ns;
            rec.end_pts_ns = tl.end_pts_ns;
            switch (tl.reason) {
            case mv::tracking::TerminationReason::LostTimeout:
                rec.termination_reason = "lost_timeout"; break;
            case mv::tracking::TerminationReason::SourceEos:
                rec.termination_reason = "source_eos"; break;
            case mv::tracking::TerminationReason::StreamReset:
                rec.termination_reason = "stream_reset"; break;
            case mv::tracking::TerminationReason::TimestampGap:
                rec.termination_reason = "timestamp_gap"; break;
            case mv::tracking::TerminationReason::IdentityConflict:
                rec.termination_reason = "identity_conflict"; break;
            case mv::tracking::TerminationReason::PipelineShutdown:
                rec.termination_reason = "pipeline_shutdown"; break;
            }
            rec.detection_count = tl.detection_count;
            writer->write_tracklet(rec);
        }
    }
}

static GstFlowReturn gst_mv_face_tracker_transform_ip(GstBaseTransform* btrans,
                                                      GstBuffer* buf) {
    auto* self = GST_MV_FACE_TRACKER(btrans);
    if (!self->tracker) {
        GST_ELEMENT_ERROR(self, CORE, FAILED,
            ("Tracker not initialized"), (NULL));
        return GST_FLOW_ERROR;
    }

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        GST_ELEMENT_ERROR(self, CORE, FAILED,
            ("No NvDsBatchMeta on buffer"), (NULL));
        return GST_FLOW_ERROR;
    }

    std::vector<NvDsFrameMeta*> frames;
    for (GList* it = batch_meta->frame_meta_list; it != nullptr; it = g_list_next(it)) {
        frames.push_back(static_cast<NvDsFrameMeta*>(it->data));
    }

    std::stable_sort(frames.begin(), frames.end(),
        [](const NvDsFrameMeta* a, const NvDsFrameMeta* b) {
            if (a->source_id != b->source_id)
                return a->source_id < b->source_id;
            if (a->buf_pts != b->buf_pts)
                return a->buf_pts < b->buf_pts;
            return a->frame_num < b->frame_num;
        });

    for (NvDsFrameMeta* frame : frames) {
        auto& ledger = self->ledgers[frame->source_id];
        if (ledger.initialized) {
            if (frame->frame_num <= ledger.last_frame_number) {
                GST_ELEMENT_ERROR(self, CORE, FAILED,
                    ("Non-monotonic frame number, source=%u", frame->source_id),
                    (NULL));
                return GST_FLOW_ERROR;
            }
            if (frame->buf_pts <= static_cast<gint64>(ledger.last_pts_ns)) {
                GST_ELEMENT_ERROR(self, CORE, FAILED,
                    ("Non-monotonic PTS, source=%u", frame->source_id),
                    (NULL));
                return GST_FLOW_ERROR;
            }
        }
        ledger.last_frame_number = frame->frame_num;
        ledger.last_pts_ns = static_cast<std::uint64_t>(frame->buf_pts);
        ledger.initialized = true;
        ledger.processed_frame_count += 1;

        std::vector<AdapterDetection> adapter_detections;
        std::uint32_t seq = 0;
        for (GList* it = frame->obj_meta_list; it != nullptr; it = g_list_next(it)) {
            NvDsObjectMeta* obj = static_cast<NvDsObjectMeta*>(it->data);
            if (obj->class_id != 0) continue;  // only face class
            AdapterDetection ad{};
            ad.obj_meta = obj;
            ad.sequence_index = seq++;
            ad.bbox = bbox_to_rectf(obj->detector_bbox_info.org_bbox_coords);
            ad.detector_score = obj->confidence;
            ad.tracking_only = ad.detector_score < 0.30f;  // TODO: expose public_detection_threshold
            adapter_detections.push_back(ad);
        }

        std::stable_sort(adapter_detections.begin(), adapter_detections.end(),
                         adapter_detection_less);

        if (self->evidence_writer) {
            for (const auto& ad : adapter_detections) {
                mv::tracking::DetectionRecord rec{};
                rec.detection_id = ad.detection_id;
                rec.source_id = static_cast<mv::tracking::SourceId>(frame->source_id);
                rec.frame = static_cast<mv::tracking::FrameNumber>(frame->frame_num);
                rec.pts_ns = static_cast<mv::tracking::TimestampNs>(frame->buf_pts);
                rec.bbox = ad.bbox;
                rec.detector_score = ad.detector_score;
                rec.role = ad.tracking_only
                               ? mv::tracking::DetectionRole::TrackingOnly
                               : mv::tracking::DetectionRole::Public;
                self->evidence_writer->write_detection(rec);
            }
        }

        std::vector<mv::tracking::Detection> core_detections;
        core_detections.reserve(adapter_detections.size());
        std::unordered_map<mv::tracking::DetectionId, NvDsObjectMeta*> det_to_obj;
        std::unordered_map<mv::tracking::DetectionId, bool> det_to_tracking_only;

        for (std::size_t i = 0; i < adapter_detections.size(); ++i) {
            auto& ad = adapter_detections[i];
            ad.detection_id = (static_cast<mv::tracking::DetectionId>(frame->frame_num) << 32u)
                              | static_cast<mv::tracking::DetectionId>(i);
            mv::tracking::Detection d{};
            d.detection_id = ad.detection_id;
            d.bbox = ad.bbox;
            d.detector_score = ad.detector_score;
            d.role = ad.tracking_only
                         ? mv::tracking::DetectionRole::TrackingOnly
                         : mv::tracking::DetectionRole::Public;
            d.sequence_index = ad.sequence_index;
            core_detections.push_back(d);
            det_to_obj[ad.detection_id] = ad.obj_meta;
            det_to_tracking_only[ad.detection_id] = ad.tracking_only;
        }

        mv::tracking::FrameKey frame_key{};
        frame_key.source_id = static_cast<mv::tracking::SourceId>(frame->source_id);
        frame_key.frame_number = static_cast<mv::tracking::FrameNumber>(frame->frame_num);
        frame_key.pts_ns = static_cast<mv::tracking::TimestampNs>(frame->buf_pts);

        mv::tracking::UpdateResult result =
            self->tracker->update(frame_key, core_detections);

        for (const auto& asgn : result.assignments) {
            auto it_obj = det_to_obj.find(asgn.detection_id);
            if (it_obj == det_to_obj.end()) continue;
            NvDsObjectMeta* obj = it_obj->second;

            const mv::tracking::TrackletId global_tracklet_id =
                encode_tracklet_id(frame_key.source_id, asgn.tracklet_id);
            obj->object_id = global_tracklet_id;

            bool is_tracking_only = false;
            auto it_to = det_to_tracking_only.find(asgn.detection_id);
            if (it_to != det_to_tracking_only.end()) is_tracking_only = it_to->second;

            mv::tracker::MvTrackletMeta meta{};
            meta.raw_object_id = obj->object_id;
            meta.tracklet_id = global_tracklet_id;
            meta.source_id = frame_key.source_id;
            meta.frame_number = frame_key.frame_number;
            meta.pts_ns = frame_key.pts_ns;
            meta.tentative = (asgn.state == mv::tracking::TrackState::Tentative);
            meta.confirmed = (asgn.state == mv::tracking::TrackState::Tracked);
            meta.hit_count = 1;  // TODO: expose hit_count from InternalTrack
            meta.lost_count = 0;
            meta.track_age = 1;

            NvDsUserMeta* user_meta =
                mv::tracker::mv_tracklet_meta_create(batch_meta, meta);
            if (user_meta) {
                nvds_add_user_meta_to_obj(obj, user_meta);
            }

            if (is_tracking_only) {
                obj->rect_params.border_width = 0;
            }
        }

        process_ended_tracklets(
            GST_ELEMENT(self), self->evidence_writer, result.ended_tracklets);
    }

    return GST_FLOW_OK;
}

static gboolean gst_mv_face_tracker_sink_event(GstBaseTransform* btrans,
                                               GstEvent* event) {
    auto* self = GST_MV_FACE_TRACKER(btrans);
    if (GST_EVENT_TYPE(event) == GST_EVENT_EOS && self->tracker) {
        auto ended = self->tracker->flush_all(
            mv::tracking::TerminationReason::SourceEos);
        process_ended_tracklets(GST_ELEMENT(self), self->evidence_writer, ended);
        self->ledgers.clear();
        if (self->evidence_writer) {
            self->evidence_writer->close();
        }
    }
    return GST_BASE_TRANSFORM_CLASS(gst_mv_face_tracker_parent_class)
        ->sink_event(btrans, event);
}

static void gst_mv_face_tracker_class_init(GstMvFaceTrackerClass* klass) {
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass* base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class->set_property = gst_mv_face_tracker_set_property;
    gobject_class->get_property = gst_mv_face_tracker_get_property;
    gobject_class->finalize = gst_mv_face_tracker_finalize;

    g_object_class_install_property(gobject_class, PROP_NOMINAL_FRAME_PERIOD_NS,
        g_param_spec_uint64("nominal-frame-period-ns", "Nominal frame period (ns)",
            "Nominal frame period used for Kalman dt in nanoseconds",
            1, G_MAXUINT64, 33333333,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_GPU_ID,
        g_param_spec_int("gpu-id", "GPU ID", "GPU device ID", 0, G_MAXINT, 0,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_EVIDENCE_DIR,
        g_param_spec_string("evidence-dir", "Evidence directory",
            "Directory to write offline tracklet evidence files", "",
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_details_simple(element_class,
        "MergenVision Face Tracker",
        "Filter",
        "ByteTrack-like face tracklet producer for DeepStream",
        "MergenVision");

    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS,
            gst_caps_from_string("video/x-raw(memory:NVMM), format=RGBA")));
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
            gst_caps_from_string("video/x-raw(memory:NVMM), format=RGBA")));

    base_transform_class->transform_caps = gst_mv_face_tracker_transform_caps;
    base_transform_class->start = gst_mv_face_tracker_start;
    base_transform_class->transform_ip = gst_mv_face_tracker_transform_ip;
    base_transform_class->sink_event = gst_mv_face_tracker_sink_event;
}

static void gst_mv_face_tracker_init(GstMvFaceTracker* self) {
    self->tracker_config = default_tracker_config();
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    mvfacetracker,
    "MergenVision face tracker plugin",
    gst_element_register_mvfacetracker,
    "1.0",
    "MIT",
    "MergenVision",
    "MergenVision"
)
