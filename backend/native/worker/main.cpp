#include <gst/gst.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"

#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <memory>

#define MERGEN_COMPONENT_ID 1
#define FACE_LANDMARK_META_NAME "mv-face-landmark"

struct FaceLandmarkMeta {
    float landmarks[10];
    float score;
};

struct TrackEntry {
    int64_t frame_num;
    double pts_ms;
    float x1, y1, x2, y2;
    float landmarks[10];
    float score;
};

struct WorkerOptions {
    std::string input_path;
    std::string output_dir;
    int gpu_id = 0;
    int batch_size = 1;
    bool tracker_enabled = true;
    std::string tracker_config;
    bool render = false;
    std::string annotated_output;
};

struct AppContext {
    WorkerOptions options;
    bool debug = false;
    bool pipeline_error = false;

    FILE* detections_f = nullptr;
    FILE* manifest_f = nullptr;

    int64_t frames_processed = 0;
    int64_t total_detections = 0;
    int64_t enqueue_count = 0;
    std::map<uint64_t, std::vector<TrackEntry>> tracks;

    GMainLoop* loop = nullptr;

    std::chrono::steady_clock::time_point t_start;
    std::chrono::steady_clock::time_point t_first_frame;
    bool first_frame = true;
};

static bool parse_worker_options(int argc, char** argv, WorkerOptions* opts) {
    opts->tracker_config = "/app/backend/native/configs/tracker_NvDCF_mergen.yml";
    if (argc < 4) return false;
    opts->input_path = argv[1];
    opts->output_dir = argv[2];
    opts->gpu_id = atoi(argv[3]);

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--batch-size" && i + 1 < argc) {
            opts->batch_size = atoi(argv[++i]);
            if (opts->batch_size <= 0) {
                g_printerr("Invalid batch-size: must be > 0\n");
                return false;
            }
        } else if (arg == "--tracker" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "off") {
                opts->tracker_enabled = false;
            } else {
                opts->tracker_enabled = true;
                opts->tracker_config = val;
            }
        } else if (arg == "--annotated-output" && i + 1 < argc) {
            opts->annotated_output = argv[++i];
            opts->render = true;
        } else if (arg == "--render") {
            opts->render = true;
        } else {
            g_printerr("Unknown option: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

static std::string write_runtime_streammux_config(const std::string& output_dir,
                                                    int batch_size) {
    std::string path = output_dir + "/streammux_b" + std::to_string(batch_size) + ".txt";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return "";
    fprintf(f, "[property]\n");
    fprintf(f, "algorithm-type=1\n");
    fprintf(f, "batch-size=%d\n", batch_size);
    fprintf(f, "max-same-source-frames=%d\n", batch_size);
    fprintf(f, "adaptive-batching=0\n");
    fclose(f);
    return path;
}

static std::string write_runtime_preprocess_config(const std::string& output_dir,
                                                   int batch_size, int gpu_id) {
    std::string path = output_dir + "/retinaface_preprocess_b" + std::to_string(batch_size) + ".txt";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return "";
    // nvdspreprocess buf-pool-size values are buffer counts, not MB.
    int pool_count = std::max(8, batch_size * 2);
    fprintf(f, "[property]\n");
    fprintf(f, "enable=1\n");
    fprintf(f, "target-unique-ids=1\n");
    fprintf(f, "process-on-frame=1\n");
    fprintf(f, "network-input-order=0\n");
    fprintf(f, "unique-id=5\n");
    fprintf(f, "gpu-id=%d\n", gpu_id);
    fprintf(f, "maintain-aspect-ratio=0\n");
    fprintf(f, "symmetric-padding=0\n");
    fprintf(f, "processing-width=640\n");
    fprintf(f, "processing-height=640\n");
    fprintf(f, "scaling-buf-pool-size=%d\n", pool_count);
    fprintf(f, "tensor-buf-pool-size=%d\n", pool_count);
    fprintf(f, "network-input-shape=%d;3;640;640\n", batch_size);
    fprintf(f, "network-color-format=1\n");
    fprintf(f, "tensor-data-type=0\n");
    fprintf(f, "tensor-name=input\n");
    fprintf(f, "scaling-pool-memory-type=2\n");
    fprintf(f, "scaling-pool-compute-hw=1\n");
    fprintf(f, "scaling-filter=1\n");
    fprintf(f, "custom-lib-path=/opt/nvidia/deepstream/deepstream-9.0/lib/gst-plugins/libcustom2d_preprocess.so\n");
    fprintf(f, "custom-tensor-preparation-function=CustomTensorPreparation\n");
    fprintf(f, "\n[user-configs]\n");
    fprintf(f, "pixel-normalization-factor=1.0\n");
    fprintf(f, "offsets=104.0;117.0;123.0\n");
    fprintf(f, "\n[group-0]\n");
    fprintf(f, "src-ids=0\n");
    fprintf(f, "process-on-roi=0\n");
    fclose(f);
    return path;
}

static void xset_int_if_exists(GObject* obj, const gchar* name, gint value) {
    if (g_object_class_find_property(G_OBJECT_GET_CLASS(obj), name)) {
        g_object_set(obj, name, value, NULL);
    }
}

static void xset_string_if_exists(GObject* obj, const gchar* name, const gchar* value) {
    if (g_object_class_find_property(G_OBJECT_GET_CLASS(obj), name)) {
        g_object_set(obj, name, value, NULL);
    }
}

static void xset_bool_if_exists(GObject* obj, const gchar* name, gboolean value) {
    if (g_object_class_find_property(G_OBJECT_GET_CLASS(obj), name)) {
        g_object_set(obj, name, value, NULL);
    }
}

static void writeJsonEscapeString(FILE* f, const char* s) {
    fputc('"', f);
    for (; *s; ++s) {
        switch (*s) {
            case '"': fprintf(f, "\\\""); break;
            case '\\': fprintf(f, "\\\\"); break;
            case '\b': fprintf(f, "\\b"); break;
            case '\f': fprintf(f, "\\f"); break;
            case '\n': fprintf(f, "\\n"); break;
            case '\r': fprintf(f, "\\r"); break;
            case '\t': fprintf(f, "\\t"); break;
            default: fputc(*s, f); break;
        }
    }
    fputc('"', f);
}

static NvDsMetaType g_face_landmark_meta_type = NVDS_USER_META;

static void ensure_face_landmark_meta_type() {
    static gboolean initialized = FALSE;
    if (!initialized) {
        g_face_landmark_meta_type = nvds_get_user_meta_type((gchar*)FACE_LANDMARK_META_NAME);
        if (g_face_landmark_meta_type == 0) {
            g_face_landmark_meta_type = NVDS_USER_META;
        }
        initialized = TRUE;
    }
}

static int g_streammux_batch_total = 0;
static int g_streammux_buffer_count = 0;

static GstPadProbeReturn streammux_src_pad_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    (void)pad; (void)u_data;
    GstBuffer* buf = (GstBuffer*)info->data;
    NvDsBatchMeta* bm = gst_buffer_get_nvds_batch_meta(buf);
    int count = 0;
    if (bm) {
        nvds_acquire_meta_lock(bm);
        for (NvDsMetaList* l = bm->frame_meta_list; l != NULL; l = l->next) count++;
        nvds_release_meta_lock(bm);
    }
    g_streammux_buffer_count++;
    g_streammux_batch_total += count;
    return GST_PAD_PROBE_OK;
}

static const FaceLandmarkMeta* find_landmark_meta(NvDsObjectMeta* obj_meta) {
    for (NvDsMetaList* l = obj_meta->obj_user_meta_list; l != NULL; l = l->next) {
        NvDsUserMeta* um = (NvDsUserMeta*)l->data;
        if (um->base_meta.meta_type == g_face_landmark_meta_type && um->user_meta_data) {
            return reinterpret_cast<const FaceLandmarkMeta*>(um->user_meta_data);
        }
    }
    return nullptr;
}

static GstPadProbeReturn detector_src_pad_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    (void)pad;
    AppContext* ctx = (AppContext*)u_data;
    GstBuffer* buf = (GstBuffer*)info->data;
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    ctx->enqueue_count++;

    nvds_acquire_meta_lock(batch_meta);
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)l_frame->data;
        if (ctx->first_frame) {
            ctx->t_first_frame = std::chrono::steady_clock::now();
            ctx->first_frame = false;
        }

        ctx->frames_processed++;
        fprintf(ctx->detections_f, "{\"frame\":%" PRId64 ",\"pts_ms\":%.3f,\"width\":%d,\"height\":%d,\"detections\":[",
            (int64_t)frame_meta->frame_num, (double)frame_meta->buf_pts / 1000000.0,
            frame_meta->source_frame_width, frame_meta->source_frame_height);

        int det_count = 0;
        for (NvDsMetaList* l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)l_obj->data;
            if (obj_meta->class_id != 0) continue;

            float x1 = obj_meta->rect_params.left;
            float y1 = obj_meta->rect_params.top;
            float x2 = x1 + obj_meta->rect_params.width;
            float y2 = y1 + obj_meta->rect_params.height;
            float score = obj_meta->confidence;

            const FaceLandmarkMeta* lm = find_landmark_meta(obj_meta);
            float landmarks[10] = {};
            if (lm) {
                std::memcpy(landmarks, lm->landmarks, sizeof(landmarks));
            }

            bool sane = score >= 0.0f && score <= 1.0f
                && x1 < x2 && y1 < y2
                && x1 >= 0.0f && y1 >= 0.0f
                && x2 <= frame_meta->source_frame_width
                && y2 <= frame_meta->source_frame_height;
            for (int k = 0; sane && k < 5; ++k) {
                float lx = landmarks[k * 2];
                float ly = landmarks[k * 2 + 1];
                sane = lx >= 0.0f && ly >= 0.0f && lx <= frame_meta->source_frame_width && ly <= frame_meta->source_frame_height;
            }
            if (!sane) {
                g_warning("Frame %d det %d failed semantic sanity; skipping JSON write",
                    frame_meta->frame_num, det_count);
                continue;
            }

            uint64_t track_id = obj_meta->object_id;
            if (det_count > 0) fprintf(ctx->detections_f, ",");
            fprintf(ctx->detections_f,
                "{\"det_id\":%d,\"track_id\":%" PRIu64 ",\"x1\":%.3f,\"y1\":%.3f,\"x2\":%.3f,\"y2\":%.3f,\"score\":%.4f,\"landmarks\":[",
                det_count, track_id, x1, y1, x2, y2, score);
            for (int k = 0; k < 10; ++k) {
                if (k) fprintf(ctx->detections_f, ",");
                fprintf(ctx->detections_f, "%.3f", landmarks[k]);
            }
            fprintf(ctx->detections_f, "]}");

            TrackEntry e;
            e.frame_num = frame_meta->frame_num;
            e.pts_ms = (double)frame_meta->buf_pts / 1000000.0;
            e.x1 = x1; e.y1 = y1; e.x2 = x2; e.y2 = y2;
            std::memcpy(e.landmarks, landmarks, sizeof(e.landmarks));
            e.score = score;
            ctx->tracks[track_id].push_back(e);
            ctx->total_detections++;
            det_count++;
        }
        fprintf(ctx->detections_f, "]}\n");
    }
    nvds_release_meta_lock(batch_meta);
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    (void)pad;
    AppContext* ctx = (AppContext*)u_data;
    GstBuffer* buf = (GstBuffer*)info->data;
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    nvds_acquire_meta_lock(batch_meta);
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)l_frame->data;
        int det_id = 0;
        for (NvDsMetaList* l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)l_obj->data;
            if (obj_meta->class_id != 0) continue;

            obj_meta->rect_params.border_width = 2;
            obj_meta->rect_params.border_color.red = 0.0f;
            obj_meta->rect_params.border_color.green = 1.0f;
            obj_meta->rect_params.border_color.blue = 0.0f;
            obj_meta->rect_params.border_color.alpha = 1.0f;

            gchar* text = g_strdup_printf("det:%d %.2f", det_id, obj_meta->confidence);
            obj_meta->text_params.display_text = text;
            obj_meta->text_params.x_offset = (unsigned int)std::max(0.0f, obj_meta->rect_params.left);
            obj_meta->text_params.y_offset = (unsigned int)std::max(0.0f, obj_meta->rect_params.top - 14.0f);
            obj_meta->text_params.font_params.font_name = (gchar*)"Serif";
            obj_meta->text_params.font_params.font_size = 12;
            obj_meta->text_params.font_params.font_color = {1.0f, 1.0f, 1.0f, 1.0f};
            obj_meta->text_params.set_bg_clr = TRUE;
            obj_meta->text_params.text_bg_clr = {0.0f, 0.0f, 0.0f, 0.7f};

            det_id++;
        }
    }
    nvds_release_meta_lock(batch_meta);
    return GST_PAD_PROBE_OK;
}

static void on_qtdemux_pad_added(GstElement* element, GstPad* pad, gpointer data) {
    GstElement* parser = (GstElement*)data;
    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (!caps) return;
    GstStructure* s = gst_caps_get_structure(caps, 0);
    const gchar* name = gst_structure_get_name(s);
    if (g_str_has_prefix(name, "video/")) {
        GstPad* sinkpad = gst_element_get_static_pad(parser, "sink");
        if (sinkpad) {
            if (GST_PAD_LINK_FAILED(gst_pad_link(pad, sinkpad))) {
                g_warning("Failed to link qtdemux to parser");
            }
            gst_object_unref(sinkpad);
        }
    }
    gst_caps_unref(caps);
}

static gboolean bus_call(GstBus* bus, GstMessage* msg, gpointer data) {
    AppContext* ctx = (AppContext*)data;
    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        g_print("End of stream\n");
        g_main_loop_quit(ctx->loop);
        break;
    case GST_MESSAGE_ERROR: {
        GError* error = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(msg, &error, &debug);
        g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
        if (debug) g_printerr("Debug: %s\n", debug);
        g_error_free(error);
        g_free(debug);
        ctx->pipeline_error = true;
        g_main_loop_quit(ctx->loop);
        break;
    }
    default:
        break;
    }
    return TRUE;
}

static gboolean ensure_nvdsretinaface_plugin() {
    if (gst_element_factory_find("nvdsretinaface")) return TRUE;

    const char* build_dir = "/app/backend/native/build";
    std::string plugin_path = std::string(build_dir) + "/libgstnvdsretinaface.so";
    if (g_access(plugin_path.c_str(), F_OK) != 0) {
        plugin_path = "backend/native/build/libgstnvdsretinaface.so";
    }
    GstPlugin* plugin = gst_plugin_load_file(plugin_path.c_str(), nullptr);
    if (!plugin) {
        g_printerr("Failed to load libgstnvdsretinaface.so from %s\n", plugin_path.c_str());
        return FALSE;
    }
    gst_object_unref(plugin);
    return gst_element_factory_find("nvdsretinaface") != nullptr;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        g_printerr("Usage: %s <input.mp4> <output_dir> <gpu_id> [tracker_config.yml]\n", argv[0]);
        return -1;
    }

    WorkerOptions opts;
    if (!parse_worker_options(argc, argv, &opts)) {
        g_printerr("Usage: %s <input.mp4> <output_dir> <gpu_id> [--batch-size N] [--tracker off|<config.yml>] [--render] [--annotated-output <path.mp4>]\n", argv[0]);
        return -1;
    }
    if (opts.render && opts.annotated_output.empty()) {
        opts.annotated_output = opts.output_dir + "/annotated.mp4";
    }
    AppContext ctx;
    ctx.options = opts;

    if (opts.tracker_config.empty()) {
        opts.tracker_config = "/app/backend/native/configs/tracker_NvDCF_mergen.yml";
    }
    if (g_access(opts.tracker_config.c_str(), F_OK) != 0 && opts.tracker_config[0] != '/') {
        opts.tracker_config = std::string("backend/native/configs/") + opts.tracker_config;
    }

    g_mkdir_with_parents(opts.output_dir.c_str(), 0755);

    std::string detections_path = opts.output_dir + "/detections.jsonl";
    ctx.detections_f = fopen(detections_path.c_str(), "w");
    if (!ctx.detections_f) {
        g_printerr("Failed to open %s\n", detections_path.c_str());
        return -1;
    }

    cudaError_t cuerr = cudaSetDevice(opts.gpu_id);
    if (cuerr != cudaSuccess) {
        g_printerr("cudaSetDevice failed: %s\n", cudaGetErrorString(cuerr));
        return -1;
    }

    // GPU/Runtime visibility for manifest.
    int visible_gpus = 0;
    cudaGetDeviceCount(&visible_gpus);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, opts.gpu_id);
    char uuid_str[40];
    for (int i = 0; i < 16; ++i) {
        snprintf(uuid_str + i * 2, 3, "%02x", (unsigned char)prop.uuid.bytes[i]);
    }
    uuid_str[32] = '\0';
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    g_print("gpu_visible=%d gpu_id=%d uuid=%s name=%s driver=%d.%d cuda=%d.%d\n",
        visible_gpus, opts.gpu_id, uuid_str, prop.name,
        driver_version / 1000, (driver_version % 1000) / 10,
        runtime_version / 1000, (runtime_version % 1000) / 10);

    gst_init(&argc, &argv);
    if (!ensure_nvdsretinaface_plugin()) {
        g_printerr("nvdsretinaface element not available\n");
        return -1;
    }
    ensure_face_landmark_meta_type();

    GMainLoop* loop = g_main_loop_new(NULL, FALSE);
    ctx.loop = loop;

    GstElement* pipeline = gst_pipeline_new("mergenvision-deepstream-worker");
    GstElement* source = gst_element_factory_make("filesrc", "file-source");
    GstElement* qtdemux = gst_element_factory_make("qtdemux", "qt-demuxer");
    GstElement* h264parse = gst_element_factory_make("h264parse", "h264-parser");
    GstElement* decoder = gst_element_factory_make("nvv4l2decoder", "nvdec");
    GstElement* streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    GstElement* preprocess = gst_element_factory_make("nvdspreprocess", "preprocess");
    GstElement* retinaface = gst_element_factory_make("nvdsretinaface", "retinaface");
    GstElement* tracker = opts.tracker_enabled ? gst_element_factory_make("nvtracker", "tracker") : nullptr;

    GstElement* osd = nullptr;
    GstElement* encoder = nullptr;
    GstElement* encparse = nullptr;
    GstElement* muxer = nullptr;
    GstElement* filesink = nullptr;
    GstElement* fakesink = nullptr;
    GstElement* demux = nullptr;
    GstElement* queue = nullptr;

    if (opts.render) {
        osd = gst_element_factory_make("nvdsosd", "osd");
        encoder = gst_element_factory_make("nvv4l2h264enc", "encoder");
        encparse = gst_element_factory_make("h264parse", "enc-parser");
        muxer = gst_element_factory_make("qtmux", "muxer");
        filesink = gst_element_factory_make("filesink", "file-sink");
        demux = gst_element_factory_make("nvstreamdemux", "render-demux");
        queue = gst_element_factory_make("queue", "post-demux-queue");
    } else {
        fakesink = gst_element_factory_make("fakesink", "fake-sink");
        if (!opts.tracker_enabled) {
            demux = gst_element_factory_make("nvstreamdemux", "demux");
        }
        queue = gst_element_factory_make("queue", "post-detector-queue");
    }

    if (!pipeline || !source || !qtdemux || !h264parse || !decoder || !streammux ||
        !preprocess || !retinaface ||
        (opts.tracker_enabled && !tracker) ||
        (opts.render && (!osd || !encoder || !encparse || !muxer || !filesink || !demux || !queue)) ||
        (!opts.render && !fakesink) ||
        (!opts.render && !queue) ||
        (!opts.render && !opts.tracker_enabled && !demux)) {
        g_printerr("Failed to create one or more elements\n");
        return -1;
    }

    g_object_set(G_OBJECT(source), "location", opts.input_path.c_str(), NULL);
    if (opts.render) {
        g_object_set(G_OBJECT(filesink), "location", opts.annotated_output.c_str(), NULL);
        g_object_set(G_OBJECT(osd),
            "gpu-id", opts.gpu_id,
            "process-mode", 1,
            "display-text", TRUE,
            "display-bbox", TRUE,
            "display-clock", FALSE,
            NULL);
        g_object_set(G_OBJECT(encoder),
            "bitrate", 4000000,
            "preset-id", 2,
            "insert-sps-pps", TRUE,
            NULL);
    } else {
        g_object_set(G_OBJECT(fakesink), "sync", 0, "async", 0, "qos", 0, NULL);
    }

    std::string preprocess_config = write_runtime_preprocess_config(
        opts.output_dir, opts.batch_size, opts.gpu_id);
    if (preprocess_config.empty()) {
        g_printerr("Failed to write runtime preprocess config\n");
        return -1;
    }

    std::string streammux_config = write_runtime_streammux_config(
        opts.output_dir, opts.batch_size);
    if (streammux_config.empty()) {
        g_printerr("Failed to write runtime streammux config\n");
        return -1;
    }

    int batched_push_timeout_us = 400000;
    if (const char* env_t = std::getenv("MV_BATCH_PUSH_TIMEOUT_US")) {
        batched_push_timeout_us = std::atoi(env_t);
        if (batched_push_timeout_us <= 0) batched_push_timeout_us = 400000;
    }
    g_object_set(G_OBJECT(streammux),
        "batch-size", opts.batch_size,
        "batched-push-timeout", batched_push_timeout_us,
        NULL);
    xset_int_if_exists(G_OBJECT(streammux), "width", 1280);
    xset_int_if_exists(G_OBJECT(streammux), "height", 720);
    xset_string_if_exists(G_OBJECT(streammux), "config-file-path", streammux_config.c_str());
    int mux_pool_size = std::max(4, opts.batch_size * 2);
    xset_int_if_exists(G_OBJECT(streammux), "buffer-pool-size", mux_pool_size);
    xset_int_if_exists(G_OBJECT(streammux), "failsafe-flush-count", mux_pool_size - 1);
    xset_bool_if_exists(G_OBJECT(streammux), "live-source", FALSE);

    g_object_set(G_OBJECT(preprocess),
        "config-file", preprocess_config.c_str(),
        "gpu-id", opts.gpu_id,
        NULL);

    std::string engine_path = "/app/backend/artifacts/engines/retinaface_r50_dynamic.bs1.opt64.max256.fp16.trt1014.engine";
    if (g_access(engine_path.c_str(), F_OK) != 0) {
        engine_path = "backend/artifacts/engines/retinaface_r50_dynamic.bs1.opt64.max256.fp16.trt1014.engine";
    }
    float conf_threshold = 0.5f;
    if (const char* diag_conf = std::getenv("MV_DIAG_CONF_THRESHOLD")) {
        conf_threshold = static_cast<float>(std::atof(diag_conf));
    }
    g_object_set(G_OBJECT(retinaface),
        "engine-file", engine_path.c_str(),
        "gpu-id", opts.gpu_id,
        "conf-threshold", conf_threshold,
        "nms-threshold", 0.4f,
        NULL);

    GstElement* probe_element = retinaface;
    if (opts.tracker_enabled) {
        g_object_set(G_OBJECT(tracker),
            "ll-lib-file", "/opt/nvidia/deepstream/deepstream-9.0/lib/libnvds_nvmultiobjecttracker.so",
            "ll-config-file", opts.tracker_config.c_str(),
            "tracker-width", 640,
            "tracker-height", 384,
            "operate-on-class-ids", "0",
            "gpu-id", opts.gpu_id,
            NULL);
        g_print("tracker-config=%s\n", opts.tracker_config.c_str());
        if (opts.render) {
            gst_bin_add_many(GST_BIN(pipeline), source, qtdemux, h264parse, decoder, streammux,
                preprocess, retinaface, tracker, demux, queue, osd, encoder, encparse, muxer, filesink, NULL);
            if (!gst_element_link_many(streammux, preprocess, retinaface, tracker, demux, NULL)) {
                g_printerr("Failed to link streammux->...->demux\n");
                return -1;
            }
        } else {
            gst_bin_add_many(GST_BIN(pipeline), source, qtdemux, h264parse, decoder, streammux,
                preprocess, retinaface, tracker, queue, fakesink, NULL);
            if (!gst_element_link_many(streammux, preprocess, retinaface, tracker, queue, fakesink, NULL)) {
                g_printerr("Failed to link streammux->preprocess->retinaface->tracker->queue->fakesink\n");
                return -1;
            }
        }
        probe_element = tracker;
    } else {
        if (opts.render) {
            gst_bin_add_many(GST_BIN(pipeline), source, qtdemux, h264parse, decoder, streammux,
                preprocess, retinaface, demux, queue, osd, encoder, encparse, muxer, filesink, NULL);
            if (!gst_element_link_many(streammux, preprocess, retinaface, demux, NULL)) {
                g_printerr("Failed to link streammux->...->demux\n");
                return -1;
            }
        } else {
            gst_bin_add_many(GST_BIN(pipeline), source, qtdemux, h264parse, decoder, streammux,
                preprocess, retinaface, demux, queue, fakesink, NULL);
            if (!gst_element_link_many(streammux, preprocess, retinaface, demux, NULL)) {
                g_printerr("Failed to link streammux->preprocess->retinaface->demux\n");
                return -1;
            }
            GstPad* demux_src = gst_element_request_pad_simple(demux, "src_0");
            GstPad* queue_sink = gst_element_get_static_pad(queue, "sink");
            if (gst_pad_link(demux_src, queue_sink) != GST_PAD_LINK_OK) {
                g_printerr("Failed to link demux->queue\n");
                gst_object_unref(demux_src);
                gst_object_unref(queue_sink);
                return -1;
            }
            gst_object_unref(demux_src);
            gst_object_unref(queue_sink);
            if (!gst_element_link_many(queue, fakesink, NULL)) {
                g_printerr("Failed to link queue->fakesink\n");
                return -1;
            }
        }
    }

    if (opts.render) {
        GstPad* demux_src = gst_element_request_pad_simple(demux, "src_0");
        GstPad* queue_sink = gst_element_get_static_pad(queue, "sink");
        if (!demux_src || !queue_sink || gst_pad_link(demux_src, queue_sink) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link render demux->queue\n");
            if (demux_src) gst_object_unref(demux_src);
            if (queue_sink) gst_object_unref(queue_sink);
            return -1;
        }
        gst_object_unref(demux_src);
        gst_object_unref(queue_sink);
        if (!gst_element_link_many(queue, osd, encoder, encparse, muxer, filesink, NULL)) {
            g_printerr("Failed to link queue->osd->encoder->muxer->filesink\n");
            return -1;
        }
    }

    GstPad* probe_pad = gst_element_get_static_pad(probe_element, "src");
    gst_pad_add_probe(probe_pad, GST_PAD_PROBE_TYPE_BUFFER, detector_src_pad_buffer_probe, &ctx, NULL);
    gst_object_unref(probe_pad);

    if (opts.render) {
        GstPad* osd_sink = gst_element_get_static_pad(osd, "sink");
        gst_pad_add_probe(osd_sink, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe, &ctx, NULL);
        gst_object_unref(osd_sink);
    }

    if (!gst_element_link_many(source, qtdemux, NULL)) {
        g_printerr("Failed to link filesrc->qtdemux\n");
        return -1;
    }
    g_signal_connect(qtdemux, "pad-added", G_CALLBACK(on_qtdemux_pad_added), h264parse);

    if (!gst_element_link_many(h264parse, decoder, NULL)) {
        g_printerr("Failed to link parser->decoder\n");
        return -1;
    }

    GstPad* dec_src = gst_element_get_static_pad(decoder, "src");
    GstPad* mux_sink = gst_element_request_pad_simple(streammux, "sink_0");
    if (gst_pad_link(dec_src, mux_sink) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link decoder->streammux\n");
        return -1;
    }
    gst_object_unref(dec_src);
    gst_object_unref(mux_sink);

    GstPad* mux_src = gst_element_get_static_pad(streammux, "src");
    if (mux_src) {
        gst_pad_add_probe(mux_src, GST_PAD_PROBE_TYPE_BUFFER,
                          streammux_src_pad_buffer_probe, &ctx, NULL);
        gst_object_unref(mux_src);
    }

    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    guint bus_watch_id = gst_bus_add_watch(bus, bus_call, &ctx);
    gst_object_unref(bus);

    ctx.t_start = std::chrono::steady_clock::now();

    g_print("Starting pipeline on GPU %d batch=%d tracker=%s for %s\n",
        opts.gpu_id, opts.batch_size, opts.tracker_enabled ? "on" : "off", opts.input_path.c_str());
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    auto t_end = std::chrono::steady_clock::now();
    double wall_sec = std::chrono::duration<double>(t_end - ctx.t_start).count();
    double first_frame_latency = 0.0;
    if (!ctx.first_frame) {
        first_frame_latency = std::chrono::duration<double>(ctx.t_first_frame - ctx.t_start).count();
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    int exit_code = ctx.pipeline_error ? 1 : 0;

    std::string manifest_path = opts.output_dir + "/run_manifest.json";
    ctx.manifest_f = fopen(manifest_path.c_str(), "w");
    const char* container_image = g_getenv("MV_CONTAINER_IMAGE");
    const char* ds_image_digest = g_getenv("MV_DEEPSTREAM_IMAGE_DIGEST");
    guint gst_major, gst_minor, gst_micro, gst_nano;
    gst_version(&gst_major, &gst_minor, &gst_micro, &gst_nano);
    if (ctx.manifest_f) {
        fprintf(ctx.manifest_f, "{\n");
        fprintf(ctx.manifest_f, "  \"input\": ");
        writeJsonEscapeString(ctx.manifest_f, opts.input_path.c_str());
        fprintf(ctx.manifest_f, ",\n");
        fprintf(ctx.manifest_f, "  \"gpu_id\": %d,\n", opts.gpu_id);
        fprintf(ctx.manifest_f, "  \"gpu_uuid\": \"%s\",\n", uuid_str);
        fprintf(ctx.manifest_f, "  \"gpu_name\": \"%s\",\n", prop.name);
        fprintf(ctx.manifest_f, "  \"gpu_visible_count\": %d,\n", visible_gpus);
        fprintf(ctx.manifest_f, "  \"cuda_driver_version\": \"%d.%d\",\n", driver_version / 1000, (driver_version % 1000) / 10);
        fprintf(ctx.manifest_f, "  \"cuda_runtime_version\": \"%d.%d\",\n", runtime_version / 1000, (runtime_version % 1000) / 10);
        fprintf(ctx.manifest_f, "  \"deepstream_version\": \"9.0-triton-multiarch\",\n");
        fprintf(ctx.manifest_f, "  \"deepstream_image\": \"%s\",\n", container_image ? container_image : "nvcr.io/nvidia/deepstream:9.0-triton-multiarch");
        fprintf(ctx.manifest_f, "  \"deepstream_image_digest\": \"%s\",\n", ds_image_digest ? ds_image_digest : "unknown");
        fprintf(ctx.manifest_f, "  \"tensorrt_version\": \"10.14.1.48+cuda13.0\",\n");
        fprintf(ctx.manifest_f, "  \"gstreamer_version\": \"%u.%u.%u\",\n", gst_major, gst_minor, gst_micro);
        fprintf(ctx.manifest_f, "  \"configured_batch_size\": %d,\n", opts.batch_size);
        fprintf(ctx.manifest_f, "  \"tracker_enabled\": %s,\n", opts.tracker_enabled ? "true" : "false");
        fprintf(ctx.manifest_f, "  \"enqueue_count\": %" PRId64 ",\n", ctx.enqueue_count);
        fprintf(ctx.manifest_f, "  \"frames_processed\": %" PRId64 ",\n", ctx.frames_processed);
        fprintf(ctx.manifest_f, "  \"total_detections\": %" PRId64 ",\n", ctx.total_detections);
        fprintf(ctx.manifest_f, "  \"unique_raw_track_ids\": %zu,\n", ctx.tracks.size());
        fprintf(ctx.manifest_f, "  \"wall_time_sec\": %.3f,\n", wall_sec);
        fprintf(ctx.manifest_f, "  \"first_frame_latency_sec\": %.3f,\n", first_frame_latency);
        fprintf(ctx.manifest_f, "  \"preprocess_config\": ");
        writeJsonEscapeString(ctx.manifest_f, preprocess_config.c_str());
        fprintf(ctx.manifest_f, ",\n");
        fprintf(ctx.manifest_f, "  \"detector\": \"nvdsretinaface-gpu\",\n");
        fprintf(ctx.manifest_f, "  \"tracker\": \"%s\",\n", opts.tracker_enabled ? "nvtracker" : "none");
        fprintf(ctx.manifest_f, "  \"completed\": %s,\n", ctx.pipeline_error ? "false" : "true");
        fprintf(ctx.manifest_f, "  \"eos_clean\": %s,\n", ctx.pipeline_error ? "false" : "true");
        fprintf(ctx.manifest_f, "  \"exit_code\": %d\n", exit_code);
        fprintf(ctx.manifest_f, "}\n");
        fclose(ctx.manifest_f);
    }

    if (opts.tracker_enabled) {
    std::string tracks_path = opts.output_dir + "/tracks.json";
    FILE* tracks_f = fopen(tracks_path.c_str(), "w");
    if (tracks_f) {
        fprintf(tracks_f, "{\n");
        int ti = 0;
        for (const auto& kv : ctx.tracks) {
            if (ti++ > 0) fprintf(tracks_f, ",\n");
            fprintf(tracks_f, "  \"%" PRIu64 "\": {\"count\":%zu,\"entries\":[", (uint64_t)kv.first, kv.second.size());
            for (size_t j = 0; j < kv.second.size(); ++j) {
                const TrackEntry& e = kv.second[j];
                if (j) fprintf(tracks_f, ",");
                fprintf(tracks_f,
                    "{\"frame\":%" PRId64 ",\"pts_ms\":%.3f,\"x1\":%.3f,\"y1\":%.3f,\"x2\":%.3f,\"y2\":%.3f,\"score\":%.4f}",
                    e.frame_num, e.pts_ms, e.x1, e.y1, e.x2, e.y2, e.score);
            }
            fprintf(tracks_f, "]}");
        }
        fprintf(tracks_f, "\n}\n");
        fclose(tracks_f);
    }
    }

    if (ctx.detections_f) fclose(ctx.detections_f);

    g_print("streammux summary: buffers=%d frames_total=%d avg_batch=%.2f\n",
        g_streammux_buffer_count, g_streammux_batch_total,
        g_streammux_buffer_count ? (double)g_streammux_batch_total / g_streammux_buffer_count : 0.0);
    g_print("Done. frames=%" PRId64 " detections=%" PRId64 " enqueue=%" PRId64 " wall=%.3fs error=%d\n",
        ctx.frames_processed, ctx.total_detections, ctx.enqueue_count, wall_sec, ctx.pipeline_error ? 1 : 0);
    g_print("completed=%s decoded_frames=%" PRId64 " processed_frames=%" PRId64 " detections=%" PRId64 " enqueue=%" PRId64 " tracklets=%zu eos_clean=%s exit_code=%d\n",
        ctx.pipeline_error ? "false" : "true",
        ctx.frames_processed, ctx.frames_processed, ctx.total_detections, ctx.enqueue_count, ctx.tracks.size(),
        ctx.pipeline_error ? "false" : "true", exit_code);
    return exit_code;
}
