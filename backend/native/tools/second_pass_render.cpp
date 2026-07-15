/* Second-pass offline annotated video renderer.
 *
 * Reads the original encoded MP4 and a labels TSV, decodes with NVDEC,
 * injects NvDsObjectMeta labels on the GPU path, draws with nvdsosd, then
 * re-encodes with nvv4l2h264enc and muxes to MP4.
 *
 * No detector, recognizer, or frame download to CPU.
 */
#include <gst/gst.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <cuda_runtime_api.h>

#include <algorithm>

#include "gstnvdsmeta.h"

#include <cstdio>
#include <cinttypes>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct LabelEntry {
    uint64_t pts_ns = 0;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    std::string label;
    float similarity = 0.0f;
    float det_score = 0.0f;
};

struct AppContext {
    std::string input_path;
    std::string output_path;
    std::string labels_path;
    int gpu_id = 0;
    bool pipeline_error = false;
    GMainLoop* loop = nullptr;
    int64_t frames_processed = 0;
    int64_t labels_drawn = 0;
    std::unordered_map<uint64_t, std::vector<LabelEntry>> labels_by_frame;
    uint32_t stream_width = 1280;
    uint32_t stream_height = 720;
};

static gboolean load_labels(AppContext* ctx) {
    std::ifstream f(ctx->labels_path);
    if (!f) {
        g_printerr("Cannot open labels file: %s\n", ctx->labels_path.c_str());
        return FALSE;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        uint64_t frame = 0;
        LabelEntry e{};
        if (!(iss >> frame >> e.pts_ns >> e.x1 >> e.y1 >> e.x2 >> e.y2 >> e.label >> e.similarity >> e.det_score)) {
            g_warning("Malformed label line: %s", line.c_str());
            continue;
        }
        ctx->labels_by_frame[frame].push_back(std::move(e));
    }
    g_print("Loaded %zu labeled frames\n", ctx->labels_by_frame.size());
    return TRUE;
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

static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    AppContext* ctx = (AppContext*)u_data;
    GstBuffer* buf = (GstBuffer*)info->data;
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    nvds_acquire_meta_lock(batch_meta);
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next) {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)l_frame->data;
        ctx->frames_processed++;
        auto it = ctx->labels_by_frame.find(frame_meta->frame_num);
        if (it == ctx->labels_by_frame.end()) continue;

        for (const LabelEntry& e : it->second) {
            NvDsObjectMeta* obj = nvds_acquire_obj_meta_from_pool(batch_meta);
            if (!obj) continue;

            obj->class_id = 0;
            obj->object_id = static_cast<uint64_t>(-1);
            obj->confidence = e.det_score;

            float w = std::max(0.0f, e.x2 - e.x1);
            float h = std::max(0.0f, e.y2 - e.y1);
            obj->detector_bbox_info.org_bbox_coords = {e.x1, e.y1, w, h};
            obj->rect_params.left = e.x1;
            obj->rect_params.top = e.y1;
            obj->rect_params.width = w;
            obj->rect_params.height = h;
            obj->rect_params.border_width = 2;
            obj->rect_params.border_color = {0.0f, 1.0f, 0.0f, 1.0f};

            gchar* text = g_strdup_printf("%s | sim:%.2f | det:%.2f",
                                          e.label.c_str(), e.similarity, e.det_score);
            obj->text_params.display_text = text;
            obj->text_params.x_offset = static_cast<unsigned int>(std::max(0.0f, e.x1));
            obj->text_params.y_offset = static_cast<unsigned int>(std::max(0.0f, e.y1));
            obj->text_params.font_params.font_name = g_strdup("Serif");
            obj->text_params.font_params.font_size = 12;
            obj->text_params.font_params.font_color = {1.0, 1.0, 1.0, 1.0};
            obj->text_params.set_bg_clr = TRUE;
            obj->text_params.text_bg_clr = {0.0, 0.0, 0.0, 0.7};

            nvds_add_obj_meta_to_frame(frame_meta, obj, nullptr);
            ctx->labels_drawn++;
        }
    }
    nvds_release_meta_lock(batch_meta);
    return GST_PAD_PROBE_OK;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        g_printerr("Usage: %s <input.mp4> <output.mp4> <labels.tsv> <gpu_id> [width height]\n", argv[0]);
        return -1;
    }

    AppContext ctx;
    ctx.input_path = argv[1];
    ctx.output_path = argv[2];
    ctx.labels_path = argv[3];
    ctx.gpu_id = atoi(argv[4]);
    if (argc >= 7) {
        ctx.stream_width = static_cast<uint32_t>(atoi(argv[5]));
        ctx.stream_height = static_cast<uint32_t>(atoi(argv[6]));
    }

    cudaError_t cuerr = cudaSetDevice(ctx.gpu_id);
    if (cuerr != cudaSuccess) {
        g_printerr("cudaSetDevice failed: %s\n", cudaGetErrorString(cuerr));
        return -1;
    }

    if (!load_labels(&ctx)) return -1;

    gst_init(&argc, &argv);

    GMainLoop* loop = g_main_loop_new(nullptr, FALSE);
    ctx.loop = loop;

    GstElement* pipeline = gst_pipeline_new("mv-second-pass-render");
    GstElement* source = gst_element_factory_make("filesrc", "file-source");
    GstElement* qtdemux = gst_element_factory_make("qtdemux", "qt-demuxer");
    GstElement* h264parse = gst_element_factory_make("h264parse", "h264-parser");
    GstElement* decoder = gst_element_factory_make("nvv4l2decoder", "nvdec");
    GstElement* streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    GstElement* osd = gst_element_factory_make("nvdsosd", "osd");
    GstElement* encoder = gst_element_factory_make("nvv4l2h264enc", "encoder");
    GstElement* encparse = gst_element_factory_make("h264parse", "enc-parser");
    GstElement* muxer = gst_element_factory_make("qtmux", "muxer");
    GstElement* sink = gst_element_factory_make("filesink", "file-sink");

    if (!pipeline || !source || !qtdemux || !h264parse || !decoder || !streammux ||
        !osd || !encoder || !encparse || !muxer || !sink) {
        g_printerr("Failed to create one or more GStreamer elements\n");
        return -1;
    }

    g_object_set(G_OBJECT(source), "location", ctx.input_path.c_str(), nullptr);
    g_object_set(G_OBJECT(sink), "location", ctx.output_path.c_str(), nullptr);
    g_object_set(G_OBJECT(streammux),
        "batch-size", 1,
        "width", ctx.stream_width,
        "height", ctx.stream_height,
        "batched-push-timeout", 40000,
        nullptr);

    g_object_set(G_OBJECT(osd),
        "gpu-id", ctx.gpu_id,
        "process-mode", 1,
        "display-text", TRUE,
        "display-bbox", TRUE,
        "display-clock", FALSE,
        nullptr);

    g_object_set(G_OBJECT(encoder),
        "bitrate", 4000000,
        "preset-id", 2,
        "insert-sps-pps", TRUE,
        nullptr);

    gst_bin_add_many(GST_BIN(pipeline), source, qtdemux, h264parse, decoder, streammux,
                     osd, encoder, encparse, muxer, sink, nullptr);

    if (!gst_element_link_many(source, qtdemux, nullptr)) {
        g_printerr("Failed to link filesrc to qtdemux\n");
        return -1;
    }
    g_signal_connect(qtdemux, "pad-added", G_CALLBACK(on_qtdemux_pad_added), h264parse);

    if (!gst_element_link_many(h264parse, decoder, nullptr)) {
        g_printerr("Failed to link parser to decoder\n");
        return -1;
    }

    GstPad* dec_src = gst_element_get_static_pad(decoder, "src");
    GstPad* mux_sink = gst_element_request_pad_simple(streammux, "sink_0");
    if (gst_pad_link(dec_src, mux_sink) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link decoder to streammux\n");
        return -1;
    }
    gst_object_unref(dec_src);
    gst_object_unref(mux_sink);

    if (!gst_element_link_many(streammux, osd, encoder, encparse, muxer, sink, nullptr)) {
        g_printerr("Failed to link pipeline elements\n");
        return -1;
    }

    GstPad* osd_sink_pad = gst_element_get_static_pad(osd, "sink");
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, &ctx, nullptr);
    gst_object_unref(osd_sink_pad);

    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, bus_call, &ctx);
    gst_object_unref(bus);

    g_print("Starting second-pass render: %s -> %s\n", ctx.input_path.c_str(), ctx.output_path.c_str());
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);
    gst_element_set_state(pipeline, GST_STATE_NULL);

    g_print("Frames processed: %" PRId64 "\n", ctx.frames_processed);
    g_print("Labels drawn: %" PRId64 "\n", ctx.labels_drawn);

    gst_object_unref(pipeline);
    g_main_loop_unref(loop);

    return ctx.pipeline_error ? 1 : 0;
}
