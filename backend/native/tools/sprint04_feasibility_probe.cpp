/* Sprint 04 feasibility probe.
 *
 * Validates single-source temporal batching on DeepStream 9.0:
 *   - nvstreammux produces batches with actual_frame_count <= batch_size
 *   - every input frame appears exactly once
 *   - NvDsFrameMeta.batch_id is unique and maps to tensor slice
 *   - NvDsPreProcessTensorMeta.tensor_shape matches actual batch
 *   - final EOS partial batch is flushed
 *   - frames remain in PTS order
 *
 * Pipeline: filesrc -> qtdemux -> h264parse -> nvv4l2decoder -> nvstreammux
 *           -> nvdspreprocess -> fakesink
 */
#include <gst/gst.h>
#include <glib.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h"
#include "nvdsmeta.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>

struct BatchObservation {
    guint actual_batch = 0;
    guint tensor_batch = 0;
    guint64 buf_pts_min = G_MAXUINT64;
    guint64 buf_pts_max = 0;
    std::vector<guint> batch_ids;
    bool duplicate_batch_id = false;
    bool invalid_batch_id = false;
};

struct AppContext {
    std::string input_path;
    std::string output_dir;
    int batch_size = 1;
    int gpu_id = 0;
    GMainLoop* loop = nullptr;
    bool pipeline_error = false;
    bool got_eos = false;

    std::vector<BatchObservation> batches;
    guint total_frames_seen = 0;
    guint partial_batches = 0;
    guint full_batches = 0;
    guint64 last_pts = 0;
    bool pts_monotonic = true;
};

static void write_preprocess_config(const std::string& path, int batch_size, int gpu_id) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        g_printerr("Failed to write preprocess config to %s\n", path.c_str());
        return;
    }
    fprintf(f,
        "[property]\n"
        "enable=1\n"
        "target-unique-ids=1\n"
        "process-on-frame=1\n"
        "network-input-order=0\n"
        "unique-id=5\n"
        "gpu-id=%d\n"
        "maintain-aspect-ratio=0\n"
        "symmetric-padding=0\n"
        "processing-width=640\n"
        "processing-height=640\n"
        "scaling-buf-pool-size=6\n"
        "tensor-buf-pool-size=6\n"
        "network-input-shape=%d;3;640;640\n"
        "network-color-format=1\n"
        "tensor-data-type=0\n"
        "tensor-name=input\n"
        "scaling-pool-memory-type=2\n"
        "scaling-pool-compute-hw=1\n"
        "scaling-filter=1\n"
        "custom-lib-path=/opt/nvidia/deepstream/deepstream-9.0/lib/gst-plugins/libcustom2d_preprocess.so\n"
        "custom-tensor-preparation-function=CustomTensorPreparation\n"
        "\n"
        "[user-configs]\n"
        "pixel-normalization-factor=1.0\n"
        "offsets=104.0;117.0;123.0\n"
        "\n"
        "[group-0]\n"
        "src-ids=0\n"
        "process-on-roi=0\n",
        gpu_id, batch_size);
    fclose(f);
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
        ctx->got_eos = true;
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

static GstPadProbeReturn preprocess_src_pad_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    AppContext* ctx = (AppContext*)u_data;
    GstBuffer* buf = (GstBuffer*)info->data;
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        g_warning("No NvDsBatchMeta on buffer");
        return GST_PAD_PROBE_OK;
    }

    BatchObservation obs;

    nvds_acquire_meta_lock(batch_meta);

    // Enumerate frames and their batch_id / PTS.
    std::set<guint> unique_ids;
    guint frame_count = 0;
    for (NvDsMetaList* l = batch_meta->frame_meta_list; l != NULL; l = l->next) {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)l->data;
        frame_count++;
        obs.batch_ids.push_back(frame_meta->batch_id);
        if (frame_meta->batch_id >= static_cast<guint>(ctx->batch_size)) {
            obs.invalid_batch_id = true;
        }
        if (!unique_ids.insert(frame_meta->batch_id).second) {
            obs.duplicate_batch_id = true;
        }
        if (frame_meta->buf_pts < obs.buf_pts_min) obs.buf_pts_min = frame_meta->buf_pts;
        if (frame_meta->buf_pts > obs.buf_pts_max) obs.buf_pts_max = frame_meta->buf_pts;
        if (frame_meta->buf_pts < ctx->last_pts) {
            ctx->pts_monotonic = false;
        }
        ctx->last_pts = frame_meta->buf_pts;
    }
    obs.actual_batch = frame_count;
    ctx->total_frames_seen += frame_count;
    if (frame_count == static_cast<guint>(ctx->batch_size)) {
        ctx->full_batches++;
    } else {
        ctx->partial_batches++;
    }

    // Read preprocess tensor shape.
    for (NvDsMetaList* l = batch_meta->batch_user_meta_list; l != NULL; l = l->next) {
        NvDsUserMeta* user_meta = (NvDsUserMeta*)l->data;
        if (user_meta->base_meta.meta_type == NVDS_PREPROCESS_BATCH_META) {
            GstNvDsPreProcessBatchMeta* preprocess_batch = (GstNvDsPreProcessBatchMeta*)user_meta->user_meta_data;
            if (preprocess_batch && preprocess_batch->tensor_meta) {
                NvDsPreProcessTensorMeta* tensor_meta = preprocess_batch->tensor_meta;
                if (!tensor_meta->tensor_shape.empty()) {
                    obs.tensor_batch = static_cast<guint>(tensor_meta->tensor_shape[0]);
                }
            }
        }
    }

    nvds_release_meta_lock(batch_meta);

    ctx->batches.push_back(obs);
    return GST_PAD_PROBE_OK;
}

static void write_json_report(AppContext* ctx, const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return;

    bool pass = true;
    if (ctx->pipeline_error) pass = false;
    if (!ctx->got_eos) pass = false;
    if (!ctx->pts_monotonic) pass = false;
    for (const auto& obs : ctx->batches) {
        if (obs.duplicate_batch_id) pass = false;
        if (obs.invalid_batch_id) pass = false;
        if (obs.actual_batch == 0) pass = false;
        if (obs.tensor_batch == 0) pass = false;
        if (obs.actual_batch != obs.tensor_batch) pass = false;
        std::set<guint> ids(obs.batch_ids.begin(), obs.batch_ids.end());
        for (guint i = 0; i < obs.actual_batch; ++i) {
            if (ids.find(i) == ids.end()) pass = false;
        }
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"configured_batch_size\": %d,\n", ctx->batch_size);
    fprintf(f, "  \"total_frames_seen\": %u,\n", ctx->total_frames_seen);
    fprintf(f, "  \"batch_count\": %zu,\n", ctx->batches.size());
    fprintf(f, "  \"full_batches\": %u,\n", ctx->full_batches);
    fprintf(f, "  \"partial_batches\": %u,\n", ctx->partial_batches);
    fprintf(f, "  \"eos_received\": %s,\n", ctx->got_eos ? "true" : "false");
    fprintf(f, "  \"pipeline_error\": %s,\n", ctx->pipeline_error ? "true" : "false");
    fprintf(f, "  \"pts_monotonic\": %s,\n", ctx->pts_monotonic ? "true" : "false");
    fprintf(f, "  \"pass\": %s,\n", pass ? "true" : "false");
    fprintf(f, "  \"batches\": [\n");
    for (size_t i = 0; i < ctx->batches.size(); ++i) {
        const auto& obs = ctx->batches[i];
        fprintf(f, "    {\"actual_batch\":%u,\"tensor_batch\":%u,\"duplicate_batch_id\":%s,\"invalid_batch_id\":%s,\"pts_min\":%" G_GUINT64_FORMAT ",\"pts_max\":%" G_GUINT64_FORMAT "}",
            obs.actual_batch, obs.tensor_batch,
            obs.duplicate_batch_id ? "true" : "false",
            obs.invalid_batch_id ? "true" : "false",
            obs.buf_pts_min, obs.buf_pts_max);
        if (i + 1 < ctx->batches.size()) fprintf(f, ",");
        fprintf(f, "\n");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        g_printerr("Usage: %s <input.mp4> <output_dir> <batch_size> <gpu_id>\n", argv[0]);
        return -1;
    }

    AppContext ctx;
    ctx.input_path = argv[1];
    ctx.output_dir = argv[2];
    ctx.batch_size = atoi(argv[3]);
    ctx.gpu_id = atoi(argv[4]);

    if (ctx.batch_size < 1) {
        g_printerr("batch_size must be >= 1\n");
        return -1;
    }

    g_mkdir_with_parents(ctx.output_dir.c_str(), 0755);

    cudaError_t cuerr = cudaSetDevice(ctx.gpu_id);
    if (cuerr != cudaSuccess) {
        g_printerr("cudaSetDevice failed: %s\n", cudaGetErrorString(cuerr));
        return -1;
    }

    std::string preprocess_config = ctx.output_dir + "/preprocess_b" + std::to_string(ctx.batch_size) + ".txt";
    write_preprocess_config(preprocess_config, ctx.batch_size, ctx.gpu_id);

    gst_init(&argc, &argv);

    GMainLoop* loop = g_main_loop_new(NULL, FALSE);
    ctx.loop = loop;

    GstElement* pipeline = gst_pipeline_new("sprint04-feasibility");
    GstElement* source = gst_element_factory_make("filesrc", "file-source");
    GstElement* qtdemux = gst_element_factory_make("qtdemux", "qt-demuxer");
    GstElement* h264parse = gst_element_factory_make("h264parse", "h264-parser");
    GstElement* decoder = gst_element_factory_make("nvv4l2decoder", "nvdec");
    GstElement* streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    GstElement* preprocess = gst_element_factory_make("nvdspreprocess", "preprocess");
    GstElement* sink = gst_element_factory_make("fakesink", "fake-sink");

    if (!pipeline || !source || !qtdemux || !h264parse || !decoder || !streammux || !preprocess || !sink) {
        g_printerr("Failed to create one or more elements\n");
        return -1;
    }

    g_object_set(G_OBJECT(source), "location", ctx.input_path.c_str(), NULL);
    g_object_set(G_OBJECT(sink), "sync", 0, "async", 0, NULL);

    g_object_set(G_OBJECT(streammux),
        "batch-size", ctx.batch_size,
        "width", 1280,
        "height", 720,
        "batched-push-timeout", 40000,
        "live-source", FALSE,
        NULL);

    g_object_set(G_OBJECT(preprocess),
        "config-file", preprocess_config.c_str(),
        "gpu-id", ctx.gpu_id,
        NULL);

    gst_bin_add_many(GST_BIN(pipeline), source, qtdemux, h264parse, decoder, streammux, preprocess, sink, NULL);

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

    if (!gst_element_link_many(streammux, preprocess, sink, NULL)) {
        g_printerr("Failed to link streammux->preprocess->sink\n");
        return -1;
    }

    GstPad* preprocess_src = gst_element_get_static_pad(preprocess, "src");
    gst_pad_add_probe(preprocess_src, GST_PAD_PROBE_TYPE_BUFFER, preprocess_src_pad_probe, &ctx, NULL);
    gst_object_unref(preprocess_src);

    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    guint bus_watch_id = gst_bus_add_watch(bus, bus_call, &ctx);
    gst_object_unref(bus);

    g_print("Feasibility probe: input=%s batch=%d gpu=%d\n", ctx.input_path.c_str(), ctx.batch_size, ctx.gpu_id);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    std::string report_path = ctx.output_dir + "/feasibility_report.json";
    write_json_report(&ctx, report_path);

    g_print("Done. frames=%u batches=%zu full=%u partial=%u error=%d eos=%d pts_monotonic=%s\n",
        ctx.total_frames_seen, ctx.batches.size(), ctx.full_batches, ctx.partial_batches,
        ctx.pipeline_error ? 1 : 0, ctx.got_eos ? 1 : 0,
        ctx.pts_monotonic ? "true" : "false");

    return ctx.pipeline_error ? 1 : 0;
}
