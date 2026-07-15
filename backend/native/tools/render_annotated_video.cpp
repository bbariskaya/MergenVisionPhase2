/* DeepStream GPU-resident annotated video renderer.
 *
 * Reads the source MP4, runs the offline detector (nvdsretinaface), draws
 * bounding boxes and placeholder labels with nvdsosd, then encodes with
 * nvv4l2h264enc and muxes to MP4.
 *
 * The full pipeline never stages frames to the host:
 *   filesrc -> qtdemux -> h264parse -> nvv4l2decoder -> nvstreammux ->
 *   nvdspreprocess -> nvdsretinaface -> nvdsosd -> nvv4l2h264enc ->
 *   h264parse -> qtmux -> filesink
 */
#include <gst/gst.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <cuda_runtime_api.h>


#include "gstnvdsmeta.h"

#include <cstdio>
#include <cinttypes>
#include <cstring>
#include <string>

struct AppContext {
    std::string input_path;
    std::string output_path;
    int gpu_id = 0;
    bool pipeline_error = false;
    GMainLoop* loop = nullptr;
    int64_t frames_processed = 0;
    int64_t total_detections = 0;
};

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
    const char* build_dir = "/app/native/build";
    std::string plugin_path = std::string(build_dir) + "/libgstnvdsretinaface.so";
    if (g_access(plugin_path.c_str(), F_OK) != 0) {
        plugin_path = "native/build/libgstnvdsretinaface.so";
    }
    GstPlugin* plugin = gst_plugin_load_file(plugin_path.c_str(), nullptr);
    if (!plugin) {
        g_printerr("Failed to load libgstnvdsretinaface.so from %s\n", plugin_path.c_str());
        return FALSE;
    }
    gst_object_unref(plugin);
    return gst_element_factory_find("nvdsretinaface") != nullptr;
}

static void set_bbox_color(NvDsObjectMeta* obj_meta, float r, float g, float b, float a) {
    obj_meta->rect_params.border_width = 2;
    obj_meta->rect_params.border_color.red = r;
    obj_meta->rect_params.border_color.green = g;
    obj_meta->rect_params.border_color.blue = b;
    obj_meta->rect_params.border_color.alpha = a;
}

static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    AppContext* ctx = (AppContext*)u_data;
    GstBuffer* buf = (GstBuffer*)info->data;
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    nvds_acquire_meta_lock(batch_meta);
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)l_frame->data;
        ctx->frames_processed++;
        int det_id = 0;
        for (NvDsMetaList* l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)l_obj->data;
            if (obj_meta->class_id != 0) continue;

            set_bbox_color(obj_meta, 0.0, 1.0, 0.0, 1.0);

            float x1 = obj_meta->rect_params.left;
            float y1 = obj_meta->rect_params.top;
            float score = obj_meta->confidence;
            gchar* text = g_strdup_printf("faceId:unknown track:UNTRACKED det:%d %.2f", det_id, score);

            obj_meta->text_params.display_text = text;
            obj_meta->text_params.x_offset = (unsigned int)std::max(0.0f, x1);
            obj_meta->text_params.y_offset = (unsigned int)std::max(0.0f, y1);
            obj_meta->text_params.font_params.font_name = g_strdup("Serif");
            obj_meta->text_params.font_params.font_size = 12;
            obj_meta->text_params.font_params.font_color = {1.0, 1.0, 1.0, 1.0};
            obj_meta->text_params.set_bg_clr = TRUE;
            obj_meta->text_params.text_bg_clr = {0.0, 0.0, 0.0, 0.7};

            ctx->total_detections++;
            det_id++;
        }
    }
    nvds_release_meta_lock(batch_meta);
    return GST_PAD_PROBE_OK;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        g_printerr("Usage: %s <input.mp4> <output.mp4> <gpu_id>\n", argv[0]);
        return -1;
    }

    AppContext ctx;
    ctx.input_path = argv[1];
    ctx.output_path = argv[2];
    ctx.gpu_id = atoi(argv[3]);

    cudaError_t cuerr = cudaSetDevice(ctx.gpu_id);
    if (cuerr != cudaSuccess) {
        g_printerr("cudaSetDevice failed: %s\n", cudaGetErrorString(cuerr));
        return -1;
    }

    gst_init(&argc, &argv);
    if (!ensure_nvdsretinaface_plugin()) {
        g_printerr("nvdsretinaface element not available\n");
        return -1;
    }

    GMainLoop* loop = g_main_loop_new(NULL, FALSE);
    ctx.loop = loop;

    GstElement* pipeline = gst_pipeline_new("mergenvision-render");
    GstElement* source = gst_element_factory_make("filesrc", "file-source");
    GstElement* qtdemux = gst_element_factory_make("qtdemux", "qt-demuxer");
    GstElement* h264parse = gst_element_factory_make("h264parse", "h264-parser");
    GstElement* decoder = gst_element_factory_make("nvv4l2decoder", "nvdec");
    GstElement* streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    GstElement* preprocess = gst_element_factory_make("nvdspreprocess", "preprocess");
    GstElement* retinaface = gst_element_factory_make("nvdsretinaface", "retinaface");
    GstElement* osd = gst_element_factory_make("nvdsosd", "osd");
    GstElement* encoder = gst_element_factory_make("nvv4l2h264enc", "encoder");
    GstElement* encparse = gst_element_factory_make("h264parse", "enc-parser");
    GstElement* muxer = gst_element_factory_make("qtmux", "muxer");
    GstElement* sink = gst_element_factory_make("filesink", "file-sink");

    if (!pipeline || !source || !qtdemux || !h264parse || !decoder || !streammux ||
        !preprocess || !retinaface || !osd || !encoder || !encparse || !muxer || !sink) {
        g_printerr("Failed to create one or more elements\n");
        return -1;
    }

    g_object_set(G_OBJECT(source), "location", ctx.input_path.c_str(), NULL);
    g_object_set(G_OBJECT(sink), "location", ctx.output_path.c_str(), NULL);

    g_object_set(G_OBJECT(streammux),
        "batch-size", 1,
        "width", 1280,
        "height", 720,
        "batched-push-timeout", 40000,
        NULL);

    std::string preprocess_config = "/app/backend/native/configs/retinaface_preprocess.txt";
    if (g_access(preprocess_config.c_str(), F_OK) != 0) {
        preprocess_config = "backend/native/configs/retinaface_preprocess.txt";
    }
    g_object_set(G_OBJECT(preprocess),
        "config-file", preprocess_config.c_str(),
        "gpu-id", ctx.gpu_id,
        NULL);

    std::string engine_path = "/app/backend/artifacts/engines/retinaface_r50_dynamic.bs1.opt64.max256.fp16.trt1014.engine";
    if (g_access(engine_path.c_str(), F_OK) != 0) {
        engine_path = "backend/artifacts/engines/retinaface_r50_dynamic.bs1.opt64.max256.fp16.trt1014.engine";
    }
    g_object_set(G_OBJECT(retinaface),
        "engine-file", engine_path.c_str(),
        "gpu-id", ctx.gpu_id,
        "conf-threshold", 0.5f,
        "nms-threshold", 0.4f,
        NULL);

    g_object_set(G_OBJECT(osd),
        "gpu-id", ctx.gpu_id,
        "process-mode", 1,  // MODE_GPU
        "display-text", TRUE,
        "display-bbox", TRUE,
        "display-clock", FALSE,
        NULL);

    g_object_set(G_OBJECT(encoder),
        "bitrate", 4000000,
        "preset-id", 2,
        "insert-sps-pps", TRUE,
        NULL);

    gst_bin_add_many(GST_BIN(pipeline), source, qtdemux, h264parse, decoder, streammux,
        preprocess, retinaface, osd, encoder, encparse, muxer, sink, NULL);

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

    if (!gst_element_link_many(streammux, preprocess, retinaface, osd, encoder, encparse, muxer, sink, NULL)) {
        g_printerr("Failed to link streammux->...->filesink\n");
        return -1;
    }

    GstPad* osd_sink = gst_element_get_static_pad(osd, "sink");
    gst_pad_add_probe(osd_sink, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe, &ctx, NULL);
    gst_object_unref(osd_sink);

    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    guint bus_watch_id = gst_bus_add_watch(bus, bus_call, &ctx);
    gst_object_unref(bus);

    g_print("Rendering annotated video on GPU %d\n  input: %s\n  output: %s\n",
        ctx.gpu_id, ctx.input_path.c_str(), ctx.output_path.c_str());

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    int exit_code = ctx.pipeline_error ? 1 : 0;
    g_print("Done. frames=%" PRId64 " detections=%" PRId64 " error=%d\n",
        ctx.frames_processed, ctx.total_detections, ctx.pipeline_error ? 1 : 0);
    return exit_code;
}
