/* GStreamer element: nvdsretinaface
 *
 * Fully GPU-resident RetinaFace R50 inference for DeepStream 9.0.
 *
 * Sink pad: video/x-raw(memory:NVMM), batch metadata.
 * Source pad: passes the same NVMM buffer downstream.
 *
 * The element consumes the device-side tensor prepared by gst-nvdspreprocess
 * (NvDsPreProcessTensorMeta), runs TensorRT inference and CUDA decode/NMS/
 * landmark decode, then attaches NvDsObjectMeta for each detected face.
 * No full detector output tensor is copied to the host.
 */
#include <cstring>

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <cuda_runtime.h>

#ifndef PACKAGE
#define PACKAGE "nvdsretinaface"
#endif

#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h"
#include "nvdsmeta.h"

#include "retinaface_engine.h"
#include "retinaface_postproc.h"

namespace mv = mergenvision;

#define GST_TYPE_NVDS_RETINAFACE (gst_nvdsretinaface_get_type())
#define GST_NVDS_RETINAFACE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDS_RETINAFACE, GstNvDsRetinaFace))
#define GST_NVDS_RETINAFACE_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVDS_RETINAFACE, GstNvDsRetinaFaceClass))

struct FaceLandmarkMeta {
    float landmarks[10];
    float score;
};

struct GstNvDsRetinaFace {
    GstBaseTransform base_transform;

    // Properties
    gchar* engine_file = nullptr;
    gfloat conf_threshold = 0.5f;
    gfloat nms_threshold = 0.4f;
    gint gpu_id = 0;

    // State
    std::unique_ptr<mv::RetinaFaceEngine> engine;
    std::unique_ptr<mv::RetinaFacePostproc> postproc;
    cudaStream_t cuda_stream = nullptr;
    gboolean started = FALSE;
};

struct GstNvDsRetinaFaceClass {
    GstBaseTransformClass base_transform_class;
};

G_DEFINE_TYPE(GstNvDsRetinaFace, gst_nvdsretinaface, GST_TYPE_BASE_TRANSFORM)

GST_ELEMENT_REGISTER_DEFINE(nvdsretinaface, "nvdsretinaface",
    GST_RANK_PRIMARY, GST_TYPE_NVDS_RETINAFACE)

enum {
    PROP_0,
    PROP_ENGINE_FILE,
    PROP_CONF_THRESHOLD,
    PROP_NMS_THRESHOLD,
    PROP_GPU_ID,
};

static void release_obj_landmark_meta(gpointer data, gpointer /*user_data*/) {
    if (!data) return;
    NvDsObjectMeta* obj_meta = static_cast<NvDsObjectMeta*>(data);
    if (obj_meta->misc_obj_info[0]) {
        delete reinterpret_cast<FaceLandmarkMeta*>(obj_meta->misc_obj_info[0]);
        obj_meta->misc_obj_info[0] = 0;
    }
}

static void gst_nvdsretinaface_set_property(GObject* object, guint prop_id,
                                            const GValue* value, GParamSpec* pspec) {
    GstNvDsRetinaFace* self = GST_NVDS_RETINAFACE(object);
    switch (prop_id) {
    case PROP_ENGINE_FILE:
        g_free(self->engine_file);
        self->engine_file = g_value_dup_string(value);
        break;
    case PROP_CONF_THRESHOLD:
        self->conf_threshold = g_value_get_float(value);
        break;
    case PROP_NMS_THRESHOLD:
        self->nms_threshold = g_value_get_float(value);
        break;
    case PROP_GPU_ID:
        self->gpu_id = g_value_get_int(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_nvdsretinaface_get_property(GObject* object, guint prop_id,
                                            GValue* value, GParamSpec* pspec) {
    GstNvDsRetinaFace* self = GST_NVDS_RETINAFACE(object);
    switch (prop_id) {
    case PROP_ENGINE_FILE:
        g_value_set_string(value, self->engine_file);
        break;
    case PROP_CONF_THRESHOLD:
        g_value_set_float(value, self->conf_threshold);
        break;
    case PROP_NMS_THRESHOLD:
        g_value_set_float(value, self->nms_threshold);
        break;
    case PROP_GPU_ID:
        g_value_set_int(value, self->gpu_id);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_nvdsretinaface_finalize(GObject* object) {
    GstNvDsRetinaFace* self = GST_NVDS_RETINAFACE(object);
    g_free(self->engine_file);
    if (self->cuda_stream) {
        cudaStreamDestroy(self->cuda_stream);
        self->cuda_stream = nullptr;
    }
    G_OBJECT_CLASS(gst_nvdsretinaface_parent_class)->finalize(object);
}

static gboolean gst_nvdsretinaface_start(GstBaseTransform* btrans) {
    GstNvDsRetinaFace* self = GST_NVDS_RETINAFACE(btrans);

    if (!self->engine_file || !self->engine_file[0]) {
        GST_ELEMENT_ERROR(self, RESOURCE, NOT_FOUND,
            ("engine-file property not set"), (NULL));
        return FALSE;
    }

    cudaError_t cuerr = cudaSetDevice(self->gpu_id);
    if (cuerr != cudaSuccess) {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
            ("cudaSetDevice failed"), ("%s", cudaGetErrorString(cuerr)));
        return FALSE;
    }
    cuerr = cudaStreamCreate(&self->cuda_stream);
    if (cuerr != cudaSuccess) {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
            ("cudaStreamCreate failed"), ("%s", cudaGetErrorString(cuerr)));
        return FALSE;
    }

    self->engine.reset(new mv::RetinaFaceEngine(self->engine_file, self->gpu_id, self->cuda_stream));
    if (!self->engine->init()) {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
            ("failed to initialize RetinaFace engine"), (NULL));
        return FALSE;
    }

    self->postproc.reset(new mv::RetinaFacePostproc(
        self->engine->inputSize(), 2000, self->gpu_id, self->cuda_stream));

    self->started = TRUE;
    return TRUE;
}

static gboolean gst_nvdsretinaface_stop(GstBaseTransform* btrans) {
    GstNvDsRetinaFace* self = GST_NVDS_RETINAFACE(btrans);
    self->engine.reset();
    self->postproc.reset();
    if (self->cuda_stream) {
        cudaStreamDestroy(self->cuda_stream);
        self->cuda_stream = nullptr;
    }
    self->started = FALSE;
    return TRUE;
}

static GstFlowReturn gst_nvdsretinaface_transform_ip(GstBaseTransform* btrans,
                                                     GstBuffer* buf) {
    GstNvDsRetinaFace* self = GST_NVDS_RETINAFACE(btrans);

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("no NvDsBatchMeta on buffer"), (NULL));
        return GST_FLOW_ERROR;
    }

    // Locate the preprocess tensor prepared by gst-nvdspreprocess.
    NvDsPreProcessTensorMeta* tensor_meta = nullptr;
    for (NvDsMetaList* l = batch_meta->batch_user_meta_list; l != NULL; l = l->next) {
        NvDsUserMeta* user_meta = (NvDsUserMeta*)l->data;
        if (user_meta->base_meta.meta_type == NVDS_PREPROCESS_BATCH_META) {
            GstNvDsPreProcessBatchMeta* preprocess_batch =
                (GstNvDsPreProcessBatchMeta*)user_meta->user_meta_data;
            if (preprocess_batch && preprocess_batch->tensor_meta) {
                tensor_meta = preprocess_batch->tensor_meta;
                break;
            }
        }
    }
    if (!tensor_meta || !tensor_meta->raw_tensor_buffer) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("no preprocess tensor meta found"), (NULL));
        return GST_FLOW_ERROR;
    }

    // sanity: pointer should be device memory
    cudaPointerAttributes attr{};
    cudaError_t ptr_err = cudaPointerGetAttributes(&attr, tensor_meta->raw_tensor_buffer);
    if (ptr_err != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("preprocess tensor is not device memory"), (NULL));
        return GST_FLOW_ERROR;
    }

    const float* d_loc = nullptr;
    const float* d_conf = nullptr;
    const float* d_landms = nullptr;
    int num_anchors = 0;
    if (!self->engine->infer(tensor_meta->raw_tensor_buffer, 1,
                             &d_loc, &d_conf, &d_landms, &num_anchors)) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("RetinaFace inference failed"), (NULL));
        return GST_FLOW_ERROR;
    }

    // For batch-size 1 there is one frame meta.
    NvDsFrameMeta* frame_meta = nullptr;
    if (batch_meta->frame_meta_list) {
        frame_meta = (NvDsFrameMeta*)batch_meta->frame_meta_list->data;
    }
    if (!frame_meta) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("no frame meta"), (NULL));
        return GST_FLOW_ERROR;
    }

    auto detections = self->postproc->processFrame(
        d_loc, d_conf, d_landms, num_anchors,
        frame_meta->source_frame_width, frame_meta->source_frame_height,
        self->conf_threshold, self->nms_threshold);

    nvds_acquire_meta_lock(batch_meta);
    for (const auto& det : detections) {
        NvDsObjectMeta* obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
        // Do not memset the whole struct: the pool initializes internal links
        // and base_meta. Set only application-visible fields.
        obj_meta->unique_component_id = 0;  // match tracker's default primary detector
        obj_meta->object_id = UNTRACKED_OBJECT_ID;  // tracker will assign its own ID
        obj_meta->class_id = 0;
        obj_meta->confidence = det.score;
        obj_meta->rect_params.left = det.x1;
        obj_meta->rect_params.top = det.y1;
        obj_meta->rect_params.width = det.x2 - det.x1;
        obj_meta->rect_params.height = det.y2 - det.y1;
        obj_meta->text_params.display_text = nullptr;

        FaceLandmarkMeta* lm = new FaceLandmarkMeta();
        std::memcpy(lm->landmarks, det.landmarks, sizeof(lm->landmarks));
        lm->score = det.score;

        obj_meta->misc_obj_info[0] = reinterpret_cast<gint64>(lm);
        obj_meta->base_meta.release_func = release_obj_landmark_meta;
        obj_meta->base_meta.copy_func = NULL;

        nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
    }
    nvds_release_meta_lock(batch_meta);

    return GST_FLOW_OK;
}

static void gst_nvdsretinaface_class_init(GstNvDsRetinaFaceClass* klass) {
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass* btrans_class = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class->set_property = gst_nvdsretinaface_set_property;
    gobject_class->get_property = gst_nvdsretinaface_get_property;
    gobject_class->finalize = gst_nvdsretinaface_finalize;

    btrans_class->start = gst_nvdsretinaface_start;
    btrans_class->stop = gst_nvdsretinaface_stop;
    btrans_class->transform_ip = gst_nvdsretinaface_transform_ip;

    gst_element_class_set_static_metadata(element_class,
        "NvDsRetinaFace", "Filter/Video",
        "GPU-only RetinaFace detection",
        "MergenVision");

    // Sink and source templates accept NVMM.
    static GstStaticPadTemplate sink_template =
        GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
            GST_STATIC_CAPS("video/x-raw(memory:NVMM)"));
    static GstStaticPadTemplate src_template =
        GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS,
            GST_STATIC_CAPS("video/x-raw(memory:NVMM)"));
    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    g_object_class_install_property(gobject_class, PROP_ENGINE_FILE,
        g_param_spec_string("engine-file", "Engine file",
            "Path to TensorRT engine", "",
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_CONF_THRESHOLD,
        g_param_spec_float("conf-threshold", "Confidence threshold",
            "Minimum detection confidence", 0.0f, 1.0f, 0.5f,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_NMS_THRESHOLD,
        g_param_spec_float("nms-threshold", "NMS IoU threshold",
            "IoU threshold for CUDA NMS", 0.0f, 1.0f, 0.4f,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_GPU_ID,
        g_param_spec_int("gpu-id", "GPU ID",
            "GPU device ID", 0, G_MAXINT, 0,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
}

static void gst_nvdsretinaface_init(GstNvDsRetinaFace* /*self*/) {
}

extern "C" {

static gboolean plugin_init(GstPlugin* plugin) {
    return GST_ELEMENT_REGISTER(nvdsretinaface, plugin);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsretinaface,
    "MergenVision GPU-only RetinaFace plugin",
    plugin_init,
    "1.0.0",
    "Proprietary",
    "nvdsretinaface",
    "mergenvision"
)

} // extern "C"
