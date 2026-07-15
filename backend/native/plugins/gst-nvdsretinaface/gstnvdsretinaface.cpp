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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>

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

#define MERGEN_COMPONENT_ID 1
#define FACE_LANDMARK_META_NAME "mv-face-landmark"

struct FaceLandmarkMeta {
    float landmarks[10];
    float score;
};

static NvDsMetaType g_face_landmark_meta_type = NVDS_USER_META;

namespace mv = mergenvision;

#define GST_TYPE_NVDS_RETINAFACE (gst_nvdsretinaface_get_type())
#define GST_NVDS_RETINAFACE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDS_RETINAFACE, GstNvDsRetinaFace))
#define GST_NVDS_RETINAFACE_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVDS_RETINAFACE, GstNvDsRetinaFaceClass))

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

    // Diagnostic (off by default): dump preprocessed input tensor per frame.
    static constexpr size_t PREPROC_ELM = 1 * 3 * 640 * 640;
    float* h_preproc_dump = nullptr;
    bool dump_preproc = false;
    std::string preproc_dump_dir;
};

struct GstNvDsRetinaFaceClass {
    GstBaseTransformClass base_transform_class;
};

G_DEFINE_TYPE(GstNvDsRetinaFace, gst_nvdsretinaface, GST_TYPE_BASE_TRANSFORM)

GST_ELEMENT_REGISTER_DEFINE(nvdsretinaface, "nvdsretinaface",
    GST_RANK_PRIMARY, GST_TYPE_NVDS_RETINAFACE)

static gpointer copy_landmark_user_meta(gpointer data, gpointer /*user_data*/) {
    if (!data) return nullptr;
    NvDsUserMeta* src = static_cast<NvDsUserMeta*>(data);
    if (!src->user_meta_data) return nullptr;
    FaceLandmarkMeta* src_lm = static_cast<FaceLandmarkMeta*>(src->user_meta_data);
    return new FaceLandmarkMeta(*src_lm);
}

static void release_landmark_user_meta(gpointer data, gpointer /*user_data*/) {
    if (!data) return;
    NvDsUserMeta* user_meta = static_cast<NvDsUserMeta*>(data);
    if (user_meta->user_meta_data) {
        delete static_cast<FaceLandmarkMeta*>(user_meta->user_meta_data);
        user_meta->user_meta_data = nullptr;
    }
}

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

enum {
    PROP_0,
    PROP_ENGINE_FILE,
    PROP_CONF_THRESHOLD,
    PROP_NMS_THRESHOLD,
    PROP_GPU_ID,
};

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
    if (self->h_preproc_dump) {
        cudaFreeHost(self->h_preproc_dump);
        self->h_preproc_dump = nullptr;
    }
    G_OBJECT_CLASS(gst_nvdsretinaface_parent_class)->finalize(object);
}

static gboolean gst_nvdsretinaface_start(GstBaseTransform* btrans) {
    GstNvDsRetinaFace* self = GST_NVDS_RETINAFACE(btrans);
    ensure_face_landmark_meta_type();

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

    if (const char* dump_dir = getenv("MV_DUMP_PREPROC_TENSOR")) {
        cudaError_t dump_err = cudaHostAlloc(
            &self->h_preproc_dump, self->PREPROC_ELM * sizeof(float), cudaHostAllocDefault);
        if (dump_err != cudaSuccess) {
            GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
                ("failed to allocate pinned dump buffer"), (NULL));
            return FALSE;
        }
        self->dump_preproc = true;
        self->preproc_dump_dir = dump_dir;
        g_mkdir_with_parents(dump_dir, 0755);
    }

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

// Test-only helper. Writes a JSON sidecar describing the dumped tensor.
// Only called when MV_DUMP_PREPROC_TENSOR is set.
static void write_preproc_sidecar(GstBaseTransform* btrans,
                                  NvDsFrameMeta* frame_meta,
                                  NvDsPreProcessTensorMeta* tensor_meta,
                                  const std::string& dump_dir,
                                  int frame_num) {
    std::string path = dump_dir + "/preproc_" + std::to_string(frame_num) + ".json";
    std::ofstream out(path);
    if (!out) return;

    const gchar* caps_str = "unknown";
    GstCaps* caps = gst_pad_get_current_caps(GST_BASE_TRANSFORM_SINK_PAD(btrans));
    if (caps) {
        GstStructure* s = gst_caps_get_structure(caps, 0);
        if (s) caps_str = gst_structure_to_string(s);
    }

    out << "{\n";
    out << "  \"frame_num\": " << frame_num << ",\n";
    out << "  \"buf_pts\": " << (frame_meta ? frame_meta->buf_pts : -1) << ",\n";
    out << "  \"source_id\": " << (frame_meta ? static_cast<int>(frame_meta->source_id) : -1) << ",\n";
    out << "  \"batch_id\": " << (frame_meta ? static_cast<int>(frame_meta->batch_id) : -1) << ",\n";
    out << "  \"original_width\": " << (frame_meta ? frame_meta->source_frame_width : 0) << ",\n";
    out << "  \"original_height\": " << (frame_meta ? frame_meta->source_frame_height : 0) << ",\n";
    out << "  \"tensor_shape\": [1, 3, 640, 640],\n";
    out << "  \"tensor_dtype\": \"float32\",\n";
    out << "  \"tensor_layout\": \"NCHW\",\n";
    out << "  \"tensor_size_bytes\": " << (1 * 3 * 640 * 640 * sizeof(float)) << ",\n";
    out << "  \"effective_conf_threshold\": " << GST_NVDS_RETINAFACE(btrans)->conf_threshold << ",\n";
    out << "  \"source_caps\": \"" << caps_str << "\"\n";
    out << "}\n";

    if (caps) gst_caps_unref(caps);
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

    // Determine actual batch size from frame metadata and map each frame by batch_id.
    std::vector<NvDsFrameMeta*> frames_by_batch;
    int actual_batch = 0;
    nvds_acquire_meta_lock(batch_meta);
    for (NvDsMetaList* l = batch_meta->frame_meta_list; l != NULL; l = l->next) {
        NvDsFrameMeta* fm = (NvDsFrameMeta*)l->data;
        if (fm->batch_id < 0) {
            nvds_release_meta_lock(batch_meta);
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                ("invalid batch_id on frame meta"), (NULL));
            return GST_FLOW_ERROR;
        }
        if (fm->batch_id >= actual_batch) actual_batch = fm->batch_id + 1;
        if (static_cast<size_t>(fm->batch_id) >= frames_by_batch.size()) {
            frames_by_batch.resize(fm->batch_id + 1, nullptr);
        }
        if (frames_by_batch[fm->batch_id] != nullptr) {
            nvds_release_meta_lock(batch_meta);
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                ("duplicate batch_id in batch"), (NULL));
            return GST_FLOW_ERROR;
        }
        frames_by_batch[fm->batch_id] = fm;
    }
    nvds_release_meta_lock(batch_meta);

    if (actual_batch <= 0 || frames_by_batch.size() != static_cast<size_t>(actual_batch)) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("missing batch_id slot in frame meta list"), (NULL));
        return GST_FLOW_ERROR;
    }

    if (actual_batch > self->engine->maxBatchSize()) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("actual batch size exceeds engine maximum"), ("actual=%d max=%d",
             actual_batch, self->engine->maxBatchSize()));
        return GST_FLOW_ERROR;
    }

    if (!tensor_meta->tensor_shape.empty() &&
        tensor_meta->tensor_shape[0] != actual_batch) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("preprocess tensor batch dimension does not match frame meta count"),
            ("tensor_shape[0]=%d actual_batch=%d", tensor_meta->tensor_shape[0], actual_batch));
        return GST_FLOW_ERROR;
    }

    // Optional diagnostic: dump the preprocessed input tensor for the first frame only.
    if (self->dump_preproc && self->h_preproc_dump) {
        cudaError_t dump_err = cudaMemcpyAsync(
            self->h_preproc_dump,
            tensor_meta->raw_tensor_buffer,
            self->PREPROC_ELM * sizeof(float),
            cudaMemcpyDeviceToHost,
            self->cuda_stream);
        if (dump_err == cudaSuccess && !frames_by_batch.empty() && frames_by_batch[0]) {
            cudaStreamSynchronize(self->cuda_stream);
            int dump_frame = static_cast<int>(frames_by_batch[0]->frame_num);
            std::string path = self->preproc_dump_dir + "/preproc_" +
                std::to_string(dump_frame) + ".bin";
            FILE* fp = std::fopen(path.c_str(), "wb");
            if (fp) {
                std::fwrite(self->h_preproc_dump, sizeof(float), self->PREPROC_ELM, fp);
                std::fclose(fp);
            }
            write_preproc_sidecar(btrans, frames_by_batch[0], tensor_meta,
                                  self->preproc_dump_dir, dump_frame);
        }
    }

    const float* d_loc = nullptr;
    const float* d_conf = nullptr;
    const float* d_landms = nullptr;
    int num_anchors = 0;
    if (!self->engine->infer(tensor_meta->raw_tensor_buffer, actual_batch,
                             &d_loc, &d_conf, &d_landms, &num_anchors)) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("RetinaFace inference failed"), (NULL));
        return GST_FLOW_ERROR;
    }

    std::vector<std::pair<int, int>> original_dims;
    original_dims.reserve(actual_batch);
    for (int b = 0; b < actual_batch; ++b) {
        original_dims.emplace_back(
            frames_by_batch[b]->source_frame_width,
            frames_by_batch[b]->source_frame_height);
    }

    auto per_frame_detections = self->postproc->processBatch(
        d_loc, d_conf, d_landms, num_anchors, actual_batch,
        original_dims, self->conf_threshold, self->nms_threshold);

    nvds_acquire_meta_lock(batch_meta);
    for (int b = 0; b < actual_batch; ++b) {
        NvDsFrameMeta* fm = frames_by_batch[b];
        for (const auto& det : per_frame_detections[b]) {
            NvDsObjectMeta* obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
            obj_meta->unique_component_id = MERGEN_COMPONENT_ID;
            obj_meta->object_id = UNTRACKED_OBJECT_ID;
            obj_meta->class_id = 0;
            obj_meta->confidence = det.score;
            obj_meta->rect_params.left = det.x1;
            obj_meta->rect_params.top = det.y1;
            obj_meta->rect_params.width = det.x2 - det.x1;
            obj_meta->rect_params.height = det.y2 - det.y1;
            obj_meta->text_params.display_text = nullptr;

            NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
            if (user_meta) {
                FaceLandmarkMeta* lm = new FaceLandmarkMeta();
                std::memcpy(lm->landmarks, det.landmarks, sizeof(lm->landmarks));
                lm->score = det.score;
                user_meta->user_meta_data = lm;
                user_meta->base_meta.meta_type = g_face_landmark_meta_type;
                user_meta->base_meta.copy_func = copy_landmark_user_meta;
                user_meta->base_meta.release_func = release_landmark_user_meta;
                nvds_add_user_meta_to_obj(obj_meta, user_meta);
            }

            nvds_add_obj_meta_to_frame(fm, obj_meta, NULL);
        }
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
