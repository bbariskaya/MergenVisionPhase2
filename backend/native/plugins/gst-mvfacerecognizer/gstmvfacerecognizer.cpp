/* GStreamer element: mvfacerecognizer
 *
 * GPU face recognizer for Sprint 05. Receives batched RGBA/NVMM buffers with
 * NvDsBatchMeta and NvDsObjectMeta carrying face landmarks, performs GPU
 * five-point alignment, TensorRT GlintR100 embedding, GPU L2 normalization,
 * deterministic gallery matching, and attaches structured recognition metadata
 * (MvFaceRecognitionMeta) for downstream nvdsosd and JSONL emission.
 *
 * The element does not copy full frames to CPU and does not retain pointers
 * into reusable engine buffers after processing a buffer.
 */
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <cuda_runtime.h>

#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

#include "mergenvision_kernels.h"
#include "glintr100_engine.h"
#include "gallery.h"
#include "mv_face_recognition_meta.h"

#include <openssl/sha.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef PACKAGE
#define PACKAGE "mvfacerecognizer"
#endif
#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION "1.0.0"
#endif

#define GST_TYPE_MV_FACE_RECOGNIZER (gst_mv_face_recognizer_get_type())
#define GST_MV_FACE_RECOGNIZER(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_MV_FACE_RECOGNIZER, GstMvFaceRecognizer))

#define FACE_LANDMARK_META_NAME "mv-face-landmark"

struct FaceLandmarkMeta {
    float landmarks[10];
    float score;
};

struct GstMvFaceRecognizer;

struct RecognizerImpl {
    cudaStream_t cuda_stream = nullptr;

    std::unique_ptr<mergenvision::GlintR100Engine> engine;
    std::unique_ptr<mergenvision::Gallery> gallery;

    int max_faces = 0;
    int max_batch_surfaces = 0;

    // Face description buffers (device).
    float* d_landmarks = nullptr;      // [max_faces, 10]
    float* d_matrices = nullptr;       // [max_faces, 6]
    int* d_surface_indices = nullptr;  // [max_faces]
    int* d_pitches = nullptr;          // [max_faces]
    int* d_widths = nullptr;           // [max_faces]
    int* d_heights = nullptr;          // [max_faces]
    float* d_aligned = nullptr;        // [max_faces, 3, 112, 112]
    float* d_normalized = nullptr;     // [max_faces, 512]
    int* d_status = nullptr;           // shared status flag for CUDA kernels
    uint8_t** d_surface_ptrs = nullptr; // [max_batch_surfaces]

    // Host scratch.
    float* h_embeddings = nullptr;     // pinned [max_faces, 512]
    float* h_matrices = nullptr;       // pinned [max_faces, 6]

    std::string engine_sha256;
    std::string gallery_sha256;
    std::string contract_sha256;

    ~RecognizerImpl() { release(); }

    void release() {
        if (d_landmarks) { cudaFree(d_landmarks); d_landmarks = nullptr; }
        if (d_matrices) { cudaFree(d_matrices); d_matrices = nullptr; }
        if (d_surface_indices) { cudaFree(d_surface_indices); d_surface_indices = nullptr; }
        if (d_pitches) { cudaFree(d_pitches); d_pitches = nullptr; }
        if (d_widths) { cudaFree(d_widths); d_widths = nullptr; }
        if (d_heights) { cudaFree(d_heights); d_heights = nullptr; }
        if (d_aligned) { cudaFree(d_aligned); d_aligned = nullptr; }
        if (d_normalized) { cudaFree(d_normalized); d_normalized = nullptr; }
        if (d_status) { cudaFree(d_status); d_status = nullptr; }
        if (d_surface_ptrs) { cudaFree(d_surface_ptrs); d_surface_ptrs = nullptr; }
        if (h_embeddings) { cudaFreeHost(h_embeddings); h_embeddings = nullptr; }
        if (h_matrices) { cudaFreeHost(h_matrices); h_matrices = nullptr; }
        if (cuda_stream) { cudaStreamDestroy(cuda_stream); cuda_stream = nullptr; }
        engine.reset();
        gallery.reset();
        max_faces = 0;
        max_batch_surfaces = 0;
    }
};

struct GstMvFaceRecognizer {
    GstBaseTransform base_transform;

    gchar* engine_config = nullptr;
    gchar* gallery_file = nullptr;
    gfloat threshold = 0.5f;
    gfloat margin = 0.2f;
    gint gpu_id = 0;

    RecognizerImpl* impl = nullptr;
    static NvDsMetaType landmark_meta_type;
};

NvDsMetaType GstMvFaceRecognizer::landmark_meta_type = NVDS_USER_META;

struct GstMvFaceRecognizerClass {
    GstBaseTransformClass base_transform_class;
};

G_DEFINE_TYPE(GstMvFaceRecognizer, gst_mv_face_recognizer, GST_TYPE_BASE_TRANSFORM)

GST_ELEMENT_REGISTER_DEFINE(mvfacerecognizer, "mvfacerecognizer",
    GST_RANK_PRIMARY, GST_TYPE_MV_FACE_RECOGNIZER)

enum {
    PROP_0,
    PROP_ENGINE_FILE,
    PROP_GALLERY_FILE,
    PROP_THRESHOLD,
    PROP_MARGIN,
    PROP_GPU_ID,
};

namespace {

struct FaceContext {
    NvDsFrameMeta* frame = nullptr;
    NvDsObjectMeta* obj = nullptr;
    int det_id = 0;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float score = 0.0f;
    float landmarks[10] = {0};
    int batch_id = 0;
    bool sane = false;
};

static std::string file_sha256(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(f)),
                                     std::istreambuf_iterator<char>());
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(data.data(), data.size(), hash);
    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return oss.str();
}

static const FaceLandmarkMeta* find_landmark_meta(NvDsObjectMeta* obj_meta, NvDsMetaType type) {
    for (NvDsMetaList* l = obj_meta->obj_user_meta_list; l != NULL; l = l->next) {
        NvDsUserMeta* um = (NvDsUserMeta*)l->data;
        if (um->base_meta.meta_type == type && um->user_meta_data) {
            return reinterpret_cast<const FaceLandmarkMeta*>(um->user_meta_data);
        }
    }
    return nullptr;
}

} // namespace

static void ensure_landmark_meta_type() {
    static gboolean initialized = FALSE;
    if (!initialized) {
        GstMvFaceRecognizer::landmark_meta_type =
            nvds_get_user_meta_type((gchar*)FACE_LANDMARK_META_NAME);
        if (GstMvFaceRecognizer::landmark_meta_type == 0) {
            GstMvFaceRecognizer::landmark_meta_type = NVDS_USER_META;
        }
        initialized = TRUE;
    }
}

static void gst_mv_face_recognizer_set_property(GObject* object, guint prop_id,
                                                const GValue* value, GParamSpec* pspec) {
    GstMvFaceRecognizer* self = GST_MV_FACE_RECOGNIZER(object);
    switch (prop_id) {
    case PROP_ENGINE_FILE:
        g_free(self->engine_config);
        self->engine_config = g_value_dup_string(value);
        break;
    case PROP_GALLERY_FILE:
        g_free(self->gallery_file);
        self->gallery_file = g_value_dup_string(value);
        break;
    case PROP_THRESHOLD:
        self->threshold = g_value_get_float(value);
        break;
    case PROP_MARGIN:
        self->margin = g_value_get_float(value);
        break;
    case PROP_GPU_ID:
        self->gpu_id = g_value_get_int(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_mv_face_recognizer_get_property(GObject* object, guint prop_id,
                                                GValue* value, GParamSpec* pspec) {
    GstMvFaceRecognizer* self = GST_MV_FACE_RECOGNIZER(object);
    switch (prop_id) {
    case PROP_ENGINE_FILE:
        g_value_set_string(value, self->engine_config);
        break;
    case PROP_GALLERY_FILE:
        g_value_set_string(value, self->gallery_file);
        break;
    case PROP_THRESHOLD:
        g_value_set_float(value, self->threshold);
        break;
    case PROP_MARGIN:
        g_value_set_float(value, self->margin);
        break;
    case PROP_GPU_ID:
        g_value_set_int(value, self->gpu_id);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_mv_face_recognizer_finalize(GObject* object) {
    GstMvFaceRecognizer* self = GST_MV_FACE_RECOGNIZER(object);
    g_free(self->engine_config);
    g_free(self->gallery_file);
    delete self->impl;
    self->impl = nullptr;
    G_OBJECT_CLASS(gst_mv_face_recognizer_parent_class)->finalize(object);
}

static GstCaps* gst_mv_face_recognizer_transform_caps(GstBaseTransform* btrans,
                                                       GstPadDirection direction,
                                                       GstCaps* caps,
                                                       GstCaps* filter) {
    (void)btrans; (void)direction; (void)filter;
    return gst_caps_ref(caps);
}

static gboolean gst_mv_face_recognizer_start(GstBaseTransform* btrans) {
    GstMvFaceRecognizer* self = GST_MV_FACE_RECOGNIZER(btrans);
    ensure_landmark_meta_type();
    mv_face_recognition_meta_type();

    if (!self->impl) {
        self->impl = new RecognizerImpl();
    }

    if (!self->engine_config || !self->engine_config[0]) {
        GST_ELEMENT_ERROR(self, RESOURCE, NOT_FOUND,
            ("engine-file property not set"), (NULL));
        return FALSE;
    }
    if (!self->gallery_file || !self->gallery_file[0]) {
        GST_ELEMENT_ERROR(self, RESOURCE, NOT_FOUND,
            ("gallery-file property not set"), (NULL));
        return FALSE;
    }

    cudaError_t cuerr = cudaSetDevice(self->gpu_id);
    if (cuerr != cudaSuccess) {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
            ("cudaSetDevice failed"), ("%s", cudaGetErrorString(cuerr)));
        return FALSE;
    }

    cuerr = cudaStreamCreate(&self->impl->cuda_stream);
    if (cuerr != cudaSuccess) {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
            ("cudaStreamCreate failed"), ("%s", cudaGetErrorString(cuerr)));
        return FALSE;
    }

    self->impl->engine.reset(new mergenvision::GlintR100Engine());
    std::string error;
    if (!self->impl->engine->load(self->gpu_id, self->engine_config, &error)) {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
            ("failed to load GlintR100 engine"), ("%s", error.c_str()));
        return FALSE;
    }

    self->impl->gallery.reset(new mergenvision::Gallery());
    if (!self->impl->gallery->load(self->gallery_file, &error)) {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
            ("failed to load gallery"), ("%s", error.c_str()));
        return FALSE;
    }

    self->impl->contract_sha256 = file_sha256(self->engine_config);
    self->impl->engine_sha256 = file_sha256(self->impl->engine->engine_path());
    self->impl->gallery_sha256 = self->impl->gallery->sha256();

    const int max_faces = self->impl->engine->max_batch();
    const int max_batch_surfaces = max_faces; // upper bound on frame surfaces

    auto alloc = [&](void** p, size_t bytes, const char* what) -> bool {
        cudaError_t err = cudaMalloc(p, bytes);
        if (err != cudaSuccess) {
            GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
                ("failed to allocate GPU memory"), ("%s: %s", what, cudaGetErrorString(err)));
            return FALSE;
        }
        return TRUE;
    };

    if (!alloc(reinterpret_cast<void**>(&self->impl->d_landmarks),
               max_faces * 10 * sizeof(float), "d_landmarks")) return FALSE;
    if (!alloc(reinterpret_cast<void**>(&self->impl->d_matrices),
               max_faces * 6 * sizeof(float), "d_matrices")) return FALSE;
    if (!alloc(reinterpret_cast<void**>(&self->impl->d_surface_indices),
               max_faces * sizeof(int), "d_surface_indices")) return FALSE;
    if (!alloc(reinterpret_cast<void**>(&self->impl->d_pitches),
               max_faces * sizeof(int), "d_pitches")) return FALSE;
    if (!alloc(reinterpret_cast<void**>(&self->impl->d_widths),
               max_faces * sizeof(int), "d_widths")) return FALSE;
    if (!alloc(reinterpret_cast<void**>(&self->impl->d_heights),
               max_faces * sizeof(int), "d_heights")) return FALSE;
    if (!alloc(reinterpret_cast<void**>(&self->impl->d_aligned),
               max_faces * 3 * 112 * 112 * sizeof(float), "d_aligned")) return FALSE;
    if (!alloc(reinterpret_cast<void**>(&self->impl->d_normalized),
               max_faces * 512 * sizeof(float), "d_normalized")) return FALSE;
    if (!alloc(reinterpret_cast<void**>(&self->impl->d_status),
               sizeof(int), "d_status")) return FALSE;
    if (!alloc(reinterpret_cast<void**>(&self->impl->d_surface_ptrs),
               max_batch_surfaces * sizeof(uint8_t*), "d_surface_ptrs")) return FALSE;

    cudaError_t err = cudaHostAlloc(&self->impl->h_embeddings,
                                    max_faces * 512 * sizeof(float),
                                    cudaHostAllocDefault);
    if (err != cudaSuccess) {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
            ("failed to allocate pinned host embeddings"),
            ("%s", cudaGetErrorString(err)));
        return FALSE;
    }
    err = cudaHostAlloc(&self->impl->h_matrices,
                        max_faces * 6 * sizeof(float),
                        cudaHostAllocDefault);
    if (err != cudaSuccess) {
        GST_ELEMENT_ERROR(self, RESOURCE, FAILED,
            ("failed to allocate pinned host matrices"),
            ("%s", cudaGetErrorString(err)));
        return FALSE;
    }

    self->impl->max_faces = max_faces;
    self->impl->max_batch_surfaces = max_batch_surfaces;
    return TRUE;
}

static gboolean gst_mv_face_recognizer_stop(GstBaseTransform* btrans) {
    GstMvFaceRecognizer* self = GST_MV_FACE_RECOGNIZER(btrans);
    if (self->impl) {
        self->impl->release();
    }
    return TRUE;
}

static GstFlowReturn gst_mv_face_recognizer_transform_ip(GstBaseTransform* btrans,
                                                          GstBuffer* buf) {
    GstMvFaceRecognizer* self = GST_MV_FACE_RECOGNIZER(btrans);
    if (!self->impl || !self->impl->engine || !self->impl->gallery) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("recognizer not initialized"), (NULL));
        return GST_FLOW_ERROR;
    }

    auto& impl = *self->impl;
    cudaStream_t stream = impl.cuda_stream;

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("no NvDsBatchMeta on buffer"), (NULL));
        return GST_FLOW_ERROR;
    }

    GstMapInfo inmap = {0};
    if (!gst_buffer_map(buf, &inmap, GST_MAP_READ)) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("failed to map input buffer"), (NULL));
        return GST_FLOW_ERROR;
    }
    NvBufSurface* surface = reinterpret_cast<NvBufSurface*>(inmap.data);
    if (!surface) {
        gst_buffer_unmap(buf, &inmap);
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("no NvBufSurface in buffer"), (NULL));
        return GST_FLOW_ERROR;
    }

    std::vector<NvDsFrameMeta*> frames_by_batch;
    int actual_batch = 0;
    std::vector<FaceContext> faces;

    nvds_acquire_meta_lock(batch_meta);
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)l_frame->data;
        if (frame_meta->batch_id < 0) {
            nvds_release_meta_lock(batch_meta);
            gst_buffer_unmap(buf, &inmap);
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                ("invalid batch_id on frame meta"), (NULL));
            return GST_FLOW_ERROR;
        }
        if (frame_meta->batch_id >= actual_batch) actual_batch = frame_meta->batch_id + 1;
        if (static_cast<size_t>(frame_meta->batch_id) >= frames_by_batch.size()) {
            frames_by_batch.resize(frame_meta->batch_id + 1, nullptr);
        }
        if (frames_by_batch[frame_meta->batch_id] != nullptr) {
            nvds_release_meta_lock(batch_meta);
            gst_buffer_unmap(buf, &inmap);
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                ("duplicate batch_id in batch"), (NULL));
            return GST_FLOW_ERROR;
        }
        frames_by_batch[frame_meta->batch_id] = frame_meta;
    }

    for (NvDsFrameMeta* frame_meta : frames_by_batch) {
        if (!frame_meta) continue;
        int det_id = 0;
        for (NvDsMetaList* l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next) {
            NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)l_obj->data;
            if (obj_meta->class_id != 0) continue;

            const FaceLandmarkMeta* lm = find_landmark_meta(obj_meta,
                GstMvFaceRecognizer::landmark_meta_type);
            if (!lm) continue;

            FaceContext fc;
            fc.frame = frame_meta;
            fc.obj = obj_meta;
            fc.det_id = det_id++;
            fc.x1 = obj_meta->rect_params.left;
            fc.y1 = obj_meta->rect_params.top;
            fc.x2 = fc.x1 + obj_meta->rect_params.width;
            fc.y2 = fc.y1 + obj_meta->rect_params.height;
            fc.score = obj_meta->confidence;
            std::memcpy(fc.landmarks, lm->landmarks, sizeof(fc.landmarks));
            fc.batch_id = frame_meta->batch_id;

            bool sane = fc.score >= 0.0f && fc.score <= 1.0f
                && fc.x1 < fc.x2 && fc.y1 < fc.y2
                && fc.x1 >= 0.0f && fc.y1 >= 0.0f
                && fc.x2 <= frame_meta->source_frame_width
                && fc.y2 <= frame_meta->source_frame_height;
            for (int k = 0; sane && k < 5; ++k) {
                float lx = fc.landmarks[2 * k];
                float ly = fc.landmarks[2 * k + 1];
                sane = std::isfinite(lx) && std::isfinite(ly)
                    && lx >= 0.0f && ly >= 0.0f
                    && lx <= frame_meta->source_frame_width
                    && ly <= frame_meta->source_frame_height;
            }
            fc.sane = sane;
            faces.push_back(fc);
        }
    }
    nvds_release_meta_lock(batch_meta);

    if (actual_batch > impl.max_batch_surfaces) {
        gst_buffer_unmap(buf, &inmap);
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("batch size exceeds recognizer surface allocation"),
            ("actual_batch=%d max=%d", actual_batch, impl.max_batch_surfaces));
        return GST_FLOW_ERROR;
    }

    int n_faces = static_cast<int>(faces.size());
    if (n_faces == 0) {
        gst_buffer_unmap(buf, &inmap);
        return GST_FLOW_OK;
    }

    if (n_faces > impl.max_faces) {
        gst_buffer_unmap(buf, &inmap);
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("face count exceeds engine maximum"),
            ("faces=%d max=%d", n_faces, impl.max_faces));
        return GST_FLOW_ERROR;
    }

    // Host staging for per-face inputs.
    std::vector<float> h_landmarks(static_cast<size_t>(n_faces) * 10);
    std::vector<int> h_surface_idx(n_faces);
    std::vector<int> h_pitches(n_faces);
    std::vector<int> h_widths(n_faces);
    std::vector<int> h_heights(n_faces);
    std::vector<uint8_t*> h_surface_ptrs(actual_batch);

    for (int i = 0; i < n_faces; ++i) {
        std::memcpy(&h_landmarks[i * 10], faces[i].landmarks, sizeof(float) * 10);
        int bid = faces[i].batch_id;
        h_surface_idx[i] = bid;
        h_pitches[i] = surface->surfaceList[bid].pitch;
        h_widths[i] = surface->surfaceList[bid].width;
        h_heights[i] = surface->surfaceList[bid].height;
    }
    for (int i = 0; i < actual_batch; ++i) {
        h_surface_ptrs[i] = reinterpret_cast<uint8_t*>(surface->surfaceList[i].dataPtr);
    }

    // Reset decomposed status flag then copy inputs to device.
    cudaError_t err = cudaMemsetAsync(impl.d_status, 0, sizeof(int), stream);
    if (err != cudaSuccess) { gst_buffer_unmap(buf, &inmap); return GST_FLOW_ERROR; }

    err = cudaMemcpyAsync(impl.d_landmarks, h_landmarks.data(),
                          h_landmarks.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { gst_buffer_unmap(buf, &inmap); return GST_FLOW_ERROR; }
    err = cudaMemcpyAsync(impl.d_surface_indices, h_surface_idx.data(),
                          h_surface_idx.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { gst_buffer_unmap(buf, &inmap); return GST_FLOW_ERROR; }
    err = cudaMemcpyAsync(impl.d_pitches, h_pitches.data(),
                          h_pitches.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { gst_buffer_unmap(buf, &inmap); return GST_FLOW_ERROR; }
    err = cudaMemcpyAsync(impl.d_widths, h_widths.data(),
                          h_widths.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { gst_buffer_unmap(buf, &inmap); return GST_FLOW_ERROR; }
    err = cudaMemcpyAsync(impl.d_heights, h_heights.data(),
                          h_heights.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { gst_buffer_unmap(buf, &inmap); return GST_FLOW_ERROR; }
    err = cudaMemcpyAsync(impl.d_surface_ptrs, h_surface_ptrs.data(),
                          actual_batch * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { gst_buffer_unmap(buf, &inmap); return GST_FLOW_ERROR; }

    int cu = mergenvision_similarity_transform(
        impl.d_landmarks, impl.d_matrices, n_faces, 112 * 112, impl.d_status, stream);
    if (cu != 0) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("similarity_transform failed"), ("cuda error %d", cu));
        gst_buffer_unmap(buf, &inmap);
        return GST_FLOW_ERROR;
    }

    cu = mergenvision_warp_align_rgba_pitch(
        impl.d_surface_ptrs, impl.d_surface_indices, impl.d_pitches,
        impl.d_widths, impl.d_heights, impl.d_matrices, n_faces,
        impl.d_aligned, stream);
    if (cu != 0) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("warp_align_rgba_pitch failed"), ("cuda error %d", cu));
        gst_buffer_unmap(buf, &inmap);
        return GST_FLOW_ERROR;
    }

    // Chunked inference: copy the aligned chunk into the engine input buffer, run,
    // then L2-normalize into the per-face normalized buffer.
    const int max_batch = impl.engine->max_batch();
    for (int offset = 0; offset < n_faces; offset += max_batch) {
        int chunk = std::min(max_batch, n_faces - offset);
        size_t input_chunk_bytes = static_cast<size_t>(chunk) * 3 * 112 * 112 * sizeof(float);
        float* aligned_src = impl.d_aligned + static_cast<size_t>(offset) * 3 * 112 * 112;
        err = cudaMemcpyAsync(impl.engine->input_buffer(), aligned_src, input_chunk_bytes,
                              cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                ("aligned-to-engine copy failed"), ("%s", cudaGetErrorString(err)));
            gst_buffer_unmap(buf, &inmap);
            return GST_FLOW_ERROR;
        }

        std::string err_msg;
        if (!impl.engine->enqueue(chunk, stream, &err_msg)) {
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                ("GlintR100 enqueue failed"), ("%s", err_msg.c_str()));
            gst_buffer_unmap(buf, &inmap);
            return GST_FLOW_ERROR;
        }

        float* norm_dst = impl.d_normalized + static_cast<size_t>(offset) * 512;
        cu = mergenvision_l2_normalize(
            impl.engine->output_buffer(), norm_dst, chunk, 512, 1e-12f,
            impl.d_status, stream);
        if (cu != 0) {
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                ("l2_normalize failed"), ("cuda error %d", cu));
            gst_buffer_unmap(buf, &inmap);
            return GST_FLOW_ERROR;
        }
    }

    // Copy matrices and normalized embeddings to host for CPU-side decision.
    err = cudaMemcpyAsync(impl.h_matrices, impl.d_matrices,
                          static_cast<size_t>(n_faces) * 6 * sizeof(float),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) { gst_buffer_unmap(buf, &inmap); return GST_FLOW_ERROR; }
    err = cudaMemcpyAsync(impl.h_embeddings, impl.d_normalized,
                          static_cast<size_t>(n_faces) * 512 * sizeof(float),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) { gst_buffer_unmap(buf, &inmap); return GST_FLOW_ERROR; }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        GST_ELEMENT_ERROR(self, STREAM, FAILED,
            ("stream synchronize failed"), ("%s", cudaGetErrorString(err)));
        gst_buffer_unmap(buf, &inmap);
        return GST_FLOW_ERROR;
    }

    // Attach recognition metadata per face under the metadata lock.
    nvds_acquire_meta_lock(batch_meta);
    const float* emb = impl.h_embeddings;
    for (int i = 0; i < n_faces; ++i, emb += 512) {
        const FaceContext& fc = faces[i];

        // Check matrix health.
        const float* M = impl.h_matrices + i * 6;
        float a = M[0], b = M[3];
        float det = a * a + b * b;
        bool matrix_valid = std::isfinite(det) && det > 0.0f && fc.sane;

        mergenvision::Gallery::Match match;
        if (matrix_valid) {
            match = impl.gallery->match(emb, self->threshold, self->margin);
        } else {
            match.status = "invalid";
            match.identity_id = "";
            match.identity_name = "";
            match.top1_similarity = 0.0f;
            match.top2_similarity = 0.0f;
            match.margin = 0.0f;
        }

        MvFaceRecognitionMeta* rec = mv_face_recognition_meta_new();
        rec->schema_version = 1;
        rec->frame_num = fc.frame ? fc.frame->frame_num : 0;
        rec->pts_ns = fc.frame ? fc.frame->buf_pts : 0;
        rec->detection_id = fc.det_id;
        rec->x1 = fc.x1; rec->y1 = fc.y1; rec->x2 = fc.x2; rec->y2 = fc.y2;
        std::snprintf(rec->identity_id, sizeof(rec->identity_id), "%s", match.identity_id.c_str());
        std::snprintf(rec->identity_name, sizeof(rec->identity_name), "%s", match.identity_name.c_str());
        std::snprintf(rec->status, sizeof(rec->status), "%s", match.status.c_str());
        rec->top1_similarity = match.top1_similarity;
        rec->top2_similarity = match.top2_similarity;
        rec->margin = match.margin;
        rec->embedding_quality = match.top1_similarity; // nearest proxy
        std::snprintf(rec->engine_sha256, sizeof(rec->engine_sha256), "%s", impl.engine_sha256.c_str());
        std::snprintf(rec->gallery_sha256, sizeof(rec->gallery_sha256), "%s", impl.gallery_sha256.c_str());
        std::snprintf(rec->preprocess_contract_sha256, sizeof(rec->preprocess_contract_sha256), "%s", impl.contract_sha256.c_str());
        std::memcpy(rec->embedding, emb, sizeof(rec->embedding));
        if (!matrix_valid) {
            std::memset(rec->embedding, 0, sizeof(rec->embedding));
        }

        if (!mv_face_recognition_meta_attach(batch_meta, fc.obj, rec)) {
            mv_face_recognition_meta_release(rec, nullptr);
            nvds_release_meta_lock(batch_meta);
            GST_ELEMENT_ERROR(self, STREAM, FAILED,
                ("failed to attach recognition meta"), (NULL));
            gst_buffer_unmap(buf, &inmap);
            return GST_FLOW_ERROR;
        }
    }
    nvds_release_meta_lock(batch_meta);

    gst_buffer_unmap(buf, &inmap);
    return GST_FLOW_OK;
}

static void gst_mv_face_recognizer_class_init(GstMvFaceRecognizerClass* klass) {
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass* base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class->set_property = gst_mv_face_recognizer_set_property;
    gobject_class->get_property = gst_mv_face_recognizer_get_property;
    gobject_class->finalize = gst_mv_face_recognizer_finalize;

    g_object_class_install_property(gobject_class, PROP_ENGINE_FILE,
        g_param_spec_string("engine-file", "Engine config",
            "Path to the GlintR100 preprocess contract JSON (engine config)", NULL,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_GALLERY_FILE,
        g_param_spec_string("gallery-file", "Gallery file",
            "Path to the gallery centroids JSON", NULL,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_THRESHOLD,
        g_param_spec_float("threshold", "Recognition threshold",
            "Minimum cosine similarity for a known identity",
            0.0f, 1.0f, 0.5f,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_MARGIN,
        g_param_spec_float("margin", "Recognition margin",
            "Minimum top1 - top2 cosine margin for a known identity",
            0.0f, 2.0f, 0.2f,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_GPU_ID,
        g_param_spec_int("gpu-id", "GPU ID", "GPU device ID", 0, G_MAXINT, 0,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_details_simple(element_class,
        "MergenVision Face Recognizer",
        "Filter",
        "GPU face recognition and gallery matching",
        "MergenVision");

    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS,
            gst_caps_from_string("video/x-raw(memory:NVMM), format=RGBA")));
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
            gst_caps_from_string("video/x-raw(memory:NVMM), format=RGBA")));

    base_transform_class->transform_caps = gst_mv_face_recognizer_transform_caps;
    base_transform_class->start = gst_mv_face_recognizer_start;
    base_transform_class->stop = gst_mv_face_recognizer_stop;
    base_transform_class->transform_ip = gst_mv_face_recognizer_transform_ip;
}

static void gst_mv_face_recognizer_init(GstMvFaceRecognizer* self) {
    (void)self;
}

static gboolean plugin_init(GstPlugin* plugin) {
    return GST_ELEMENT_REGISTER(mvfacerecognizer, plugin);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    mvfacerecognizer,
    "MergenVision face recognizer plugin",
    plugin_init,
    PACKAGE_VERSION,
    GST_LICENSE_UNKNOWN,
    "MergenVision",
    "mergenvision"
)
