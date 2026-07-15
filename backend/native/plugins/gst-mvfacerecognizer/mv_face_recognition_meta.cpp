#include "mv_face_recognition_meta.h"

#include "gstnvdsmeta.h"

#include <cstring>
#include <new>

namespace {

const char* kMetaTypeName = "mv-face-recognition";

} // namespace

NvDsMetaType mv_face_recognition_meta_type() {
    static NvDsMetaType type = NVDS_USER_META;
    static gsize initialized = 0;
    if (g_once_init_enter(&initialized)) {
        NvDsMetaType t = nvds_get_user_meta_type((gchar*)kMetaTypeName);
        type = (t != 0) ? t : NVDS_USER_META;
        g_once_init_leave(&initialized, 1);
    }
    return type;
}

MvFaceRecognitionMeta* mv_face_recognition_meta_new() {
    void* p = g_malloc0(sizeof(MvFaceRecognitionMeta));
    return new (p) MvFaceRecognitionMeta();
}

void* mv_face_recognition_meta_copy(void* data, gpointer /*user_data*/) {
    NvDsUserMeta* src = reinterpret_cast<NvDsUserMeta*>(data);
    if (!src || !src->user_meta_data) return nullptr;
    MvFaceRecognitionMeta* src_meta =
        reinterpret_cast<MvFaceRecognitionMeta*>(src->user_meta_data);
    MvFaceRecognitionMeta* dst = mv_face_recognition_meta_new();
    std::memcpy(dst, src_meta, sizeof(MvFaceRecognitionMeta));
    return dst;
}

void mv_face_recognition_meta_release(void* data, gpointer /*user_data*/) {
    NvDsUserMeta* user_meta = reinterpret_cast<NvDsUserMeta*>(data);
    if (!user_meta || !user_meta->user_meta_data) return;
    MvFaceRecognitionMeta* meta =
        reinterpret_cast<MvFaceRecognitionMeta*>(user_meta->user_meta_data);
    meta->~MvFaceRecognitionMeta();
    g_free(meta);
    user_meta->user_meta_data = nullptr;
}

bool mv_face_recognition_meta_attach(NvDsBatchMeta* batch_meta,
                                     NvDsObjectMeta* obj_meta,
                                     MvFaceRecognitionMeta* rec_meta) {
    if (!batch_meta || !obj_meta || !rec_meta) return false;

    NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    if (!user_meta) {
        return false;
    }

    user_meta->user_meta_data = rec_meta;
    user_meta->base_meta.meta_type = mv_face_recognition_meta_type();
    user_meta->base_meta.copy_func = mv_face_recognition_meta_copy;
    user_meta->base_meta.release_func = mv_face_recognition_meta_release;

    nvds_add_user_meta_to_obj(obj_meta, user_meta);
    return true;
}
