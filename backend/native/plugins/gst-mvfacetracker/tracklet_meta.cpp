#include "tracklet_meta.h"

#include <cstring>

namespace mv::tracker {

static gpointer mv_tracklet_meta_copy(gpointer data, gpointer /*user_data*/) {
    if (!data) return nullptr;
    auto* src = static_cast<MvTrackletMeta*>(data);
    auto* dst = static_cast<MvTrackletMeta*>(g_malloc(sizeof(MvTrackletMeta)));
    if (!dst) return nullptr;
    std::memcpy(dst, src, sizeof(MvTrackletMeta));
    return dst;
}

static void mv_tracklet_meta_release(gpointer data, gpointer /*user_data*/) {
    if (data) {
        g_free(data);
    }
}

NvDsUserMeta* mv_tracklet_meta_create(NvDsBatchMeta* batch_meta,
                                      const MvTrackletMeta& meta) {
    if (!batch_meta) return nullptr;

    auto* copy = static_cast<MvTrackletMeta*>(g_malloc(sizeof(MvTrackletMeta)));
    if (!copy) return nullptr;
    std::memcpy(copy, &meta, sizeof(MvTrackletMeta));

    NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    if (!user_meta) {
        g_free(copy);
        return nullptr;
    }

    user_meta->user_meta_data = copy;
    user_meta->base_meta.meta_type = kMvTrackletMetaType;
    user_meta->base_meta.copy_func = mv_tracklet_meta_copy;
    user_meta->base_meta.release_func = mv_tracklet_meta_release;
    return user_meta;
}

}  // namespace mv::tracker
