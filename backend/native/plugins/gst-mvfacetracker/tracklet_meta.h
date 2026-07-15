#pragma once

#include <cstdint>

#include "gstnvdsmeta.h"

namespace mv::tracker {

inline constexpr std::uint32_t kMvTrackletMetaSchemaVersion = 1;
inline constexpr NvDsMetaType kMvTrackletMetaType =
    static_cast<NvDsMetaType>(NVDS_START_USER_META + 0x2001);

struct MvTrackletMeta {
    std::uint32_t schema_version = kMvTrackletMetaSchemaVersion;

    std::uint64_t raw_object_id = 0;
    std::uint64_t tracklet_id = 0;
    std::uint32_t source_id = 0;

    std::uint64_t frame_number = 0;
    std::uint64_t pts_ns = 0;

    std::uint32_t track_age = 0;
    std::uint32_t hit_count = 0;
    std::uint32_t lost_count = 0;

    bool tentative = false;
    bool confirmed = false;
};

void mv_tracklet_meta_register();

NvDsUserMeta* mv_tracklet_meta_create(NvDsBatchMeta* batch_meta,
                                      const MvTrackletMeta& meta);

}  // namespace mv::tracker
