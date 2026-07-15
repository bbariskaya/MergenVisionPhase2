#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include "gstnvdsmeta.h"

#include <cstdint>
#include <string>

/* Custom user metadata attached to NvDsObjectMeta by mvfacerecognizer.
 *
 * This structure is heap-allocated per face, attached via NvDsUserMeta,
 * and propagated through queues/nvstreamdemux/nvdsosd/encoder via copy/release
 * callbacks registered with NvDsMeta.  It contains no pointers into transient
 * GPU buffers (e.g., the TensorRT output buffer).
 *
 * All strings are fixed-size to avoid allocator/cross-plugin ABI issues during
 * NvDs deep-copies.
 */
struct MvFaceRecognitionMeta {
    static constexpr int kEmbeddingDim = 512;
    static constexpr int kIdLen = 64;
    static constexpr int kNameLen = 64;
    static constexpr int kStatusLen = 16;
    static constexpr int kShaLen = 65;

    int schema_version = 1;

    uint64_t frame_num = 0;
    uint64_t pts_ns = 0;
    int detection_id = 0;

    // Original-resolution bbox in xyxy form.
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;

    // Recognition decision.
    char identity_id[kIdLen];
    char identity_name[kNameLen];
    char status[kStatusLen];  // known, unknown, invalid
    float top1_similarity = 0.0f;
    float top2_similarity = 0.0f;
    float margin = 0.0f;
    float embedding_quality = 0.0f;

    // Stable SHA references for audit/evidence.
    char engine_sha256[kShaLen];
    char gallery_sha256[kShaLen];
    char preprocess_contract_sha256[kShaLen];

    // An owned copy of the L2-normalized embedding (compact, 2 KiB per face).
    float embedding[kEmbeddingDim];
};

/* Register the custom meta type if needed and return the registered type. */
NvDsMetaType mv_face_recognition_meta_type();

/* Allocate and initialize a new MvFaceRecognitionMeta. */
MvFaceRecognitionMeta* mv_face_recognition_meta_new();

/* NvDsUserMeta copy/release callbacks. */
void* mv_face_recognition_meta_copy(void* data, gpointer user_data);
void mv_face_recognition_meta_release(void* data, gpointer user_data);

/* Attach a recognition meta to an NvDsObjectMeta. Returns false on failure. */
bool mv_face_recognition_meta_attach(NvDsBatchMeta* batch_meta,
                                     NvDsObjectMeta* obj_meta,
                                     MvFaceRecognitionMeta* rec_meta);
