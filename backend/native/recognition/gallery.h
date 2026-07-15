#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace mergenvision {

static constexpr int kEmbeddingDim = 512;

struct GalleryIdentity {
    std::string id;          // stable/canonical identity ID
    std::string display_name;
    std::array<float, kEmbeddingDim> centroid{};
};

/* Deterministic CPU gallery loader and matcher.
 *
 * The JSON schema expected is:
 *   {
 *     "schema_version": "x.y.z",
 *     "identities": {
 *       "<key>": {
 *         "canonical_face_id": "id",
 *         "display_name": "Name",
 *         "centroid": [512 floats]
 *       },
 *       ...
 *     }
 *   }
 *
 * Order is deterministic: keys are sorted lexicographically by canonical_face_id.
 * Centroids are validated finite and unit-normalized (or normalized on load).
 */
class Gallery {
public:
    Gallery();
    ~Gallery();

    Gallery(const Gallery&) = default;
    Gallery& operator=(const Gallery&) = default;

    bool load(const std::string& json_path, std::string* error);

    bool loaded() const { return !identities_.empty(); }
    size_t size() const { return identities_.size(); }
    const std::vector<GalleryIdentity>& identities() const { return identities_; }

    const std::string& path() const { return path_; }
    const std::string& schema_version() const { return schema_version_; }
    const std::string& sha256() const { return sha256_; }

    struct Match {
        std::string identity_id;
        std::string identity_name;
        std::string status;       // "known", "unknown", or "invalid"
        float top1_similarity = 0.0f;
        float top2_similarity = 0.0f;
        float margin = 0.0f;
        float quality = 0.0f;     // L2 norm of raw embedding if available
    };

    /* Match one L2-normalized embedding against the gallery.
     * threshold: minimum cosine similarity for status "known".
     * margin_threshold: minimum gap between top1 and top2 distinct identity.
     */
    Match match(const float* normalized_embedding,
                float threshold,
                float margin_threshold) const;

private:
    bool parse_file(const std::string& contents, std::string* error);
    bool validate_centroid_finite(const float* centroid, std::string* error);

    std::string path_;
    std::string schema_version_;
    std::string sha256_;
    std::vector<GalleryIdentity> identities_;
};

} // namespace mergenvision
