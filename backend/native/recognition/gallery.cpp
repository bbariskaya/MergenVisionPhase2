#include "gallery.h"

#include <cjson/cJSON.h>
#include <openssl/evp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mergenvision {

namespace {

std::string compute_sha256_hex(const std::string& data) {
    unsigned char digest[EVP_MAX_MD_SIZE] = {0};
    unsigned int digest_len = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) return "";
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data.data(), data.size()) != 1 ||
        EVP_DigestFinal_ex(ctx, digest, &digest_len) != 1) {
        EVP_MD_CTX_free(ctx);
        return "";
    }
    EVP_MD_CTX_free(ctx);

    std::ostringstream oss;
    for (unsigned int i = 0; i < digest_len; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(digest[i]);
    }
    return oss.str();
}

bool read_file(const std::string& path, std::string* contents, std::string* error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (error) *error = "failed to open gallery file: " + path;
        return false;
    }
    *contents = std::string((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    return true;
}

bool is_finite(const float* v, int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) return false;
    }
    return true;
}

float l2_norm_sq(const float* v, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += v[i] * v[i];
    return s;
}

} // namespace

Gallery::Gallery() = default;
Gallery::~Gallery() = default;

bool Gallery::load(const std::string& json_path, std::string* error) {
    path_ = json_path;
    identities_.clear();
    schema_version_.clear();
    sha256_.clear();

    std::string contents;
    if (!read_file(json_path, &contents, error)) return false;

    sha256_ = compute_sha256_hex(contents);
    if (sha256_.empty()) {
        if (error) *error = "SHA-256 computation failed";
        return false;
    }

    if (!parse_file(contents, error)) return false;

    if (identities_.empty()) {
        if (error) *error = "gallery contains no valid identities";
        return false;
    }

    // Deterministic lexical order by canonical id.
    std::sort(identities_.begin(), identities_.end(),
              [](const GalleryIdentity& a, const GalleryIdentity& b) {
                  return a.id < b.id;
              });

    // Validate no duplicate ids after normalization.
    for (size_t i = 1; i < identities_.size(); ++i) {
        if (identities_[i].id == identities_[i - 1].id) {
            if (error) *error = "duplicate canonical_face_id: " + identities_[i].id;
            return false;
        }
    }

    return true;
}

bool Gallery::parse_file(const std::string& contents, std::string* error) {
    cJSON* root = cJSON_Parse(contents.c_str());
    if (!root) {
        if (error) *error = "JSON parse failed";
        return false;
    }

    cJSON* schema = cJSON_GetObjectItemCaseSensitive(root, "schema_version");
    if (cJSON_IsString(schema)) {
        schema_version_ = schema->valuestring;
    } else {
        cJSON_Delete(root);
        if (error) *error = "missing schema_version";
        return false;
    }

    cJSON* identities = cJSON_GetObjectItemCaseSensitive(root, "identities");
    if (!cJSON_IsObject(identities)) {
        cJSON_Delete(root);
        if (error) *error = "missing or invalid identities object";
        return false;
    }

    cJSON* entry = nullptr;
    cJSON_ArrayForEach(entry, identities) {
        if (!cJSON_IsObject(entry)) continue;

        cJSON* cid = cJSON_GetObjectItemCaseSensitive(entry, "canonical_face_id");
        cJSON* dname = cJSON_GetObjectItemCaseSensitive(entry, "display_name");
        cJSON* centroid = cJSON_GetObjectItemCaseSensitive(entry, "centroid");

        if (!cJSON_IsString(cid) || std::string(cid->valuestring).empty()) {
            cJSON_Delete(root);
            if (error) *error = "identity missing canonical_face_id";
            return false;
        }
        if (!cJSON_IsString(dname) || std::string(dname->valuestring).empty()) {
            cJSON_Delete(root);
            if (error) *error = "identity missing display_name";
            return false;
        }
        if (!cJSON_IsArray(centroid) ||
            cJSON_GetArraySize(centroid) != kEmbeddingDim) {
            cJSON_Delete(root);
            if (error) *error = "identity centroid must be length " + std::to_string(kEmbeddingDim);
            return false;
        }

        GalleryIdentity id;
        id.id = cid->valuestring;
        id.display_name = dname->valuestring;

        for (int i = 0; i < kEmbeddingDim; ++i) {
            cJSON* v = cJSON_GetArrayItem(centroid, i);
            if (!cJSON_IsNumber(v)) {
                cJSON_Delete(root);
                if (error) {
                    *error = "centroid value is not a number for id " + id.id;
                }
                return false;
            }
            id.centroid[i] = static_cast<float>(v->valuedouble);
        }

        if (!is_finite(id.centroid.data(), kEmbeddingDim)) {
            cJSON_Delete(root);
            if (error) *error = "non-finite centroid for id " + id.id;
            return false;
        }

        // Ensure unit length (normalize if slightly off due to float drift).
        float norm_sq = l2_norm_sq(id.centroid.data(), kEmbeddingDim);
        if (norm_sq <= 1e-12f) {
            cJSON_Delete(root);
            if (error) *error = "zero-norm centroid for id " + id.id;
            return false;
        }
        float scale = 1.0f / std::sqrt(norm_sq);
        for (float& v : id.centroid) v *= scale;

        identities_.push_back(id);
    }

    cJSON_Delete(root);
    return true;
}

Gallery::Match Gallery::match(const float* normalized_embedding,
                              float threshold,
                              float margin_threshold) const {
    Match result;
    if (!normalized_embedding || !is_finite(normalized_embedding, kEmbeddingDim)) {
        result.status = "invalid";
        return result;
    }

    float best_sim = -2.0f;
    float second_sim = -2.0f;
    size_t best_idx = identities_.size();
    for (size_t i = 0; i < identities_.size(); ++i) {
        float sim = 0.0f;
        for (int j = 0; j < kEmbeddingDim; ++j) {
            sim += normalized_embedding[j] * identities_[i].centroid[j];
        }
        if (sim > best_sim) {
            second_sim = best_sim;
            best_sim = sim;
            best_idx = i;
        } else if (sim > second_sim) {
            second_sim = sim;
        }
    }

    result.top1_similarity = best_sim;
    result.top2_similarity = second_sim;
    result.margin = best_sim - second_sim;

    if (best_idx < identities_.size()) {
        result.identity_id = identities_[best_idx].id;
        result.identity_name = identities_[best_idx].display_name;
    }

    if (best_sim >= threshold && result.margin >= margin_threshold) {
        result.status = "known";
    } else {
        result.status = "unknown";
    }

    return result;
}

} // namespace mergenvision
