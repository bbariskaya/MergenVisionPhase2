/* Unit tests for Gallery loader/matcher. */
#include <recognition/gallery.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

namespace fs = std::filesystem;
using namespace mergenvision;

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond << "\n"; \
        ++g_failures; \
    } \
} while (0)

static fs::path write_gallery(const char* name, const std::string& content) {
    fs::path p = fs::temp_directory_path() / name;
    std::ofstream f(p);
    f << content;
    return p;
}

static std::string identity_json(const char* key, const char* id,
                                  const char* display,
                                  const float* centroid) {
    std::ostringstream oss;
    oss << "\"" << key << "\":{\"canonical_face_id\":\"" << id
        << "\",\"display_name\":\"" << display
        << "\",\"centroid\":[";
    for (int i = 0; i < kEmbeddingDim; ++i) {
        if (i) oss << ",";
        oss << centroid[i];
    }
    oss << "],\"image_count\":1}";
    return oss.str();
}

static void unit_vector(float* v, int n, float offset = 0.0f) {
    for (int i = 0; i < n; ++i) {
        v[i] = ((i + 1) % 7) * 0.1f + offset;
    }
    float norm = std::sqrt(std::inner_product(v, v + n, v, 0.0f));
    for (int i = 0; i < n; ++i) v[i] /= norm;
}

int main() {
    // Build deterministic orthonormal centroids for unambiguous matches.
    float a[kEmbeddingDim] = {0};
    float b[kEmbeddingDim] = {0};
    float c[kEmbeddingDim] = {0};
    a[0] = 1.0f;
    b[1] = 1.0f;
    c[2] = 1.0f;

    std::string json = "{\"schema_version\":\"1.0.0\",\"identities\":{"
        + identity_json("alice", "alice", "Alice", a) + ","
        + identity_json("bob", "bob", "Bob", b) + ","
        + identity_json("charlie", "charlie", "Charlie", c)
        + "}}";
    fs::path path = write_gallery("mv_test_gallery.json", json);

    Gallery g;
    std::string error;
    bool ok = g.load(path.string(), &error);
    if (!ok) {
        std::cerr << "Gallery load error: " << error << "\n";
    }
    CHECK(ok);
    CHECK(g.size() == 3);

    // Known positive: exact Alice centroid.
    Gallery::Match m = g.match(a, 0.5f, 0.05f);
    CHECK(m.status == "known");
    CHECK(m.identity_id == "alice");
    CHECK(m.identity_name == "Alice");
    CHECK(std::fabs(m.top1_similarity - 1.0f) < 1e-4f);
    CHECK(m.margin > 0.5f);

    // Out-of-gallery query -> unknown.
    float other[kEmbeddingDim] = {0};
    other[3] = 1.0f;
    m = g.match(other, 0.5f, 0.05f);
    CHECK(m.status == "unknown");

    // Low threshold: identity is Alice but threshold too high -> unknown.
    m = g.match(a, 1.5f, 0.01f);
    CHECK(m.status == "unknown");

    // Insufficient margin: Alice matched but margin threshold too high.
    m = g.match(a, 0.5f, 5.0f);
    CHECK(m.status == "unknown");

    // NaN embedding -> invalid.
    float bad[kEmbeddingDim];
    std::fill(bad, bad + kEmbeddingDim, std::nanf(""));
    m = g.match(bad, 0.5f, 0.05f);
    CHECK(m.status == "invalid");

    // Deterministic repeat.
    Gallery g2;
    ok = g2.load(path.string(), &error);
    CHECK(ok);
    CHECK(g.identities()[0].id == g2.identities()[0].id);
    CHECK(g.identities()[1].id == g2.identities()[1].id);
    CHECK(g.sha256() == g2.sha256());

    // Duplicate ID must fail.
    std::string dup = "{\"schema_version\":\"1.0.0\",\"identities\":{"
        + identity_json("alice", "alice", "A", a) + ","
        + identity_json("alice2", "alice", "A2", b)
        + "}}";
    fs::path dup_path = write_gallery("mv_test_dup.json", dup);
    Gallery dup_g;
    ok = dup_g.load(dup_path.string(), &error);
    CHECK(!ok);

    // Empty gallery must fail.
    std::string empty = "{\"schema_version\":\"1.0.0\",\"identities\":{}}";
    fs::path empty_path = write_gallery("mv_test_empty.json", empty);
    Gallery empty_g;
    ok = empty_g.load(empty_path.string(), &error);
    CHECK(!ok);

    // Tie ordering: two near-identical centroids -> top1 is lexicographically first id.
    // We make alice2 slightly different from alice.
    float a2[kEmbeddingDim];
    std::copy(a, a + kEmbeddingDim, a2);
    a2[0] += 1e-5f;
    unit_vector(a2, kEmbeddingDim, 0.0f); // back to unit
    std::string tie_json = "{\"schema_version\":\"1.0.0\",\"identities\":{"
        + identity_json("alice", "alice", "Alice", a) + ","
        + identity_json("alice2", "alice2", "Alice2", a2)
        + "}}";
    fs::path tie_path = write_gallery("mv_test_tie.json", tie_json);
    Gallery tie_g;
    ok = tie_g.load(tie_path.string(), &error);
    CHECK(ok);
    m = tie_g.match(a, 0.5f, 0.0f);
    CHECK(m.identity_id == "alice");

    fs::remove(path);
    fs::remove(dup_path);
    fs::remove(empty_path);
    fs::remove(tie_path);

    if (g_failures) {
        std::cerr << "gallery tests FAILED: " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "gallery tests PASSED\n";
    return 0;
}
