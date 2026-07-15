/* Unit test for GlintR100Engine wrapper.
 *
 * Requires a compatible TensorRT engine at the standard artifact path.
 * Missing engine is reported but does not fail the test suite.
 */
#include <recognition/glintr100_engine.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

using namespace mergenvision;

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond << "\n"; \
        ++g_failures; \
    } \
} while (0)

int main() {
    const char* config_path = "/app/backend/native/configs/glintr100_preprocess_contract.json";

    GlintR100Engine engine;

    if (FILE* f = std::fopen(config_path, "rb")) {
        std::fclose(f);
    } else {
        std::cout << "GlintR100 config not present; skipping engine test\n";
        return 0;
    }

    std::string error;
    bool ok = engine.load(0, config_path, &error);
    if (!ok) {
        std::cerr << "engine load failed: " << error << "\n";
        return 1;
    }

    CHECK(engine.loaded());
    CHECK(engine.max_batch() >= 1);
    CHECK(engine.max_batch() <= 256);
    CHECK(engine.input_buffer() != nullptr);
    CHECK(engine.output_buffer() != nullptr);

    // Enqueue smallest and largest allowed counts.
    ok = engine.enqueue(1, 0, &error);
    CHECK(ok);

    ok = engine.enqueue(engine.max_batch(), 0, &error);
    CHECK(ok);

    if (g_failures) {
        std::cerr << "glintR100 engine tests FAILED: " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "glintR100 engine tests PASSED (max_batch=" << engine.max_batch() << ")\n";
    return 0;
}
