#pragma once
// Minimal self-contained test framework (no external dependencies).
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace testing {

struct Case {
    const char* name;
    void (*fn)();
};
inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}
inline int& failures() {
    static int f = 0;
    return f;
}
struct Registrar {
    Registrar(const char* n, void (*f)()) { registry().push_back({n, f}); }
};

}  // namespace testing

#define TEST(name)                                                  \
    static void test_##name();                                      \
    static ::testing::Registrar reg_##name(#name, &test_##name);    \
    static void test_##name()

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            ++::testing::failures();                                                  \
            std::cerr << "  CHECK failed " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n"; \
        }                                                                             \
    } while (0)

#define CHECK_EQ(a, b)                                                                \
    do {                                                                              \
        auto va_ = (a);                                                               \
        auto vb_ = (b);                                                               \
        if (!(va_ == vb_)) {                                                          \
            ++::testing::failures();                                                  \
            std::cerr << "  CHECK_EQ failed " << __FILE__ << ":" << __LINE__ << ": " #a " == " #b \
                      << "  (" << va_ << " vs " << vb_ << ")\n";                      \
        }                                                                             \
    } while (0)

#define CHECK_THROWS(expr)                                                            \
    do {                                                                              \
        bool threw_ = false;                                                          \
        try {                                                                         \
            expr;                                                                     \
        } catch (const std::exception&) {                                             \
            threw_ = true;                                                            \
        }                                                                             \
        if (!threw_) {                                                                \
            ++::testing::failures();                                                  \
            std::cerr << "  CHECK_THROWS failed " << __FILE__ << ":" << __LINE__ << ": " #expr "\n"; \
        }                                                                             \
    } while (0)
