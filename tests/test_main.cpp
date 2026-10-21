#include <cstring>
#include <exception>
#include <iostream>
#include <string>

#include "testing.h"

// Usage: kvpage_tests [--filter substring] [--list]
int main(int argc, char** argv) {
    std::string filter;
    bool list = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--filter") && i + 1 < argc) filter = argv[++i];
        else if (!std::strcmp(argv[i], "--list")) list = true;
    }
    int ran = 0, failed = 0;
    for (const auto& c : testing::registry()) {
        if (!filter.empty() && std::string(c.name).find(filter) == std::string::npos) continue;
        if (list) {
            std::cout << c.name << "\n";
            continue;
        }
        int before = testing::failures();
        try {
            c.fn();
        } catch (const std::exception& e) {
            ++testing::failures();
            std::cerr << "  uncaught exception: " << e.what() << "\n";
        }
        bool ok = testing::failures() == before;
        std::cout << (ok ? "[ OK ] " : "[FAIL] ") << c.name << "\n";
        ++ran;
        if (!ok) ++failed;
    }
    if (list) return 0;
    std::cout << ran << " tests run, " << failed << " failed\n";
    return failed ? 1 : 0;
}
