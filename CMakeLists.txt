cmake_minimum_required(VERSION 3.16)
project(kvpage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)  # Debug keeps the per-operation invariant checks on
endif()

# SANITIZE: none | asan (address + undefined) | tsan (thread)
set(SANITIZE "none" CACHE STRING "Sanitizer: none, asan or tsan")
option(KVPAGE_WERROR "Treat warnings as errors" OFF)
option(KVPAGE_COVERAGE "Build with gcov coverage instrumentation (GCC/Clang)" OFF)
set(KVP_FLAGS -Wall -Wextra -Wpedantic -Wshadow)
if(KVPAGE_WERROR)
  list(APPEND KVP_FLAGS -Werror)
endif()
if(KVPAGE_COVERAGE)
  list(APPEND KVP_FLAGS --coverage -O0)
  set(KVP_COV_LINK --coverage)
endif()
set(KVP_LINK "")
if(SANITIZE STREQUAL "asan")
  list(APPEND KVP_FLAGS -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
  set(KVP_LINK -fsanitize=address,undefined)
elseif(SANITIZE STREQUAL "tsan")
  list(APPEND KVP_FLAGS -fsanitize=thread -fno-omit-frame-pointer)
  set(KVP_LINK -fsanitize=thread)
endif()

add_library(kvpage
  src/block_allocator.cpp src/kv_manager.cpp src/workload.cpp
  src/metrics.cpp src/simulator.cpp src/paged_attention.cpp)
target_include_directories(kvpage PUBLIC include)
target_compile_options(kvpage PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${KVP_FLAGS}>")
find_package(Threads REQUIRED)
target_link_libraries(kvpage PUBLIC Threads::Threads ${KVP_LINK} ${KVP_COV_LINK})

enable_testing()
add_executable(kvpage_tests
  tests/test_main.cpp tests/test_allocator.cpp tests/test_kv_manager.cpp
  tests/test_prefix_cow.cpp tests/test_random.cpp tests/test_scheduler.cpp
  tests/test_concurrency.cpp tests/test_attention.cpp tests/test_features.cpp)
target_link_libraries(kvpage_tests PRIVATE kvpage)
add_test(NAME all_tests COMMAND kvpage_tests)
add_test(NAME concurrency COMMAND kvpage_tests --filter concurrency)

add_executable(kvpage_bench bench/bench.cpp)
target_link_libraries(kvpage_bench PRIVATE kvpage)

add_executable(kvpage_cpu_bench bench/cpu_bench.cpp)
target_link_libraries(kvpage_cpu_bench PRIVATE kvpage)

# Python binding (Route A model test): cmake -DKVPAGE_PYTHON=ON -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
option(KVPAGE_PYTHON "Build the kvpage_py pybind11 module" OFF)
if(KVPAGE_PYTHON)
  set_target_properties(kvpage PROPERTIES POSITION_INDEPENDENT_CODE ON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(kvpage_py python/kvpage_py.cpp)
  target_link_libraries(kvpage_py PRIVATE kvpage)
endif()

# M6 (needs an NVIDIA GPU + CUDA toolkit 11.8+): cmake -S . -B build-cuda -DKVPAGE_CUDA=ON [-DCMAKE_CUDA_ARCHITECTURES=75]
option(KVPAGE_CUDA "Build the CUDA paged decode-attention kernel and its test" OFF)
if(KVPAGE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)  # T4, A100, A10/3090, L4/4090
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_executable(kvpage_cuda_test cuda/test_paged_attention.cu cuda/paged_attention.cu)
  target_include_directories(kvpage_cuda_test PRIVATE cuda)
  target_link_libraries(kvpage_cuda_test PRIVATE kvpage)
endif()

# libFuzzer target (clang only): cmake -DCMAKE_CXX_COMPILER=clang++ -DKVPAGE_FUZZ=ON
option(KVPAGE_FUZZ "Build the libFuzzer target for KVManager" OFF)
if(KVPAGE_FUZZ)
  add_executable(fuzz_kv fuzz/fuzz_kv.cpp)
  target_compile_options(fuzz_kv PRIVATE -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined)
  target_link_options(fuzz_kv PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_libraries(fuzz_kv PRIVATE kvpage)
endif()
