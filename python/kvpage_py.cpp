// pybind11 binding: exposes the real C++ KVManager to Python so a model harness can drive it.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "kvpage/kv_manager.h"
#include "kvpage/paged_attention.h"

namespace py = pybind11;
using namespace kvp;

PYBIND11_MODULE(kvpage_py, m) {
    m.doc() = "Python binding for kvpage::KVManager";
    py::class_<KVStats>(m, "KVStats")
        .def_readonly("requests", &KVStats::requests)
        .def_readonly("used_tokens", &KVStats::used_tokens)
        .def_readonly("allocated_blocks", &KVStats::allocated_blocks)
        .def_readonly("internal_waste", &KVStats::internal_waste)
        .def_readonly("blocks_saved", &KVStats::blocks_saved)
        .def_readonly("cow_copies", &KVStats::cow_copies);
    py::class_<KVManager>(m, "KVManager")
        .def(py::init<int, int>(), py::arg("num_blocks"), py::arg("block_size"))
        .def("can_admit", &KVManager::can_admit)
        .def("blocks_needed_to_admit", &KVManager::blocks_needed_to_admit, py::arg("parent"),
             py::arg("shared_tokens"), py::arg("prompt_tokens"))
        .def("admit", &KVManager::admit)
        .def("admit_with_prefix", &KVManager::admit_with_prefix)
        .def("fork", &KVManager::fork)
        .def("append_token", &KVManager::append_token)
        .def("free", &KVManager::free)
        .def("contains", &KVManager::contains)
        .def("block_table", [](const KVManager& k, RequestId r) { return k.table(r).blocks; })
        .def("num_tokens", [](const KVManager& k, RequestId r) { return k.table(r).tokens; })
        .def("refcount", [](const KVManager& k, BlockId b) { return k.allocator().refcount(b); })
        .def("free_blocks", &KVManager::free_blocks)
        .def("total_blocks", &KVManager::total_blocks)
        .def("block_size", &KVManager::block_size)
        .def("stats", &KVManager::stats)
        .def("validate", &KVManager::validate)
        .def("set_check_each_op", &KVManager::set_check_each_op);

    // ---- decode-attention references (same layouts as cuda/paged_attention.cu) ----
    using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
    using I32 = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
    auto run = [](bool partitioned, int partitions, int nh, int kvh, int hd, int bs, F32 q, F32 k, F32 v, I32 tables,
                  I32 lens) {
        AttnShape s{nh, kvh, hd, bs};
        F32 out(q.request().shape);
        int ns = static_cast<int>(q.shape(0));
        int mb = static_cast<int>(tables.shape(1));
        if (partitioned)
            paged_attention_cpu(s, ns, mb, q.data(), k.data(), v.data(), tables.data(), lens.data(), out.mutable_data(),
                                partitions);
        else
            reference_attention(s, ns, mb, q.data(), k.data(), v.data(), tables.data(), lens.data(),
                                out.mutable_data());
        return out;
    };
    m.def("reference_attention",
          [run](int nh, int kvh, int hd, int bs, F32 q, F32 k, F32 v, I32 t, I32 l) {
              return run(false, 1, nh, kvh, hd, bs, q, k, v, t, l);
          });
    m.def("paged_attention_cpu",
          [run](int nh, int kvh, int hd, int bs, F32 q, F32 k, F32 v, I32 t, I32 l, int partitions) {
              return run(true, partitions, nh, kvh, hd, bs, q, k, v, t, l);
          });
}
