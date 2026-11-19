#include "ops.h"

PYBIND11_MODULE(_kernels_native_batch, m) {
    m.def("append", &native_append);
    m.def("rope", &native_rope);
    m.def("rms_norm_forward", &rms_norm_forward);
    m.def("estimate", &native_estimate);
    m.def("topk", &native_topk);
    m.def("prefill", &native_prefill);
    pybind11::class_<NativeDecodePlan>(m, "DecodePlan")
        .def(pybind11::init<torch::Tensor, uint32_t, uint32_t, uint32_t, uint32_t, bool>())
        .def("run", &NativeDecodePlan::run)
        .def("close", &NativeDecodePlan::close);
}
