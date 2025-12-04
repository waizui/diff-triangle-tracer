/*
 Copyright (c) 2025 Changhe Liu 
 Licensed under the Apache License, Version 2.0. 

 For inquiries contact lch01234@gmail.com
*/
#include "tritracer/tracer.h"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<FwdTraceResult>(m, "FwdTraceResult")
      .def(pybind11::init<>())
      .def_readonly("data", &FwdTraceResult::data);

  pybind11::class_<BwdTraceResult>(m, "BwdTraceResult")
      .def(pybind11::init<>())
      .def_readonly("data", &BwdTraceResult::data);

  // Bind the main OptixTracer class
  pybind11::class_<OptixTracer>(m, "OptixTracer")
      .def(pybind11::init<const std::string &>())
      .def(pybind11::init<const std::string &, const std::string &, const std::string &>())
      .def("build_tri_accel", &OptixTracer::build_tri_accel)
      .def("trace_forward", &OptixTracer::trace_fwd)
      .def("trace_backward", &OptixTracer::trace_bwd);
}
