/*
 Copyright (c) 2025 Changhe Liu 
 Licensed under the Apache License, Version 2.0. 

 For inquiries contact lch01234@gmail.com
*/
#pragma once
#include "optix_types.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <string>
#include <torch/extension.h>

template <typename T>
struct SbtRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct RayGenData {};
struct HitGroupData {};
struct MissData {};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;

typedef torch::Tensor Tensor;

struct TracerState {
  CUstream stream = 0;

  // gas resources
  OptixTraversableHandle gas_handle = 0;

  CUdeviceptr d_gas_buffer_temp = 0;
  CUdeviceptr gas_buffer_tmpe_size = 0;

  CUdeviceptr d_gas_buffer = 0;
  CUdeviceptr gas_buffer_size = 0;

  CUdeviceptr d_vertices = 0;
  torch::Tensor indexbuf_tensor;
  CUdeviceptr d_indexbuf = 0;

  // optix resources
  OptixDeviceContext context = nullptr;
  OptixModule ptx_module = nullptr;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixPipeline pipeline = nullptr;

  OptixProgramGroup raygen_prog_group = nullptr;
  OptixProgramGroup hitgroup_prog_group = nullptr;
  OptixProgramGroup miss_prog_group = nullptr;

  OptixShaderBindingTable sbt = {};
};

struct FwdTraceResult {
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> data;
};

struct BwdTraceResult {
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> data;
};

class OptixTracer {

public:
  TracerState fwdState;
  TracerState bwdState;

  OptixTracer(const std::string &pkgDir) : OptixTracer(pkgDir, "forward.ptx", "backward.ptx") {};
  OptixTracer(const std::string &pkgDir, const std::string &forward, const std::string &backward);

  void build_tri_accel(TracerState &state, uint32_t numVertices, uint32_t numTri, bool rebuild);

  FwdTraceResult trace_fwd(
      const Tensor &ray_o,
      const Tensor &ray_d,
      const Tensor &triangles_points,
      const Tensor &sigma,
      const Tensor &num_points_per_triangle,
      const Tensor &cumsum_of_points_per_triangle,
      const int number_of_points,
      const Tensor &shs,
      const int sh_degree,
      const Tensor &colors_precomp,
      const Tensor &opacities,
      Tensor &scaling,
      Tensor &density_factor,
      const Tensor &bg_color,
      const Tensor &extra,
      const int rebuild_gas_opt,
      const bool debug);

  BwdTraceResult trace_bwd(
      const Tensor &ray_o,
      const Tensor &ray_d,
      const Tensor &background,
      const Tensor &triangles_points,
      const Tensor &sigma,
      const torch::Tensor &num_points_per_triangle,
      const torch::Tensor &cumsum_of_points_per_triangle,
      const torch::Tensor &colors,
      const torch::Tensor &others,
      const int number_of_points,
      const Tensor &shs,
      const int degree,
      const Tensor &opacities,
      const Tensor &dL_dpixels,
      const Tensor &dL_dothers,
      const Tensor &scaling,
      const bool debug);

  ~OptixTracer();

  OptixTracer(OptixTracer &&) = default;
  OptixTracer &operator=(OptixTracer &&) = default;
  // no copy
  OptixTracer(const OptixTracer &) = delete;
  OptixTracer &operator=(const OptixTracer &) = delete;

private:
  OptixDeviceContext context;
};
