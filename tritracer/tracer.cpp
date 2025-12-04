/*
 Copyright (c) 2025 Changhe Liu 
 Licensed under the Apache License, Version 2.0. 

 For inquiries contact lch01234@gmail.com
*/
#include "tracer.h"
#include "cuda_runtime_api.h"
#include "exception.h"
#include "params.h"
#include <ATen/ops/stack.h>
#include <optix_function_table_definition.h> // only include once
#include <optix_stubs.h>                     // need for g_optixFunctionTable not found
#include <tuple>

FwdTraceResult OptixTracer::trace_fwd(
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
    const bool debug) {
  TracerState &state = this->fwdState;

  auto H = ray_o.size(0);
  auto W = ray_o.size(1);

  Params params;
  params.width = W;
  params.height = H;

  const int P = number_of_points;
  params.P = P;

  float *ptr_vert = triangles_points.contiguous().data_ptr<float>();
  params.vertices = reinterpret_cast<float3 *>(ptr_vert);
  state.d_vertices = reinterpret_cast<CUdeviceptr>(ptr_vert);

  params.num_points_per_triangle = num_points_per_triangle.contiguous().data_ptr<int>();
  params.cumsum_of_points_per_triangle = cumsum_of_points_per_triangle.contiguous().data_ptr<int>();

  // Build gas
  const int num_vert = triangles_points.size(0);
  const int num_tri = cumsum_of_points_per_triangle.size(0);
  if (rebuild_gas_opt == 0) {
    // Convert cumsum to index buffer
    state.indexbuf_tensor =
        torch::stack(
            {cumsum_of_points_per_triangle, cumsum_of_points_per_triangle + 1, cumsum_of_points_per_triangle + 2}, 1)
            .to(cumsum_of_points_per_triangle.device())
            .toType(torch::kInt32)
            .contiguous()
            .squeeze();
    state.d_indexbuf = reinterpret_cast<CUdeviceptr>(state.indexbuf_tensor.data_ptr<int>());
    this->build_tri_accel(state, num_vert, num_tri, true);
  } else if (rebuild_gas_opt == 1) {
    this->build_tri_accel(state, num_vert, num_tri, false);
  }

  params.gas_handle = state.gas_handle;

  params.ray_org = reinterpret_cast<float3 *>(ray_o.contiguous().data_ptr<float>());
  params.ray_dir = reinterpret_cast<float3 *>(ray_d.contiguous().data_ptr<float>());

  params.shs = shs.contiguous().data_ptr<float>();
  params.sh_degree = sh_degree;
  int M = 0;
  if (shs.size(0) != 0) {
    M = shs.size(1);
  }
  params.M = M;
  params.opacities = opacities.contiguous().data_ptr<float>();
  params.sigma = sigma.contiguous().data_ptr<float>();
  params.scaling = scaling.contiguous().data_ptr<float>();

  // Alloc buffers
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

  torch::Tensor out_color = torch::zeros({3, H, W}, float_opts);
  params.out_color = out_color.contiguous().data_ptr<float>();

  torch::Tensor out_others = torch::zeros({3 + 3 + 1, H, W}, float_opts);
  params.out_others = out_others.contiguous().data_ptr<float>();

  torch::Tensor max_blending = torch::zeros({P}, float_opts);
  params.max_blending = max_blending.contiguous().data_ptr<float>();

  params.density_factor = density_factor.contiguous().data_ptr<float>();
  params.bg_color = bg_color.contiguous().data_ptr<float>();

  params.extra = reinterpret_cast<uint8_t *>(extra.contiguous().data_ptr());
  params.extra_len = extra.numel() * extra.element_size();

  CUdeviceptr d_params;
  // Copy params after init
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice));

  // Record GPU time
  cudaEvent_t start, stop;
  float elapsed_ms = 0.0f;
  if (debug) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, state.stream));
  }

  OPTIX_CHECK(
      optixLaunch(state.pipeline, state.stream, d_params, sizeof(Params), &state.sbt, params.width, params.height, 1));

  if (debug) {
    CUDA_CHECK(cudaEventRecord(stop, state.stream));
    CUDA_CHECK(cudaEventSynchronize(stop)); // Wait for both the launch and the stop event
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
  // Wait complete
  CUDA_CHECK(cudaStreamSynchronize(state.stream));

  torch::Tensor gpu_time = torch::zeros({1}, float_opts);
  if (debug) {
    gpu_time.fill_(elapsed_ms);
  }

  FwdTraceResult res;
  res.data = std::make_tuple(out_color, out_others, gpu_time, scaling, density_factor, max_blending);
  return res;
}

BwdTraceResult OptixTracer::trace_bwd(
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
    const int sh_degree,
    const Tensor &opacities,
    const Tensor &dL_dpixels,
    const Tensor &dL_dothers,
    const Tensor &scaling,
    const bool debug) {
  TracerState &state = this->bwdState;

  auto H = ray_o.size(0);
  auto W = ray_o.size(1);

  Params params;
  // Use gas in fwdstate
  params.gas_handle = this->fwdState.gas_handle;
  params.width = W;
  params.height = H;

  params.num_points_per_triangle = num_points_per_triangle.contiguous().data_ptr<int>();
  params.cumsum_of_points_per_triangle = cumsum_of_points_per_triangle.contiguous().data_ptr<int>();

  params.ray_org = reinterpret_cast<float3 *>(ray_o.contiguous().data_ptr<float>());
  params.ray_dir = reinterpret_cast<float3 *>(ray_d.contiguous().data_ptr<float>());
  params.vertices = reinterpret_cast<float3 *>(triangles_points.contiguous().data_ptr<float>());
  params.shs = shs.contiguous().data_ptr<float>();
  params.sh_degree = sh_degree;
  params.opacities = opacities.contiguous().data_ptr<float>();
  params.sigma = sigma.contiguous().data_ptr<float>();
  params.out_color = colors.contiguous().data_ptr<float>();
  params.out_others = others.contiguous().data_ptr<float>();
  params.scaling = scaling.contiguous().data_ptr<float>();
  params.bg_color = background.contiguous().data_ptr<float>();
  const int P = number_of_points;
  params.P = P;

  /**********************
  backward parameters
  ***********************/
  params.dL_dpixels = dL_dpixels.contiguous().data_ptr<float>();
  params.dL_dothers = dL_dothers.contiguous().data_ptr<float>();

  int M = 0;
  if (shs.size(0) != 0) {
    M = shs.size(1);
  }
  params.M = M;
  const int total_nb_points =
      num_points_per_triangle[P - 1].item<int>() + cumsum_of_points_per_triangle[P - 1].item<int>();

  torch::Tensor dL_dtriangle = torch::zeros({total_nb_points, 3}, triangles_points.options());
  torch::Tensor dL_dsigma = torch::zeros_like(sigma);
  torch::Tensor dL_dopacity = torch::zeros_like(opacities);
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, triangles_points.options());

  params.out_dL_dtriangle = dL_dtriangle.contiguous().data_ptr<float>();
  params.out_dL_dsigma = dL_dsigma.contiguous().data_ptr<float>();
  params.out_dL_dopacity = dL_dopacity.contiguous().data_ptr<float>();
  params.out_dL_dsh = dL_dsh.contiguous().data_ptr<float>();

  CUdeviceptr d_params;
  // Copy params after init
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice));

  OPTIX_CHECK(
      optixLaunch(state.pipeline, state.stream, d_params, sizeof(Params), &state.sbt, params.width, params.height, 1));

  CUDA_CHECK(cudaStreamSynchronize(state.stream));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));

  BwdTraceResult res;
  res.data = std::make_tuple(dL_dtriangle, dL_dsigma, dL_dopacity, dL_dsh);
  return res;
}
