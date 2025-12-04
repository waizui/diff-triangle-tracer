/*
 Copyright (c) 2025 Changhe Liu 
 Licensed under the Apache License, Version 2.0. 

 For inquiries contact lch01234@gmail.com
*/
#include "auxiliary.h"
#include "closestk.h"
#include "common.h"
#include "params.h"
#include "vecmath.h"
#include "vector_types.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <optix.h>
#include <optix_types.h>

extern "C" {
__constant__ Params params;
}

__device__ __inline__ void traceChunk(
    const int pix_id,
    const OptixTraversableHandle &gas_handle,
    float3 ray_org,
    float3 ray_dir,
    const float3 *vertices,
    const int *num_points_per_triangle,
    const int *cumsum_of_points_per_triangle,
    const float *sigmas,
    const float *shs,
    const int sh_degree,
    const int max_coeffs,
    const float *opacities,
    const float r,
    const float3 cnt,
    float *C,
    float &T) {
  float sphere_t;
  if (hit_sphere(r, cnt, ray_org, ray_dir, sphere_t)) {
    ray_org += ray_dir * sphere_t;
    const float3 normal = normalize(ray_org - cnt);
    ray_org += normal * STEP_EPSILON;
    ray_dir = reflect(normalize(ray_dir), normal);
  }

  float last_hit_t = 0.f;
  RayPayload payload;
  while (true) {
    trace(gas_handle, ray_org, ray_dir, payload, last_hit_t, far_n);
    for (int i = 0; i < CHUNK_SIZE; i++) {
      RayHit hit = payload[i];
      if (hit.prim_index == RayHit::invalid_id) {
        return;
      }

      int prim_idx = hit.prim_index;
      float hit_t = hit.t;
      last_hit_t = fmaxf(last_hit_t + STEP_EPSILON, hit_t);
      if (hit_t < near_n)
        continue;

      int cumsum_for_triangle = cumsum_of_points_per_triangle[prim_idx];
      int prim_offset = cumsum_for_triangle;

      float3 hit_p = ray_org + hit_t * ray_dir;
      float Cx = 1.f; // response
      float3 face_normal = {0};
      float phi_final = 0.f; // phi/dist
      float phi_x = 0.f;     // max phi
      computeWindowFunc(prim_idx, prim_offset, vertices, hit_p, sigmas, Cx, face_normal, phi_final, phi_x, nullptr);

      // Eq. (2) from 3D Gaussian splatting paper.
      // Obtain alpha by multiplying with Gaussian opacity
      // and its exponential falloff from mean.
      // Avoid numerical instabilities (see paper appendix).
      float alpha = min(0.99f, opacities[prim_idx] * Cx);
      if (alpha < 1.0f / 255.0f)
        continue;

      float test_T = T * (1 - alpha);
      if (test_T < 0.0001f) {
        return;
      }

      float blending_weight = alpha * T;
      float3 col = computeColorFromSH(prim_idx, sh_degree, max_coeffs, ray_dir, shs, nullptr);
      blendf3(C, col, blending_weight);

      T = test_T;
    }
  }
}

extern "C" __global__ void __anyhit__ah() {
  anyhit();
}

extern "C" __global__ void __raygen__rg() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();

  int W = dim.x;
  int H = dim.y;

  uint32_t pix_id = idx.y * W + idx.x;

  float3 ray_org = params.ray_org[pix_id];
  float3 ray_dir = params.ray_dir[pix_id];

  float r = 1.0f;
  float3 cnt = float3{0.f, 0.f, 0.f};
  extract_sphere(params.extra, params.extra_len, r, cnt);

  float C[3] = {0};
  float T = 1.0f;

  traceChunk(
      pix_id,
      params.gas_handle,
      ray_org,
      ray_dir,
      params.vertices,
      params.num_points_per_triangle,
      params.cumsum_of_points_per_triangle,
      params.sigma,
      params.shs,
      params.sh_degree,
      params.M,
      params.opacities,
      r,
      cnt,
      C,
      T);

  // blending bg color
  for (int ch = 0; ch < 3; ch++)
    params.out_color[ch * H * W + pix_id] = C[ch] + T * params.bg_color[ch];
}
