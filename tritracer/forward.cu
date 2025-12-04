/*
 Copyright (c) 2025 Changhe Liu 
 Licensed under the Apache License, Version 2.0. 

 For inquiries contact lch01234@gmail.com
*/
#include "auxiliary.h"
#include "closestk.h"
#include "common.h"
#include "optix_device.h"
#include "params.h"
#include "vecmath.h"
#include "vector_types.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_types.h>

extern "C" {
__constant__ Params params;
}

__device__ __inline__ float angleBetween(const float3 &a, const float3 &b) {
  float len_ab = length(a) * length(b);
  if (len_ab < 1e-12f)
    return 0.0f;

  float cosTheta = fmaxf(-1.0f, fminf(1.0f, dot(a, b) / len_ab));
  return acosf(cosTheta);
}

__device__ __inline__ void computeScaling(
    const int idx,
    const float3 *vertices,
    const int *num_points_per_triangle,
    const int *cumsum_of_points_per_triangle,
    const float *sigmas,
    const float3 ray_org,
    float *scaling) {

  /*
    This func assumes that all ray's origins are very close
    their sizes can be mesured in an unified metric
  */
  if (scaling[idx] > 0.0f) {
    return;
  }

  const int cumsum_for_triangle = cumsum_of_points_per_triangle[idx];
  const int offset = cumsum_for_triangle;
  const int num_points_triangle = num_points_per_triangle[idx];

  float3 center_triangle = {0.0f, 0.0f, 0.0f};

  for (int i = 0; i < num_points_triangle; i++) {
    center_triangle += vertices[offset + i];
  }
  center_triangle /= num_points_triangle;

  float3 tri2org = center_triangle - ray_org;

  float max_angle = 0.0f;

  for (int i = 0; i < num_points_triangle; i++) {
    float3 triangle_point = vertices[offset + i];

    float angle = angleBetween(triangle_point - ray_org, tri2org);
    max_angle = fmaxf(angle, max_angle);
  }

  scaling[idx] = max_angle;
}

__device__ __inline__ void traceChunk(
    const int pix_id,
    const OptixTraversableHandle &gas_handle,
    const float3 ray_org,
    const float3 ray_dir,
    const float3 *vertices,
    const int *num_points_per_triangle,
    const int *cumsum_of_points_per_triangle,
    const float *sigmas,
    const float *shs,
    const int sh_degree,
    const int max_coeffs,
    const float *opacities,
    float *density_factor,
    float *max_blending,
    float *scaling,
    float *C,
    float &T,
    float *N,
    float &D,
    uint32_t &contributor) {
  float last_hit_t = 0.f;
  RayPayload payload;
  while (true) {
    trace(gas_handle, ray_org, ray_dir, payload, last_hit_t, far_n);
    for (int i = 0; i < CHUNK_SIZE; i++) {
      RayHit hit = payload[i];
      if (hit.prim_index == RayHit::invalid_id) {
        return;
      }

      contributor++;

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

      density_factor[prim_idx] = 1; // Only check if hit, no need sync

      computeScaling(
          prim_idx, vertices, num_points_per_triangle, cumsum_of_points_per_triangle, sigmas, ray_org, scaling);

      float blending_weight = alpha * T;

      // Update the maximum blending weight in a thread-safe way
      atomicMax(((int *)max_blending) + prim_idx, *((int *)(&blending_weight)));

      blendf3(N, face_normal, blending_weight);
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

  uint32_t contributor = 0;
  // Initialize the volume rendering data
  float C[3] = {0};
  float T = 1.0f; // transmittance
  // Added from 2DGS
  float N[3] = {0}; // normal
  float D = 0.0f;   // depth

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
      params.density_factor,
      params.max_blending,
      params.scaling,
      C,
      T,
      N,
      D,
      contributor);

  // Blending bg color
  for (int ch = 0; ch < 3; ch++)
    params.out_color[ch * H * W + pix_id] = C[ch] + T * params.bg_color[ch];

  params.out_others[pix_id + DEPTH_OFFSET * H * W] = D;
  params.out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
  for (int ch = 0; ch < 3; ch++)
    params.out_others[pix_id + (NORMAL_OFFSET + ch) * H * W] = N[ch];

  // Currently not supported
  params.out_others[pix_id + MIDDEPTH_OFFSET * H * W] = 0.;
  params.out_others[pix_id + DISTORTION_OFFSET * H * W] = 0.;
}
