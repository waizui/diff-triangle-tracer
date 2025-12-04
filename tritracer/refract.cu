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
#include <optix.h>
#include <optix_types.h>

extern "C" {
__constant__ Params params;
}

__device__ __inline__ bool refract(float3 &out_dir, const float3 in_dir, float3 normal, const float eta_org) {
  // https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission#Refract
  // Incident direction is different from pbrt
  float cosTheta_i = dot(in_dir, normal);
  float eta = eta_org;
  if (cosTheta_i > 0.0) {
    eta = 1.0 / eta_org;
    normal = -normal;
  }

  float sin2Theta_i = max(0.0, 1.0 - sqrt(cosTheta_i));
  float sin2Theta_t = sin2Theta_i / sqrt(eta);

  if (sin2Theta_t >= 1)
    return false;

  float cosTheta_t = sqrt(max(0.0, 1.0 - sin2Theta_t));

  out_dir = in_dir / eta + (-cosTheta_i / eta - cosTheta_t) * normal;

  return true;
}

__device__ inline void refract_sphere(const float r, const float3 cnt, float3 &ray_org, float3 &ray_dir) {
  float t;
  if (!hit_sphere(r, cnt, ray_org, ray_dir, t)) {
    return;
  }

  const float3 hit_point = ray_org + ray_dir * t;
  float3 normal = normalize(hit_point - cnt);
  const float3 in_dir = normalize(ray_dir);

  const float n1 = 1.0003f;
  const float n2 = 1.5f;
  const float ior = n2 / n1;

  float3 new_dir = in_dir;
  if (!refract(new_dir, in_dir, normal, ior)) {
    float3 reflect_normal = dot(in_dir, normal) < 0.0f ? normal : -normal;
    new_dir = reflect(in_dir, reflect_normal);
    ray_org = hit_point + normal * STEP_EPSILON;
  } else {
    ray_org = hit_point + normal * STEP_EPSILON;
  }

  ray_dir = new_dir;
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

  refract_sphere(r, cnt, ray_org, ray_dir);

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
