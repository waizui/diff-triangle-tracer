/*
 Copyright (c) 2025 Changhe Liu 
 Licensed under the Apache License, Version 2.0. 

 For inquiries contact lch01234@gmail.com
*/

#pragma once

#include "auxiliary.h"
#include "vecmath.h"
#include <math.h>

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color
static __device__ __inline__ float3
computeColorFromSH(const int idx, const int deg, const int max_coeffs, float3 dir, const float *shs, bool clamped[3]) {
  // The implementation is loosely based on code for
  // "Differentiable Point-Based Radiance Fields for
  // Efficient View Synthesis" by Zhang et al. (2022)
  dir = dir / length(dir);

  const float3 *sh = reinterpret_cast<const float3 *>(shs) + idx * max_coeffs;
  float3 result = SH_C0 * sh[0];

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] + SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
               SH_C2[3] * xz * sh[7] + SH_C2[4] * (xx - yy) * sh[8];

      if (deg > 2) {
        result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[9] + SH_C3[1] * xy * z * sh[10] +
                 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] + SH_C3[5] * z * (xx - yy) * sh[14] +
                 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
      }
    }
  }

  result = result + 0.5f;
  if (clamped != nullptr) {
    clamped[0] = (result.x < 0);
    clamped[1] = (result.y < 0);
    clamped[2] = (result.z < 0);
  }
  return make_float3(max(result.x, 0.f), max(result.y, 0.f), max(result.z, 0.f));
}

static __device__ __inline__ void computeWindowFunc(
    const int prim_idx,
    const int prim_vert_offset,
    const float3 *vertices,
    const float3 hit_p,
    const float *sigmas,
    float &Cx,
    float3 &face_normal,
    float &phi_final,
    float &phi_max,
    float distances[3]) {

  // Compute in world space instead of screen space
  float3 A = vertices[prim_vert_offset];
  float3 B = vertices[prim_vert_offset + 1];
  float3 C = vertices[prim_vert_offset + 2];

  float a = length(B - C);
  float b = length(A - C);
  float c = length(A - B);

  float3 incenter = (a * A + b * B + c * C) / (a + b + c);
  // In triangle-splatting there is "4. Flip the normal if needed (ensure it faces the camera)"
  // it might causing by incorrect subdivision vertives arrangement, thus not needed in tracer.
  // https://github.com/trianglesplatting/triangle-splatting/issues/38
  float3 edge_cross = cross(B - A, C - A);
  face_normal = normalize(edge_cross);
  float dist = INFINITY;
  float phi = -INFINITY;

  for (int i = 0; i < 3; i++) {
    float3 p1 = vertices[prim_vert_offset + i];
    float3 p2 = vertices[prim_vert_offset + (i + 1) % 3];

    float3 edge_normal = normalize(cross(edge_cross, p1 - p2));

    float offset = -dot(edge_normal, p1);
    if (dist == INFINITY) {
      dist = dot(edge_normal, incenter) + offset;
      // Make edge_normal inward if dist>0 for compatibility, lowest cost operation
      if (dist > 0) {
        edge_normal = -edge_normal;
        offset = -offset;
        dist = -dist;
      }
    }

    float edge_dist = dot(edge_normal, hit_p) + offset;
    phi = fmaxf(phi, edge_dist);
    if (distances != nullptr) {
      distances[i] = edge_dist;
    }
  }

  phi_max = phi;
  phi_final = phi / dist;
  Cx = fmaxf(0.0f, __powf(phi_final, sigmas[prim_idx]));
}

static __device__ __inline__ bool
hit_sphere(const float r, const float3 cnt, float3 &ray_org, float3 &ray_dir, float &t) {
  // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
  const float3 op = ray_org - cnt;
  const float a = dot(ray_dir, ray_dir);
  if (a <= 0.0f) {
    return false;
  }
  const float b = dot(op, ray_dir);
  const float c = dot(op, op) - r * r;
  float det = b * b - c * a;
  if (det < 0.0f) {
    return false;
  }

  det = sqrtf(det);
  t = (-b - det) / a;
  if (t < STEP_EPSILON) {
    t = (-b + det) / a;
    if (t < STEP_EPSILON) {
      return false;
    }
  }

  return true;
}

static __device__ __inline__ void
extract_sphere(const uint8_t *extra_param, const uint32_t extra_len, float &r, float3 &cnt) {
  const float *extra = reinterpret_cast<const float *>(extra_param);
  r = extra[0];
  cnt = float3{extra[1], extra[2], extra[3]};
}
