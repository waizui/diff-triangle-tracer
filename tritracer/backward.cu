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
#include "vector_functions.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_types.h>

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void addCoeffGrd(float3 *coefficientsGrad, int idx, const float3 &val) {
  atomicAdd(&coefficientsGrad[idx].x, val.x);
  atomicAdd(&coefficientsGrad[idx].y, val.y);
  atomicAdd(&coefficientsGrad[idx].z, val.z);
}

// Backward pass for conversion of spherical harmonics to RGB for
// each Triangle.
__device__ __inline__ void computeColorFromSHBwd(
    const int idx, // primitive index
    const int deg,
    const int max_coeffs,
    const float3 dir_orig,
    const float *shs,
    const bool *clamped,
    const float dL_dcolor[3],
    float3 *dL_dshs) {

  float3 dir = dir_orig / length(dir_orig);
  float3 dL_dRGB = make_float3(dL_dcolor[0], dL_dcolor[1], dL_dcolor[2]);
  dL_dRGB.x *= clamped[0] ? 0 : 1;
  dL_dRGB.y *= clamped[1] ? 0 : 1;
  dL_dRGB.z *= clamped[2] ? 0 : 1;

  float x = dir.x;
  float y = dir.y;
  float z = dir.z;

  // Target location for this Gaussian to write SH gradients to
  float3 *dL_dsh = dL_dshs + idx * max_coeffs;

  // No tricks here, just high school-level calculus.
  float dRGBdsh0 = SH_C0;
  addCoeffGrd(dL_dsh, 0, dRGBdsh0 * dL_dRGB); // dL_dsh0
  if (deg > 0) {
    float dRGBdsh1 = -SH_C1 * y;
    float dRGBdsh2 = SH_C1 * z;
    float dRGBdsh3 = -SH_C1 * x;
    addCoeffGrd(dL_dsh, 1, dRGBdsh1 * dL_dRGB);
    addCoeffGrd(dL_dsh, 2, dRGBdsh2 * dL_dRGB);
    addCoeffGrd(dL_dsh, 3, dRGBdsh3 * dL_dRGB);

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;

      float dRGBdsh4 = SH_C2[0] * xy;
      float dRGBdsh5 = SH_C2[1] * yz;
      float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
      float dRGBdsh7 = SH_C2[3] * xz;
      float dRGBdsh8 = SH_C2[4] * (xx - yy);

      addCoeffGrd(dL_dsh, 4, dRGBdsh4 * dL_dRGB);
      addCoeffGrd(dL_dsh, 5, dRGBdsh5 * dL_dRGB);
      addCoeffGrd(dL_dsh, 6, dRGBdsh6 * dL_dRGB);
      addCoeffGrd(dL_dsh, 7, dRGBdsh7 * dL_dRGB);
      addCoeffGrd(dL_dsh, 8, dRGBdsh8 * dL_dRGB);

      if (deg > 2) {
        float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
        float dRGBdsh10 = SH_C3[1] * xy * z;
        float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
        float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
        float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
        float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);

        addCoeffGrd(dL_dsh, 9, dRGBdsh9 * dL_dRGB);
        addCoeffGrd(dL_dsh, 10, dRGBdsh10 * dL_dRGB);
        addCoeffGrd(dL_dsh, 11, dRGBdsh11 * dL_dRGB);
        addCoeffGrd(dL_dsh, 12, dRGBdsh12 * dL_dRGB);
        addCoeffGrd(dL_dsh, 13, dRGBdsh13 * dL_dRGB);
        addCoeffGrd(dL_dsh, 14, dRGBdsh14 * dL_dRGB);
        addCoeffGrd(dL_dsh, 15, dRGBdsh15 * dL_dRGB);
      }
    }
  }
}

__device__ __inline__ void computeWindowFuncBwd(
    const float3 ray_org,
    const float3 ray_dir,
    const int prim_idx,
    const int prim_vert_offset,
    const float3 *vertices,
    const float3 hit_p,
    const float dL_dphi_x,
    const float phi_x,
    const float *distances,    // Distance of each edge to hit p
    const float3 dL_dnormal3D, // Face normal
    float *dL_dsigma,
    float3 *dL_dtriangle,
    const int pix_id) {

  const float3 *vert = vertices + prim_vert_offset;

#pragma unroll
  // dL_dphi_x gradient w.r.t vertex
  for (int k = 0; k < 3; k++) {
    // gradient only flows to max value edge
    if (fabsf(distances[k] - phi_x) > 1e-6f) {
      continue;
    }

    float3 p1 = vert[k];
    float3 p2 = vert[(k + 1) % 3];
    float3 p3 = vert[(k + 2) % 3];

    float3 n = cross(p1 - p3, p2 - p3);

    float3 ne = cross(n, p1 - p2);
    float3 Ne = normalize(ne);

    float dNe_dne[9];
    dnormvdv(Ne, ne, dNe_dne);

    float3 dL_dNe = dL_dphi_x * (hit_p - p1);
    float3 dL_dne = vecmat3x3(dL_dNe, dNe_dne);

    float dne_dp2[9];
    float dne_dp3[9];
    float dne_dp1[9];
    dnorm_dp1(p1, p2, p3, dne_dp1);
    dnorm_dp2(p1, p2, p3, dne_dp2);
    dnorm_dp3(p1, p2, p3, dne_dp3);

    float3 dL_dp1 = vecmat3x3(dL_dne, dne_dp1) - dL_dphi_x * Ne;
    float3 dL_dp2 = vecmat3x3(dL_dne, dne_dp2);
    float3 dL_dp3 = vecmat3x3(dL_dne, dne_dp3);

    addCoeffGrd(dL_dtriangle, prim_vert_offset + k, dL_dp1);
    addCoeffGrd(dL_dtriangle, prim_vert_offset + (k + 1) % 3, dL_dp2);
    addCoeffGrd(dL_dtriangle, prim_vert_offset + (k + 2) % 3, dL_dp3);
  }

  // Normal gradient w.r.t vertices
  {
    float3 p1 = vert[0];
    float3 p2 = vert[1];
    float3 p3 = vert[2];

    float3 n = cross(p1 - p3, p2 - p3);
    float3 N = normalize(n);
    float dN_dn[9];
    dnormvdv(N, n, dN_dn);

    float3 dL_dn = vecmat3x3(dL_dnormal3D, dN_dn);

    float dn_dp1[9];
    float dn_dp2[9];
    float dn_dp3[9];

    skew3x3(p3 - p2, dn_dp1); // -skew(p2-p3)
    skew3x3(p1 - p3, dn_dp2);
    skew3x3(p2 - p1, dn_dp3);

    float3 dL_dp1 = vecmat3x3(dL_dn, dn_dp1);
    float3 dL_dp2 = vecmat3x3(dL_dn, dn_dp2);
    float3 dL_dp3 = vecmat3x3(dL_dn, dn_dp3);

    addCoeffGrd(dL_dtriangle, prim_vert_offset + 0, dL_dp1);
    addCoeffGrd(dL_dtriangle, prim_vert_offset + 1, dL_dp2);
    addCoeffGrd(dL_dtriangle, prim_vert_offset + 2, dL_dp3);
  }
}

__device__ __inline__ void traceChunkBwd(
    const int pix_id,
    const OptixTraversableHandle &gas_handle,
    const int W,
    const int H,
    const float3 ray_org,
    const float3 ray_dir,
    const float *out_color,
    const float *out_others,
    const float3 *vertices,
    const int *num_points_per_triangle,
    const int *cumsum_of_points_per_triangle,
    const float *sigmas,
    const float *shs,
    const int sh_degree,
    const int max_coeffs,
    const float *opacities,
    const float *dL_dpixels,
    const float *dL_dothers,
    float *dL_dtriangle,
    float *dL_dopacity,
    float *dL_dsigma,
    float *dL_dsh) {

  float final_depth = out_others[pix_id + DEPTH_OFFSET * H * W];
  float3 final_normal = make_float3(
      out_others[pix_id + (NORMAL_OFFSET + 0) * H * W],
      out_others[pix_id + (NORMAL_OFFSET + 1) * H * W],
      out_others[pix_id + (NORMAL_OFFSET + 2) * H * W]);

  float final_color[3];
  for (int ch = 0; ch < 3; ch++)
    final_color[ch] = out_color[ch * H * W + pix_id];

  float dL_dpixel[3];
  for (int ch = 0; ch < 3; ch++)
    dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id];

  float dL_ddepth = dL_dothers[DEPTH_OFFSET * H * W + pix_id];

  float3 dL_dnormal2D = make_float3(
      dL_dothers[(NORMAL_OFFSET + 0) * H * W + pix_id],
      dL_dothers[(NORMAL_OFFSET + 1) * H * W + pix_id],
      dL_dothers[(NORMAL_OFFSET + 2) * H * W + pix_id]);

  float C[3] = {0.f};
  float N[3] = {0.f}; // normal
  float D = 0.f;      // depth

  uint32_t contributor = 0;
  float T = 1.0f;
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
      int prim_vert_offset = cumsum_for_triangle;

      float3 hit_p = ray_org + hit_t * ray_dir;
      float Cx = 1.f; // response
      float3 face_normal = {0};
      float phi_final = 0.f; // phi/dist
      float phi_x = 0.f;     // max phi
      float distances[3] = {0.f};
      computeWindowFunc(
          prim_idx, prim_vert_offset, vertices, hit_p, sigmas, Cx, face_normal, phi_final, phi_x, distances);

      const float opacity = opacities[prim_idx];

      // Eq. (2) from 3D Gaussian splatting paper.
      // Obtain alpha by multiplying with Gaussian opacity
      // and its exponential falloff from mean.
      // Avoid numerical instabilities (see paper appendix).
      float alpha = min(0.99f, opacity * Cx);
      if (alpha < 1.0f / 255.0f)
        continue;

      float test_T = T * (1 - alpha);
      if (test_T < 0.0001f) {
        return;
      }

      float blending_weight = alpha * T;

      bool clamped[3];
      float3 col = computeColorFromSH(prim_idx, sh_degree, max_coeffs, ray_dir, shs, clamped);
      blendf3(C, col, blending_weight);
      blendf3(N, face_normal, blending_weight);
      D += blending_weight * hit_t;

      float dL_dcolor[3];
      float c[3] = {col.x, col.y, col.z};
      float dL_dalpha = 0.0f;
      float inv_1_alpha = 1.0f / (1.0f - alpha);
      for (int ch = 0; ch < 3; ch++) {
        // C_pixel = C_prev+T*alpha*c + T(1-alpha)C_rem,
        // where C_prev is accumated color contribution, C_rem is remaining color contributon
        const float dL_dchannel = dL_dpixel[ch];
        dL_dcolor[ch] = dL_dchannel * blending_weight;
        dL_dalpha += dL_dchannel * (T * c[ch] - (final_color[ch] - C[ch]) * inv_1_alpha);
      }

      // w.r.t. depth: D_pixel = D_prev+T*alpha*D + T(1-alpha)D_rem,
      dL_dalpha += dL_ddepth * (T * hit_t - (final_depth - D) * inv_1_alpha);

      float3 Nf3 = {N[0], N[1], N[2]};
      dL_dalpha += sumf3(dL_dnormal2D * (T * face_normal - (final_normal - Nf3) * inv_1_alpha));

      // Update gradients w.r.t. opacity of the Triangle
      atomicAdd(&(dL_dopacity[prim_idx]), dL_dalpha * Cx);

      const float dL_dC = opacity * dL_dalpha;
      if (phi_final > 0.0f) {
        // Derivative with respect to sigma
        float dL_dsigma_value = dL_dC * Cx * __logf(phi_final);
        atomicAdd(&dL_dsigma[prim_idx], dL_dsigma_value);
      }

      // Gradient w.r.t phi_x
      float dL_dphi_x = sigmas[prim_idx] / phi_x * dL_dC * Cx; // dL_dc*(sigma*Cx/phi_x)

      float3 dL_dnormal = dL_dnormal2D * blending_weight;
      computeWindowFuncBwd(
          ray_org,
          ray_dir,
          prim_idx,
          prim_vert_offset,
          vertices,
          hit_p,
          dL_dphi_x,
          phi_x,
          distances,
          dL_dnormal,
          dL_dsigma,
          (float3 *)dL_dtriangle,
          pix_id);

      if (shs != nullptr) {
        computeColorFromSHBwd(prim_idx, sh_degree, max_coeffs, ray_dir, shs, clamped, dL_dcolor, (float3 *)dL_dsh);
      }

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

  traceChunkBwd(
      pix_id,
      params.gas_handle,
      W,
      H,
      ray_org,
      ray_dir,
      params.out_color,
      params.out_others,
      params.vertices,
      params.num_points_per_triangle,
      params.cumsum_of_points_per_triangle,
      params.sigma,
      params.shs,
      params.sh_degree,
      params.M,
      params.opacities,
      params.dL_dpixels,
      params.dL_dothers,
      params.out_dL_dtriangle,
      params.out_dL_dopacity,
      params.out_dL_dsigma,
      params.out_dL_dsh);
}
