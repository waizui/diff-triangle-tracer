/*
 Copyright (c) 2025 Changhe Liu 
 Licensed under the Apache License, Version 2.0. 

 For inquiries contact lch01234@gmail.com
*/
#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <stdint.h>    


// tracer params
struct Params {
  // optix gas handle in state
  OptixTraversableHandle gas_handle;

  unsigned int width;
  unsigned int height;

  float3 *ray_org;
  float3 *ray_dir;

  unsigned int P; // primitive number
  unsigned int sh_degree;
  // shs count = (degree+1)^2
  unsigned int M;

  float3 *vertices;
  int *num_points_per_triangle;
  int *cumsum_of_points_per_triangle;

  float *cols; // 0-th deg color
  float *shs;
  float *opacities;
  float *sigma;

  float *out_color;      //{3,H,W}
  float *out_others;     // {3+3+1,H,W}
  float *max_blending;   // {N}
  float *density_factor; // {N}

  float *scaling;
  float *bg_color;

  /**********************
  backward
  ***********************/

  float *dL_dpixels;
  float *dL_dothers;

  float *out_dL_dtriangle;
  float *out_dL_dopacity;
  float *out_dL_dsigma;
  float *out_dL_dsh;

  // extra parameters
  uint8_t *extra;
  uint32_t extra_len;
};

// Define the primitive info
struct RayHit {
  uint32_t prim_index; // intersection primitive ID
  float t;             // t range along the ray
  static constexpr float infinite_dist = 1e6f;
  static constexpr unsigned int invalid_id = 0xFFFFFFFF;
};
