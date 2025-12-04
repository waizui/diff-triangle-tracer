/*
 Copyright (c) 2025 Changhe Liu
 Licensed under the Apache License, Version 2.0.

 For inquiries contact lch01234@gmail.com
*/

// Cloest k volumertic tracing algorithm in paper: https://gaussiantracer.github.io/
#pragma once

#include "config.h"
#include "optix_device.h"
#include "params.h"

using RayPayload = RayHit[CHUNK_SIZE];

static __device__ __inline__ void trace(
    OptixTraversableHandle gas_handle,
    const float3 &ray_org,
    const float3 &ray_dir,
    RayPayload &payload,
    float min_t,
    float max_t) {
  uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23,
      r24, r25, r26, r27, r28, r29, r30, r31;
  r0 = r2 = r4 = r6 = r8 = r10 = r12 = r14 = r16 = r18 = r20 = r22 = r24 = r26 = r28 = r30 = RayHit::invalid_id;
  r1 = r3 = r5 = r7 = r9 = r11 = r13 = r15 = r17 = r19 = r21 = r23 = r25 = r27 = r29 = r31 =
      __float_as_int(RayHit::infinite_dist);
  optixTrace(
      gas_handle,
      ray_org,
      ray_dir,
      min_t,
      max_t,
      0.0f,
      OptixVisibilityMask(0xFF),
      OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_NONE,
      0, // SBT offset
      0, // SBT stride
      0, // MissSBT Index
      r0,
      r1,
      r2,
      r3,
      r4,
      r5,
      r6,
      r7,
      r8,
      r9,
      r10,
      r11,
      r12,
      r13,
      r14,
      r15,
      r16,
      r17,
      r18,
      r19,
      r20,
      r21,
      r22,
      r23,
      r24,
      r25,
      r26,
      r27,
      r28,
      r29,
      r30,
      r31);

  payload[0].prim_index = r0;
  payload[0].t = __uint_as_float(r1);
  payload[1].prim_index = r2;
  payload[1].t = __uint_as_float(r3);
  payload[2].prim_index = r4;
  payload[2].t = __uint_as_float(r5);
  payload[3].prim_index = r6;
  payload[3].t = __uint_as_float(r7);
  payload[4].prim_index = r8;
  payload[4].t = __uint_as_float(r9);
  payload[5].prim_index = r10;
  payload[5].t = __uint_as_float(r11);
  payload[6].prim_index = r12;
  payload[6].t = __uint_as_float(r13);
  payload[7].prim_index = r14;
  payload[7].t = __uint_as_float(r15);
  payload[8].prim_index = r16;
  payload[8].t = __uint_as_float(r17);
  payload[9].prim_index = r18;
  payload[9].t = __uint_as_float(r19);
  payload[10].prim_index = r20;
  payload[10].t = __uint_as_float(r21);
  payload[11].prim_index = r22;
  payload[11].t = __uint_as_float(r23);
  payload[12].prim_index = r24;
  payload[12].t = __uint_as_float(r25);
  payload[13].prim_index = r26;
  payload[13].t = __uint_as_float(r27);
  payload[14].prim_index = r28;
  payload[14].t = __uint_as_float(r29);
  payload[15].prim_index = r30;
  payload[15].t = __uint_as_float(r31);
}

#define CALL_GET_PAYLOAD(n) optixGetPayload_##n()
#define CALL_SET_PAYLOAD(n, val) optixSetPayload_##n(val)

#define compareAndSwapHitPayloadValue(hit, i_id, i_distance)                                                           \
  {                                                                                                                    \
    const float distance = __uint_as_float(CALL_GET_PAYLOAD(i_distance));                                              \
    if (hit.t < distance) {                                                                                            \
      CALL_SET_PAYLOAD(i_distance, __float_as_uint(hit.t));                                                            \
      const uint32_t id = CALL_GET_PAYLOAD(i_id);                                                                      \
      CALL_SET_PAYLOAD(i_id, hit.prim_index);                                                                          \
      hit.t = distance;                                                                                                \
      hit.prim_index = id;                                                                                             \
    }                                                                                                                  \
  }

static __device__ __inline__ void anyhit() {
  RayHit hit = RayHit{optixGetPrimitiveIndex(), optixGetRayTmax()};

  if (hit.t < __uint_as_float(optixGetPayload_31())) {
    compareAndSwapHitPayloadValue(hit, 0, 1);
    compareAndSwapHitPayloadValue(hit, 2, 3);
    compareAndSwapHitPayloadValue(hit, 4, 5);
    compareAndSwapHitPayloadValue(hit, 6, 7);
    compareAndSwapHitPayloadValue(hit, 8, 9);
    compareAndSwapHitPayloadValue(hit, 10, 11);
    compareAndSwapHitPayloadValue(hit, 12, 13);
    compareAndSwapHitPayloadValue(hit, 14, 15);
    compareAndSwapHitPayloadValue(hit, 16, 17);
    compareAndSwapHitPayloadValue(hit, 18, 19);
    compareAndSwapHitPayloadValue(hit, 20, 21);
    compareAndSwapHitPayloadValue(hit, 22, 23);
    compareAndSwapHitPayloadValue(hit, 24, 25);
    compareAndSwapHitPayloadValue(hit, 26, 27);
    compareAndSwapHitPayloadValue(hit, 28, 29);
    compareAndSwapHitPayloadValue(hit, 30, 31);

    // ignore all inserted hits, expect if the last one
    if (__uint_as_float(optixGetPayload_31()) > optixGetRayTmax()) {
      optixIgnoreIntersection();
    }
  }
}
