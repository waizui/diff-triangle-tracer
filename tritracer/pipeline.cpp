/*
 Copyright (c) 2025 Changhe Liu 
 Licensed under the Apache License, Version 2.0. 

 For inquiries contact lch01234@gmail.com
*/

// Impl of Optix pipeline creation
#include "config.h"
#include "cuda_runtime_api.h"
#include "exception.h"
#include "optix_types.h"
#include "tracer.h"
#include <fstream>
#include <iostream>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <string>

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
  if (level > 3) {
    return;
  }
  std::cout << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

// Align x with y: (7,4) = ((10)/4 ) * 4 = 8
template <typename IntegerType>
__device__ __host__ IntegerType roundUp(IntegerType x, IntegerType y) {
  return ((x + y - 1) / y) * y;
}

void create_module(TracerState &state, const std::string &ptx) {
  OptixModuleCompileOptions module_compile_options = {};

#ifdef DEBUG
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

  state.pipeline_compile_options.usesMotionBlur = false;
  state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  state.pipeline_compile_options.numPayloadValues = 2 * CHUNK_SIZE; // use how many 32bit registers
  state.pipeline_compile_options.numAttributeValues = 2;            // primitive attributes words

#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant
             // performance cost and should only be done during development.
  state.pipeline_compile_options.exceptionFlags =
      OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
  state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
  state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
  state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  OPTIX_CHECK(optixModuleCreate(
      state.context,
      &module_compile_options,
      &state.pipeline_compile_options,
      ptx.c_str(),
      ptx.size(),
      nullptr, // logstring
      nullptr, // logstring size
      &state.ptx_module));
}

void create_program_group(TracerState &state, const char *raygen_name, const char *hitgroup_name) {
  OptixProgramGroupOptions program_group_options = {};

  // Raygen group
  OptixProgramGroupDesc raygen_prog_group_desc = {};
  raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygen_prog_group_desc.raygen.module = state.ptx_module;
  raygen_prog_group_desc.raygen.entryFunctionName = raygen_name;
  OPTIX_CHECK(optixProgramGroupCreate(
      state.context, &raygen_prog_group_desc, 1, &program_group_options, nullptr, nullptr, &state.raygen_prog_group));

  // Hitgroup program group
  OptixProgramGroupDesc hitgroup_prog_group_desc = {};
  hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroup_prog_group_desc.hitgroup.moduleCH = nullptr;
  hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
  hitgroup_prog_group_desc.hitgroup.moduleAH = state.ptx_module;
  hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = hitgroup_name;

  OPTIX_CHECK(optixProgramGroupCreate(
      state.context,
      &hitgroup_prog_group_desc,
      1,
      &program_group_options,
      nullptr,
      nullptr,
      &state.hitgroup_prog_group));

  // Miss, mandatory even not needed
  OptixProgramGroupDesc miss_prog_group_desc = {};
  miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  miss_prog_group_desc.miss.module = nullptr;
  miss_prog_group_desc.miss.entryFunctionName = nullptr;
  OPTIX_CHECK(optixProgramGroupCreate(
      state.context, &miss_prog_group_desc, 1, &program_group_options, nullptr, nullptr, &state.miss_prog_group));
}

void create_pipline(TracerState &state) {
  // Single shot, no bounce needed
  const uint32_t max_trace_depth = 1;

  // Pipeline
  OptixProgramGroup program_groups[] = {state.raygen_prog_group, state.hitgroup_prog_group};
  OptixPipelineLinkOptions pipeline_link_options = {};
  pipeline_link_options.maxTraceDepth = max_trace_depth;
  OPTIX_CHECK(optixPipelineCreate(
      state.context,
      &state.pipeline_compile_options,
      &pipeline_link_options,
      program_groups,
      sizeof(program_groups) / sizeof(program_groups[0]),
      nullptr,
      nullptr,
      &state.pipeline));
  // stack size
  OptixStackSizes stack_sizes = {};
  for (auto &prog_group : program_groups) {
    OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, state.pipeline));
  }

  uint32_t direct_callable_stack_size_from_traversal;
  uint32_t direct_callable_stack_size_from_state;
  uint32_t continuation_stack_size;

  // maxDCDepth =0, maxCCDepth=0, not using callables
  OPTIX_CHECK(optixUtilComputeStackSizes(
      &stack_sizes,
      max_trace_depth,
      0,
      0,
      &direct_callable_stack_size_from_traversal,
      &direct_callable_stack_size_from_state,
      &continuation_stack_size));

  OPTIX_CHECK(optixPipelineSetStackSize(
      state.pipeline,
      direct_callable_stack_size_from_traversal,
      direct_callable_stack_size_from_state,
      continuation_stack_size,
      1));
}

void create_sbt(TracerState &state) {
  // Raygen
  CUdeviceptr raygen_record;
  const size_t raygen_record_size = sizeof(RayGenSbtRecord);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
  RayGenSbtRecord rg_sbt;
  OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

  // Hit
  CUdeviceptr hitgroup_record;
  const size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
  HitGroupSbtRecord hg_sbt;
  OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_prog_group, &hg_sbt));
  CUDA_CHECK(
      cudaMemcpy(reinterpret_cast<void *>(hitgroup_record), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));

  state.sbt.raygenRecord = raygen_record;
  state.sbt.hitgroupRecordBase = hitgroup_record;
  state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
  state.sbt.hitgroupRecordCount = 1;

  // Miss, need to set even not needed
  CUdeviceptr miss_record;
  const size_t miss_record_size = sizeof(MissSbtRecord);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
  MissSbtRecord ms_sbt;
  OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(miss_record), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));
  state.sbt.missRecordBase = miss_record;
  state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
  state.sbt.missRecordCount = 1;
}

// Impl OptixTracer
std::string read_ptx(const std::string &pkgDir, const std::string &ptxName) {
  std::string path(pkgDir + "/" + ptxName);
  std::ifstream ptx_in(path.c_str());
  if (!ptx_in) {
    throw Exception((std::string("Failed to open PTX file ptx: ") + path).c_str());
  }

  std::string ptx((std::istreambuf_iterator<char>(ptx_in)), std::istreambuf_iterator<char>());
  return ptx;
}

void init_state(TracerState &state, std::string &ptx, char *raygen, char *anyhit) {
  create_module(state, ptx);
  create_program_group(state, raygen, anyhit);
  create_pipline(state);
  create_sbt(state);
  CUDA_CHECK(cudaStreamCreate(&state.stream));
}

// Set d_vertices, d_indexbuf before calling
void OptixTracer::build_tri_accel(TracerState &state, uint32_t numVertices, uint32_t numTri, bool rebuild) {

  OptixBuildInput triangle_input = {};
  triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangle_input.triangleArray.numVertices = numVertices;
  triangle_input.triangleArray.vertexBuffers = &state.d_vertices;
  triangle_input.triangleArray.numSbtRecords = 1;
  triangle_input.triangleArray.indexBuffer = state.d_indexbuf;
  triangle_input.triangleArray.numIndexTriplets = numTri;
  triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  triangle_input.triangleArray.indexStrideInBytes = sizeof(int3);

  uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
  triangle_input.triangleArray.flags = triangle_input_flags;

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  // Not adding those will causing misalign of params
  // accel_options.buildFlags |=
  //     OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS |
  //     OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  accel_options.operation = rebuild ? OPTIX_BUILD_OPERATION_BUILD : OPTIX_BUILD_OPERATION_UPDATE;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(
      state.context,
      &accel_options,
      &triangle_input,
      1, // Num_build_inputs
      &gas_buffer_sizes));

  if (state.gas_buffer_tmpe_size < gas_buffer_sizes.tempSizeInBytes) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_gas_buffer_temp)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_buffer_temp), gas_buffer_sizes.tempSizeInBytes));
    state.gas_buffer_tmpe_size = gas_buffer_sizes.tempSizeInBytes;
  }

  if (rebuild && (state.gas_buffer_size < gas_buffer_sizes.outputSizeInBytes)) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_gas_buffer)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_buffer), gas_buffer_sizes.outputSizeInBytes));
    state.gas_buffer_size = gas_buffer_sizes.outputSizeInBytes;
  }

  OPTIX_CHECK(optixAccelBuild(
      state.context,
      0, // CUDA stream
      &accel_options,
      &triangle_input,
      1, // Num build inputs
      state.d_gas_buffer_temp,
      gas_buffer_sizes.tempSizeInBytes,
      state.d_gas_buffer,
      gas_buffer_sizes.outputSizeInBytes,
      &state.gas_handle,
      nullptr, // Emitted property list
      0        // Num emitted properties
      ));
}

OptixTracer::OptixTracer(const std::string &pkgDir, const std::string &forward, const std::string &backward) {
  try {

    // Initialize CUDA
    this->context = nullptr;
    {
      CUDA_CHECK(cudaFree(0));

      OPTIX_CHECK(optixInit());

      OptixDeviceContextOptions options = {};

      options.logCallbackFunction = &context_log_cb;
      options.logCallbackLevel = 4;

      CUcontext cu_ctx = 0; // Zero means take the current context
      OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &this->context));
    }

    fwdState.context = this->context;
    bwdState.context = this->context;

    std::string ptxFwd = read_ptx(pkgDir, forward);
    std::string ptxBwd = read_ptx(pkgDir, backward);

    char rgName[] = "__raygen__rg";
    char ahName[] = "__anyhit__ah";

    init_state(fwdState, ptxFwd, rgName, ahName);
    init_state(bwdState, ptxBwd, rgName, ahName);

  } catch (const std::exception &e) {
    std::cerr << "OptixTracer construction failed: " << e.what() << std::endl;
    throw;
  }
}

void cleanup_state(TracerState &state) {
  // CUDA resources
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_gas_buffer)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_gas_buffer_temp)));
  CUDA_CHECK(cudaStreamDestroy(state.stream));
  // OptiX resources
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.hitgroupRecordBase)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.missRecordBase)));

  OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
  OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
  OPTIX_CHECK(optixProgramGroupDestroy(state.hitgroup_prog_group));
  OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
  OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
  state.context = nullptr;
}

OptixTracer::~OptixTracer() {
  cleanup_state(fwdState);
  cleanup_state(bwdState);
  OPTIX_CHECK(optixDeviceContextDestroy(this->context));
}
