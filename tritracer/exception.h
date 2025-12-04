/*
 Copyright (c) 2025 Changhe Liu 
 Licensed under the Apache License, Version 2.0. 

 For inquiries contact lch01234@gmail.com
*/
#pragma once

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include <optix.h>
#include <sstream>
#include <stdexcept>
#include <string>

class Exception : public std::runtime_error {
public:
  Exception(const char *msg) : std::runtime_error(msg) {
  }

  Exception(OptixResult res, const char *msg) : std::runtime_error(createMessage(res, msg).c_str()) {
  }

private:
  std::string createMessage(OptixResult res, const char *msg) {
    std::ostringstream out;
    out << optixGetErrorName(res) << ": " << msg;
    return out.str();
  }
};

inline void cudaCheck(cudaError_t error, const char *call, const char *file, unsigned int line) {
  if (error != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA call (" << call << " ) failed with error: '" << cudaGetErrorString(error) << "' (" << file << ":"
       << line << ")\n";
    throw Exception(ss.str().c_str());
  }
}

inline void optixCheck(OptixResult res, const char *call, const char *file, unsigned int line) {
  if (res != OPTIX_SUCCESS) {
    std::stringstream ss;
    ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
    throw Exception(res, ss.str().c_str());
  }
}

#define CUDA_CHECK(call) cudaCheck(call, #call, __FILE__, __LINE__)

#define OPTIX_CHECK(call) optixCheck(call, #call, __FILE__, __LINE__)
