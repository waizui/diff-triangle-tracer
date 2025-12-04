#
# Copyright (c) 2025 Changhe Liu 
# Licensed under the Apache License, Version 2.0. 
#
# For inquiries contact lch01234@gmail.com
#

cmake_minimum_required(VERSION 3.20)

# for neovim lsp
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(TriTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add path to our CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Use torch from conda, python3.11 only
set(TORCH_ROOT "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/torch")
set(TORCH_INCLUDE_DIRS
    ${TORCH_ROOT}/include
    ${TORCH_ROOT}/include/torch/csrc/api/include
)
set(TORCH_LIBRARIES
    ${TORCH_ROOT}/lib/libtorch_python.so
    ${TORCH_ROOT}/lib/libtorch_cuda.so
    ${TORCH_ROOT}/lib/libtorch_cpu.so
    ${TORCH_ROOT}/lib/libc10_cuda.so
    ${TORCH_ROOT}/lib/libc10.so
)

# We use this to be able to link against CUDA::cudart and/or CUDA::cuda_driver
find_package(CUDAToolkit REQUIRED)
# Make sure we have access to Optix7
find_package(OptiX7 REQUIRED)

# Utility function to get PTX compilation & copying working
include(PTXUtilities)


#TODO: conditional compile executable
add_library(tritracer
	tritracer/exception.h
	tritracer/pipeline.cpp
	tritracer/tracer.h
	tritracer/tracer.cpp
)

# Use target_link_libraries to link against all dependencies.
# This automatically handles include paths and other settings.
target_link_libraries(tritracer
    PUBLIC
    ${TORCH_LIBRARIES}
    CUDA::cudart_static
    Python3::Python
)

# Explicitly add all necessary include directories
# This is where you manually add the OptiX include folder that you saw in the log.
target_include_directories(tritracer
    PUBLIC 
    ${OptiX7_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${TORCH_INCLUDE_DIRS}
)


add_subdirectory(tritracer)
