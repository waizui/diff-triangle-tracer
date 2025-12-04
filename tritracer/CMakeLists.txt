# CMakeLists.txt for tritracer PTX kernels

# Create PTX targets using the utility function
add_ptx_targets(tritracer tritracer "forward;backward;reflect;refract;envlight")

# Add include directories to the PTX targets
target_include_directories(tritracer_forward PRIVATE 
    ${OptiX7_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_include_directories(tritracer_backward PRIVATE 
    ${OptiX7_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_include_directories(tritracer_reflect PRIVATE 
    ${OptiX7_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_include_directories(tritracer_refract PRIVATE 
    ${OptiX7_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_include_directories(tritracer_envlight PRIVATE 
    ${OptiX7_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
