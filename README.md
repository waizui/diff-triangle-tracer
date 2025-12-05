# Differentiable Triangle Tracer

This repository contains the OptiX implementation of the Differentiable Triangle Tracer used in UTrice: Unifying Primitives in Differentiable Ray Tracing and Rasterization via Triangles for Particle-Based 3D Scenes.

## Compilation

The tracer can be built using either the Conda toolchain or a manual setup. For the Conda workflow, please refer to the instructions in [UTrice](https://github.com/waizui/UTrice).

For manual compilation, you need to provide the libTorch include directories and libraries in CMakeLists.txt. Replace the following paths with your own configuration:

```bash
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
```

Then run build.sh.

## Citation
```
@misc{liu2025utriceunifyingprimitivesdifferentiable,
      title={UTrice: Unifying Primitives in Differentiable Ray Tracing and Rasterization via Triangles for Particle-Based 3D Scenes}, 
      author={Changhe Liu and Ehsan Javanmardi and Naren Bao and Alex Orsholits and Manabu Tsukada},
      year={2025},
      eprint={2512.04421},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.04421}, 
}
```

