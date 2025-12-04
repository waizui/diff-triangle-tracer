#
# Copyright (c) 2025 Changhe Liu 
# Licensed under the Apache License, Version 2.0. 
#
# For inquiries contact lch01234@gmail.com
#

import os
import glob, shutil
import sysconfig
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIX_HOME = os.path.join(ROOT_DIR, "third", "optix")
OPTIX_INCLUDE = os.path.join(OPTIX_HOME, "include")

class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        super().build_extensions()

        print("Copying ptx files...")
        
        # Compile ptx files
        build_res = os.system(f"{ROOT_DIR}/build.sh")
        if build_res != 0:
            raise ValueError("build.sh failed")

        pkg_target = sysconfig.get_path("purelib") + "/diff_tritracer"
        os.makedirs(pkg_target, exist_ok=True)

        # Copy the ptx files to the python package
        ptx_files = glob.glob(os.path.join(ROOT_DIR, "build", "ptx", "*.ptx"))

        if len(ptx_files) == 0:
            raise ValueError("no ptx found")

        for ptx_file in ptx_files:
            shutil.copy(ptx_file, pkg_target)
            print(f"copied {ptx_file} to {pkg_target}")

        print(f"{len(ptx_files)} ptx files copied")


setup(
    name="diff_tritracer",
    packages=["diff_tritracer"],
    ext_modules=[
        CUDAExtension(
            name="diff_tritracer._C",
            sources=[
                "ext.cpp",
                "tritracer/pipeline.cpp",
                "tritracer/tracer.cpp",
            ],
            include_dirs=[
                OPTIX_INCLUDE,
            ],
        ),
    ],
    cmdclass={"build_ext": CustomBuildExtension},
)
