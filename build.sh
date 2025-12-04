#!/bin/bash
pytorch_cmake_prefix="$(python - <<'EOF'
import torch
print(torch.utils.cmake_prefix_path, end="")
EOF
)"

mkdir -p build

cd ./build

cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DOPTIX_HOME="$(pwd)/../third/optix" \
  -DOptiX7_INCLUDE_DIR="$(pwd)/../third/optix/include" \
  -DCMAKE_PREFIX_PATH="${pytorch_cmake_prefix}"

make

mv ./compile_commands.json ../
