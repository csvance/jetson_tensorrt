project(ros_jetson_tensorrt)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Compiler options
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -std=c++11 -gencode arch=compute_62,code=sm_62
)

set(
    CMAKE_CXX_FLAGS
    ${CMAKE_CXX_FLAGS};
    -std=c++11
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(
  src/tensorrt
  src/tensorrt/cuda
  /usr/local/cuda/include
)

add_subdirectory(src)
