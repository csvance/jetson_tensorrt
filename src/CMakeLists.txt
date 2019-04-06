find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_53,code=sm_53
    -gencode arch=compute_62,code=sm_62
)

if(CUDA_VERSION_MAJOR GREATER 9)
	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS};
		-gencode arch=compute_72,code=sm_72
	)
endif()

# file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(
  tensorrt
  tensorrt/cuda
  /usr/local/cuda/include
)

add_subdirectory(tensorrt)
add_subdirectory(nodes)
