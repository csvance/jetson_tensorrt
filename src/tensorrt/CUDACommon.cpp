/**
 * @file	CUDACommon.cpp
 * @author	Carroll Vance
 * @brief	Wrappers for common CUDA functions
 *
 * Copyright (c) 2018 Carroll Vance.
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDACommon.h"

namespace jetson_tensorrt {

void *safeCudaHostMalloc(size_t memSize) {

  void *hostMem;
  cudaError_t cudaMallocError =
      cudaHostAlloc(&hostMem, memSize, cudaHostAllocMapped);
  if (cudaMallocError != 0) {
    throw std::runtime_error("CUDA Host Malloc Error: " +
                             std::to_string(cudaMallocError));
  } else if (hostMem == nullptr) {
    throw std::runtime_error("Out of device memory");
  }
}

void *safeCudaMalloc(size_t memSize) {
  void *deviceMem;

  cudaError_t cudaMallocError = cudaMalloc(&deviceMem, memSize);
  if (cudaMallocError != 0) {
    throw std::runtime_error("CUDA Malloc Error: " +
                             std::to_string(cudaMallocError));
  } else if (deviceMem == nullptr) {
    throw std::runtime_error("Out of device memory");
  }

  return deviceMem;
}

void *safeCudaUnifiedMalloc(size_t memSize) {
  void *deviceMem;

  cudaError_t cudaMallocManagedError = cudaMallocManaged(&deviceMem, memSize);
  if (cudaMallocManagedError != 0) {
    throw std::runtime_error("CUDA Malloc Error: " +
                             std::to_string(cudaMallocManagedError));
  } else if (deviceMem == nullptr) {
    throw std::runtime_error("Out of device memory");
  }

  return deviceMem;
}

void *safeMalloc(size_t memSize) {
  void *deviceMem;

  deviceMem = malloc(memSize);
  if (deviceMem == 0)
    throw std::bad_alloc();

  return deviceMem;
}

} // namespace jetson_tensorrt
