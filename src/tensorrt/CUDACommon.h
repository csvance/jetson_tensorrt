/**
 * @file	CUDACommon.h
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

#ifndef TENSORRT_CUDA_CUDACOMMON_H_
#define TENSORRT_CUDA_CUDACOMMON_H_

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

namespace jetson_tensorrt {

/**
 * @brief	Allocates CUDA device memory or throws an exception
 * @param	memSize	The size of the requested memory allocation in bytes
 * @return	A handle corresponding to the device memory allocation.
 */
void *safeCudaMalloc(size_t memSize);

/**
 * @brief	Allocates host/device mapped device memory or throws an
 * exception
 * @param	memSize	The size of the requested memory allocation in bytes
 * @return	A handle corresponding to the device memory allocation.
 */
void *safeCudaHostMalloc(size_t memSize);

/**
 * @brief	Allocates unified device memory or throws an
 * exception
 * @param	memSize	The size of the requested memory allocation in bytes
 * @return	A handle corresponding to the device memory allocation.
 */
void *safeCudaUnifiedMalloc(size_t memSize);

/**
 * @brief	Allocates host memory or throws an exception
 * @param	memSize	The size of the requested memory allocation in bytes
 * @return	A handle corresponding to the device memory allocation.
 */
void *safeMalloc(size_t memSize);

enum MemoryLocation { HOST, DEVICE, MAPPED, UNIFIED, NONE };

} // namespace jetson_tensorrt

#endif /* TENSORRT_CUDA_CUDACOMMON_H_ */
