/**
 * @file	CUDASizedMemCache.cpp
 * @author	Carroll Vance
 * @brief	Abstract class of a cached CUDA allocation of a specific size
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

#include <string>

#include <cuda_runtime.h>
#include <cuda.h>

#include "CUDASizedMemCache.h"
#include "RTCommon.h"
#include "RTExceptions.h"

namespace jetson_tensorrt {

/**
 * @brief	CUDASizedMemCache constructor
 */
CUDASizedMemCache::CUDASizedMemCache() {
	outputAlloc = nullptr;
	outputAllocSize = 0;
}

/**
 * @brief	CUDASizedMemCache destructor
 */
CUDASizedMemCache::~CUDASizedMemCache() {
	freeCUDAAlloc();
}

/**
 * @brief	Gets a memory allocation from the cache
 * The allocation will be reused between calls if size remains the same
 * If the size changes, the original allocation will be freed and a new one allocated
 * @param	size	Size of the requested memory allocation
 * @return	Pointer/handle to the device memory
 */
void* CUDASizedMemCache::getCUDAAlloc(size_t size) {
	if (outputAlloc && size == outputAllocSize)
		return outputAlloc;
	else if (outputAlloc && size != outputAllocSize) {
		freeCUDAAlloc();
		outputAlloc = safeCudaMalloc((size_t) size);
	} else {
		outputAlloc = safeCudaMalloc((size_t) size);
	}

	return outputAlloc;
}

/**
 * @brief	Frees the allocated cache memory
 */
void CUDASizedMemCache::freeCUDAAlloc() {
	if (outputAlloc) {
		cudaError_t deviceFreeError = cudaFree(outputAlloc);
		if (deviceFreeError != 0)
			throw DeviceMemoryFreeException(
					"Error freeing device memory. CUDA Error: "
							+ std::to_string(deviceFreeError));
	}

	outputAlloc = nullptr;
}

} /* namespace jetson_tensorrt */

