/**
 * @file	CUDASizedMemCache.h
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

#ifndef CUDASIZEDMEMCACHE_H_
#define CUDASIZEDMEMCACHE_H_

#include <cstddef>

namespace jetson_tensorrt {

/**
 * @brief	Abstract class of a cached CUDA allocation of a specific size
 */
class CUDASizedMemCache {
public:

	/**
	 * @brief	CUDASizedMemCache constructor
	 */
	CUDASizedMemCache();

	/**
	 * @brief	CUDASizedMemCache destructor
	 */
	virtual ~CUDASizedMemCache();

	/**
	 * @brief	Gets a memory allocation from the cache
	 * The allocation will be reused between calls if size remains the same
	 * If the size changes, the original allocation will be freed and a new one allocated
	 * @param	size	Size of the requested memory allocation
	 * @return	Pointer/handle to the device memory
	 */
	void* getCUDAAlloc(size_t size);

	/**
	 * @brief	Frees the allocated cache memory
	 */
	void freeCUDAAlloc();

private:
	void* outputAlloc;
	size_t outputAllocSize;
};

} /* namespace jetson_tensorrt */

#endif /* CUDASIZEDMEMCACHE_H_ */
