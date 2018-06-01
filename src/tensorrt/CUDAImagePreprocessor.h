/**
 * @file	CUDAImagePreprocessor.h
 * @author	Carroll Vance
 * @brief	Abstract class representing a CUDA accelerated image preprocessor
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

#ifndef IMAGEPREPROCESSOR_H_
#define IMAGEPREPROCESSOR_H_

#include <cstddef>

#include "CUDASizedMemCache.h"

namespace jetson_tensorrt {

/**
 * @brief	Abstract class representing a CUDA accelerated image preprocessor
 */
class CUDAImagePreprocessor {
public:
	CUDAImagePreprocessor();
	virtual ~CUDAImagePreprocessor();

	/**
	 * @brief	DEVICE -> DEVICE conversion and resizing of an RBGA image to BGR
	 * @param	inputWidth	The width of the input image
	 * @param	inputHeight	The height of the input image
	 * @param	outputWidth	The width of the converted BGR image
	 * @param	outputHeight	The height of the converted BGR image
	 * @return	Device pointer to the BGR image
	 */
	virtual float* RBGAtoBGR(size_t inputWidth, size_t inputHeight,
			size_t outputWidth, size_t outputHeight) = 0;

	/**
	 * @brief	Loads host memory into the preprocessors input
	 * @param	hostMemory	Pointer to the memory on the host
	 * @param	size	Size of the memory on the host
	 */
	void inputFromHost(void* hostMemory, size_t size);

protected:
	CUDASizedMemCache inputCache;
	CUDASizedMemCache outputCache;

};

} /* namespace jetson_tensorrt */

#endif /* IMAGEPREPROCESSOR_H_ */

