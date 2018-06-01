/**
 * @file	ImageNetPreprocessor.h
 * @author	Carroll Vance
 * @brief	Class representing a CUDA accelerated ImageNet image preprocessor
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

#ifndef IMAGENETPREPROCESSOR_H_
#define IMAGENETPREPROCESSOR_H_

#include <cuda_runtime.h>
#include <cuda.h>

#include "CUDAImagePreprocessor.h"

namespace jetson_tensorrt {

/**
 * @brief Class representing a CUDA accelerated ImageNet image preprocessor
 */
class ImageNetPreprocessor: public CUDAImagePreprocessor {
public:

	/**
	 * @brief	ImageNetPreprocessor default constructor. Does not use mean preprocessing.
	 */
	ImageNetPreprocessor();

	/**
	 * @brief	ImageNetPreprocessor constructor using a custom mean preprocessing value for all channels
	 */
	ImageNetPreprocessor(float3);

	/**
	 * @brief	ImageNetPreprocessor constructor using custom mean preprocessing values for each channel
	 */
	ImageNetPreprocessor(float);

	/**
	 * @brief	ImageNetPreprocesor destructor
	 */
	virtual ~ImageNetPreprocessor();

	/**
	 * @brief	Implements resizing and conversion of an RBGA image in device memory to BGR
	 * @param	inputWidth	The width of the input image
	 * @param	inputHeight	The height of the input image
	 * @param	outputWidth	The width of the converted BGR image
	 * @param	outputHeight	The height of the converted BGR image
	 * @return	Device pointer to the BGR image
	 */
	float* RBGAtoBGR(size_t inputWidth, size_t inputHeight, size_t outputWidth, size_t outputHeight);

	/**
	 * @brief	DEVICE -> DEVICE conversion and resizing of an NV12 image to RGBA
	 * @param	inputWidth	The width of the input image
	 * @param	inputHeight	The height of the input image
	 * @return	Device pointer to the RBGA image
	 */
	float* NV12toRGBA(size_t inputWidth, size_t inputHeight);


private:
	float3 mean;
};

} /* namespace jetson_tensorrt */

#endif /* IMAGENETPREPROCESSOR_H_ */
