/**
 * @file	ImageNetPreprocessor.cpp
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

#include <cuda_runtime.h>
#include <cuda.h>

#include "ImageNetPreprocessor.h"

#include "cudaImageNetPreprocess.h"

namespace jetson_tensorrt {


ImageNetPreprocessor::ImageNetPreprocessor() :
		CUDAImagePreprocessor() {
	this->mean.x = 0.0;
	this->mean.y = 0.0;
	this->mean.z = 0.0;
}


ImageNetPreprocessor::ImageNetPreprocessor(float3 mean) :
		CUDAImagePreprocessor() {
	this->mean = mean;
}

ImageNetPreprocessor::ImageNetPreprocessor(float mean) :
		CUDAImagePreprocessor() {
	this->mean.x = mean;
	this->mean.y = mean;
	this->mean.z = mean;
}


ImageNetPreprocessor::~ImageNetPreprocessor() {
}


float* ImageNetPreprocessor::RBGAtoBGR(size_t inputWidth, size_t inputHeight,
		size_t outputWidth, size_t outputHeight) {

	float* preprocessOutput = (float*) outputCache.getCUDAAlloc(
			outputWidth * outputHeight * sizeof(float));

	if (mean.x == 0 && mean.y == 0.0 && mean.z == 0.0) {
		cudaPreImageNet(
				(float4*) inputCache.getCUDAAlloc(
						inputWidth * inputHeight * sizeof(float4)), inputWidth,
				inputHeight, preprocessOutput, outputWidth, outputHeight);
	} else {
		cudaPreImageNetMean(
				(float4*) inputCache.getCUDAAlloc(
						inputWidth * inputHeight * sizeof(float4)), inputWidth,
				inputHeight, preprocessOutput, outputWidth, outputHeight, mean);
	}

	return preprocessOutput;
}

} /* namespace jetson_tensorrt */

