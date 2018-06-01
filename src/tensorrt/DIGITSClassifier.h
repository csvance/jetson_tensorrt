/**
 * @file	DIGITSClassifier.h
 * @author	Carroll Vance
 * @brief	Loads and manages a DIGITS ImageNet graph with TensorRT
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

#ifndef CLASSIFICATIONRTENGINE_H_
#define CLASSIFICATIONRTENGINE_H_

#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUtils.h"

#include "CaffeRTEngine.h"
#include "ImageNetPreprocessor.h"

namespace jetson_tensorrt {

/**
 * @brief	Loads and manages a DIGITS ImageNet graph with TensorRT
 */
class DIGITSClassifier: public CaffeRTEngine {
public:
	DIGITSClassifier(std::string, std::string, std::string =
			"classification.tensorcache", size_t = CHANNELS_BGR, size_t = 224,
			size_t = 224, size_t = 1, size_t = 1000, float3 imageNetMean = {
					0.0, 0.0, 0.0 }, nvinfer1::DataType =
					nvinfer1::DataType::kFLOAT, size_t = (1 << 30));
	virtual ~DIGITSClassifier();

	float* classifyRBGA(float*, size_t, size_t);

	static const size_t CHANNELS_GREYSCALE = 1;
	static const size_t CHANNELS_BGR = 3;

	size_t modelWidth;
	size_t modelHeight;
	size_t modelDepth;

private:
	static const std::string INPUT_NAME;
	static const std::string OUTPUT_NAME;

	ImageNetPreprocessor preprocessor;

};

}

#endif /* CLASSIFICATIONRTENGINE_H_ */
