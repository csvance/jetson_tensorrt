/**
 * @file	TensorflowRTEngine.h
 * @author	Carroll Vance
 * @brief	Loads and manages a Tensorflow graph with TensorRT
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

#ifndef TFRTENGINE_H_
#define TFRTENGINE_H_

#include <string>
#include <vector>

#include "NvUffParser.h"

#include "TensorRTEngine.h"

namespace jetson_tensorrt {

/**
 * @brief Loads and manages a Tensorflow network in TensorRT
 */
class TensorflowRTEngine: public TensorRTEngine {
public:
	TensorflowRTEngine();
	virtual ~TensorflowRTEngine();

	void loadModel(std::string, size_t = 1, nvinfer1::DataType =
			nvinfer1::DataType::kFLOAT, size_t = (1 << 30));

	void addInput(std::string, nvinfer1::Dims, size_t);
	void addOutput(std::string, nvinfer1::Dims dims, size_t);

private:
	nvuffparser::IUffParser* parser;

};

}

#endif /* TFRTENGINE_H_ */
