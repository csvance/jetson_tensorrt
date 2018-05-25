/**
 * @file	TensorRTEngine.h
 * @author	Carroll Vance
 * @brief	Abstract class which loads and manages a TensorRT graph
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

#ifndef RTENGINE_H_
#define RTENGINE_H_

#include <string>
#include <vector>

#include "host_common.h"

class TensorRTEngine {
public:
	TensorRTEngine();
	virtual ~TensorRTEngine();

	std::vector<std::vector<void*>> predict(std::vector<std::vector<void*>>);

	string engineSummary();

	virtual void addInput(string, nvinfer1::DimsCHW, size_t);
	virtual void addOutput(string, nvinfer1::Dims, size_t);

	int maxBatchSize;
	int numBindings;

protected:
	ICudaEngine* engine;
	IExecutionContext* context;
	Logger logger;

	vector<size_t> networkInputs;
	vector<size_t> networkOutputs;

	void allocGPUBuffer();
	void freeGPUBuffer();

private:
	std::vector<void*> GPU_Buffers;

};

#endif
