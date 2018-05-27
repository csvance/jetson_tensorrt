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
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvUtils.h"

#include "NetworkIO.h"

namespace jetson_tensorrt{


/**
 * @brief Logger for GIE info/warning/errors
 */
class Logger: public nvinfer1::ILogger {
public:
	void log(nvinfer1::ILogger::Severity, const char*) override;
};

/**
 * @brief Abstract base class which loads and manages a TensorRT model which hides device/host memory management
 */
class TensorRTEngine {
public:
	TensorRTEngine();
	virtual ~TensorRTEngine();

	std::vector<std::vector<void*>> predict(std::vector<std::vector<void*>>, bool=false);

	void loadCache(std::string, size_t=1);
	void saveCache(std::string);

	std::string engineSummary();

	virtual void addInput(std::string, nvinfer1::Dims, size_t) = 0;
	virtual void addOutput(std::string, nvinfer1::Dims, size_t) = 0;

	int maxBatchSize;
	int numBindings;

protected:
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;
	Logger logger;

	std::vector<NetworkInput> networkInputs;
	std::vector<NetworkOutput> networkOutputs;

	void allocGPUBuffer();
	void freeGPUBuffer();

private:
	std::vector<void*> preAllocatedGPUBuffers;

};

}
#endif
