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

#include "NetworkOutput.h"
#include "NetworkInput.h"

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = " " + std::string(message);            					\
        logger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

/**
 * @brief Abstract base class which loads and manages a TensorRT model which hides device/host memory management
 */
class TensorRTEngine {
public:
	TensorRTEngine();
	virtual ~TensorRTEngine();

	std::vector<std::vector<void*>> predict(std::vector<std::vector<void*>>);

	bool loadCache(std::string, size_t);
	bool saveCache(std::string);

	string engineSummary();

	virtual void addInput(std::string, nvinfer1::Dims, size_t) = 0;
	virtual void addOutput(std::string, nvinfer1::Dims, size_t) = 0;

	int maxBatchSize;
	int numBindings;

protected:
	ICudaEngine* engine;
	IExecutionContext* context;
	Logger logger;

	vector<NetworkInput> networkInputs;
	vector<NetworkOutput> networkOutputs;

	void allocGPUBuffer();
	void freeGPUBuffer();

private:
	std::vector<void*> GPU_Buffers;

};

/**
 * @brief Logger for GIE info/warning/errors
 */
class Logger: public nvinfer1::ILogger {
public:
	void log(nvinfer1::ILogger::Severity, const char*) override;
};

#endif
