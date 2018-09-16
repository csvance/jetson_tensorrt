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
#include "RTCommon.h"

namespace jetson_tensorrt {

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

	/**
	 * @brief	Creates and manages a new instance of TensorRTEngine
	 */
	TensorRTEngine();

	/**
	 * @brief	TensorRTEngine destructor
	 */
	virtual ~TensorRTEngine();

	/**
	 * @brief	Does a forward pass of the neural network loaded in TensorRT
	 * @usage	Should be called after loading the graph and calling allocGPUBuffer()
	 * @param	inputs Graph inputs on either the device or host, indexed by [batchIndex][inputIndex]
	 * @param	copyOutputToHost	Controls whether the outputs are copied back to host memory (true) or left in device memory (false) after executing the prediction
	 * @return	A LocatedResult of outputs, indexed by [batchIndex][outputNumber]
	 */
	LocatedExecutionMemory predict(LocatedExecutionMemory& inputs, bool copyOutputToHost = true);

	/**
	 * @brief	Quick load the TensorRT optimized network
	 * @usage	Should be called after addInput and addOutput without calling loadModel
	 * @param	cachePath	Path to the network cache file
	 * @param	maxBatchSize	The max batch size of the saved network. If the batch size needs to be changed, the network should be rebuilt with the new size and not simply changed here.
	 */
	void loadCache(std::string cachePath, size_t maxBatchSize = 1);

	/**
	 * @brief	Save the TensorRT optimized network for quick loading in the future
	 * @usage	Should be called after loadModel()
	 * @param	cachePath	Path to the network cache file
	 */
	void saveCache(std::string cachePath);

	/**
	 * @brief	Returns a summary of the loaded network, inputs, and outputs
	 * @return	String containing the summary
	 */
	std::string engineSummary();

	/**
	 * @brief	Abstract method which registers an input to the network.
	 * @param	layer	The name of the input layer (i.e. "input_1")
	 * @param	dims	Dimensions of the input layer. Must be in CHW format. Ex: (3, 640, 480)
	 * @param	eleSize	Size of each element in bytes
	 */
	virtual void addInput(std::string layerName, nvinfer1::Dims dims, size_t eleSize) = 0;

	/**
	 * @brief	Abstract method which registers an output to the network.
	 * @param	layer	The name of the input layer (i.e. "input_1")
	 * @param	dims	Dimension of outputs
	 * @param	eleSize	Size of each element in bytes
	 */
	virtual void addOutput(std::string layerName, nvinfer1::Dims dims, size_t eleSize) = 0;

	int maxBatchSize;
	int numBindings;

	nvinfer1::DataType dataType;

protected:
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;
	Logger logger;

	std::vector<NetworkInput> networkInputs;
	std::vector<NetworkOutput> networkOutputs;

private:
	std::vector<void*> preAllocatedGPUBuffers;

	/**
	 * @brief	Allocates the buffers required to copy batches to the GPU
	 * @usage	Should be called before the first prediction from host memory
	 */
	void allocGPUBuffer();

	/**
	 * @brief	Frees buffers required to copy batches to the GPU
	 */
	void freeGPUBuffer();

	bool gpuBufferPreAllocated;

};

}
#endif
