/**
 * @file	TensorRTEngine.cpp
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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <string>
#include <sstream>

#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"

using namespace nvuffparser;
using namespace nvinfer1;

#include "TensorRTEngine.h"
#include "host_common.h"


/**
 * @brief	Creates and manages a new instance of TensorRTEngine
 */
TensorRTEngine::TensorRTEngine() {
	numBindings = 0;
	maxBatchSize = 0;
	context = NULL;
	engine = NULL;
}

/**
 * @brief	TensorRTEngine destructor
 */
TensorRTEngine::~TensorRTEngine() {
	/* Clean up */
	context->destroy();
	engine->destroy();
	shutdownProtobufLibrary();
}

/**
 * @brief	Allocates the buffers required to copy batches to the GPU
 * @usage	Should be called by a subclass after determining the number and size of all graph inputs and outputs
 */
void TensorRTEngine::allocGPUBuffer() {
	int stepSize = networkInputs.size() + networkOutputs.size();

	GPU_Buffers = vector<void*>(maxBatchSize * stepSize);

	/* Allocate GPU Input memory and move input data to it for each batch*/
	for (int b = 0; b < maxBatchSize; b++) {

		/* Allocate GPU Input memory */
		int bindingIdx = 0;
		for (int i = 0; i < networkInputs.size(); i++) {
			size_t inputSize = networkInputs[i];

			GPU_Buffers[bindingIdx + b * stepSize] = safeCudaMalloc(inputSize);
			bindingIdx++;
		}

		/* Allocate GPU Output memory */
		for (int i = 0; i < networkOutputs.size(); i++) {
			size_t outputSize = networkOutputs[i];

			GPU_Buffers[bindingIdx + b * stepSize] = safeCudaMalloc(outputSize);
			bindingIdx++;
		}

	}
}

/**
 * @brief	Frees buffers required to copy batches to the GPU
 * @usage	Should be called by subclass destructor
 */
void TensorRTEngine::freeGPUBuffer() {
	/* Move outputs back to host memory for each batch */
	for (int b = 0; b < GPU_Buffers.size(); b++)
		CHECK(cudaFree(GPU_Buffers[b]));
}


/**
 * @brief	Does a forward pass of the neural network loaded in TensorRT
 * @usage	Should be called after loading the graph and calling allocGPUBuffer()
 * @param	batchInputs Raw graph inputs, indexed by [batchIndex][inputIndex]
 * @return	Raw graph outputs, indexed by [batchIndex][outputNumber]
 */
std::vector<std::vector<void*>> TensorRTEngine::predict(
		std::vector<std::vector<void*>> batchInputs) {

	assert(batchInputs.size() <= maxBatchSize);

	int batchCount = batchInputs.size();
	int stepSize = networkInputs.size() + networkOutputs.size();

	/* Copy batch to GPU */
	for (int b = 0; b < batchCount; b++) {

		int bindingIdx = 0;
		for (int i = 0; i < networkInputs.size(); i++) {
			size_t inputSize = networkInputs[i];

			CHECK(
					cudaMemcpy(GPU_Buffers[bindingIdx + b * stepSize],
							batchInputs[b][i], inputSize,
							cudaMemcpyHostToDevice));
			bindingIdx++;
		}

	}

	/* Do the inference */
	assert(context->execute(batchCount, &GPU_Buffers[0]) == true);

	std::vector<std::vector<void*>> Output_Buffers(batchCount);

	/* Move outputs back to host memory for each batch */
	for (int b = 0; b < batchCount; b++) {

		int bindingIdx = batchInputs[b].size();
		for (int i = 0; i < networkOutputs.size(); i++) {
			size_t outputSize = networkOutputs[i];

			/* Allocate a host buffer for the network output */
			Output_Buffers[b].push_back(new unsigned char[outputSize]);

			CHECK(
					cudaMemcpy(Output_Buffers[b][i],
							GPU_Buffers[bindingIdx + b * stepSize], outputSize,
							cudaMemcpyDeviceToHost));

			bindingIdx++;
		}

	}

	return Output_Buffers;

}

/**
 * @brief	Returns a summary of the loaded network, inputs, and outputs
 * @return	String containing the summary
 */
string TensorRTEngine::engineSummary() {

	std::stringstream summary;

	for (int i = 0; i < numBindings; ++i) {

		Dims dims = engine->getBindingDimensions(i);
		DataType dtype = engine->getBindingDataType(i);

		summary << "--Binding " << i << "--" << std::endl;
		if (engine->bindingIsInput(i))
			summary << "Type: Input";
		else
			summary << "Type: Output";
		summary << " DataType: ";
		if (dtype == DataType::kFLOAT)
			summary << "kFLOAT";
		else if (dtype == DataType::kHALF)
			summary << "kHALF";
		else if (dtype == DataType::kINT8)
			summary << "kINT8";

		summary << " Dims: (";
		for (int j = 0; j < dims.nbDims; j++)
			summary << dims.d[j] << ",";
		summary << ")" << std::endl;

	}
	return summary.str();

}
